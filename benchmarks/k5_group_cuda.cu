// T4-only experimental grouped TK k5 prefill. No format conversion/cache.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

template<int MODE>
__global__ void k5_group_cuda(const signed char *codes, const short *scales,
                            const half *x, half *y, int m, int n, int k,
                            int cs, int ss, int xs, float anchor, float levels,
                            int coded, int logarithmic) {
    constexpr int B = 32, G = 128;
    __shared__ __align__(32) half ax[B * G];
    __shared__ __align__(32) half wh[B * G];
    __shared__ __align__(32) signed char wi[B * G];
    __shared__ __align__(32) signed char aq[B * G], ar[B * G];
    __shared__ __align__(32) float partial[4][256];
    __shared__ __align__(32) int residual_partial[4][256];
    __shared__ float sw[B], sx[B], sr[B];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / 2, wn = warp % 2;
    const int mb = blockIdx.x * B, nb = blockIdx.y * B;
    float sum[8] = {};
    for (int g = 0; g < k / G; ++g) {
        for (int i = tid; i < B * G; i += 128) {
            int r = i / G, c = i % G;
            ax[i] = mb + r < m ? x[(mb + r) * xs + g * G + c] : __float2half(0);
            signed char q = nb + r < n ? codes[(nb + r) * cs + g * G + c] : 0;
            if constexpr (MODE == 0) wh[i] = __int2half_rn((int)q);
            else wi[i] = q;
        }
        if (tid < B) {
            float s = nb + tid < n ? (float)scales[(nb + tid) * ss + g] : 0;
            if (coded) {
                s = (s / levels) * anchor;
                if (logarithmic) s = exp2f(s);
            }
            sw[tid] = s / 121.0f;
        }
        __syncthreads();
        if constexpr (MODE != 0) {
            for (int r = warp; r < B; r += 4) {
                float vmax = 0;
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    vmax = fmaxf(vmax, fabsf(__half2float(ax[r * G + lane + t * 32])));
                #pragma unroll
                for (int d = 16; d; d /= 2)
                    vmax = fmaxf(vmax, __shfl_down_sync(0xffffffff, vmax, d));
                vmax = __shfl_sync(0xffffffff, vmax, 0);
                float a = vmax > 0 ? vmax / 127.0f : 1.0f;
                if (lane == 0) sx[r] = a;
                float rmax = 0;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int i = r * G + lane + t * 32;
                    float v = __half2float(ax[i]);
                    int q = max(-127, min(127, __float2int_rn(v / a)));
                    aq[i] = (signed char)q;
                    rmax = fmaxf(rmax, fabsf(v - (float)q * a));
                }
                if constexpr (MODE == 2) {
                    #pragma unroll
                    for (int d = 16; d; d /= 2)
                        rmax = fmaxf(rmax, __shfl_down_sync(0xffffffff, rmax, d));
                    rmax = __shfl_sync(0xffffffff, rmax, 0);
                    float a2 = rmax > 0 ? rmax / 127.0f : 1.0f;
                    if (lane == 0) sr[r] = a2;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int i = r * G + lane + t * 32;
                        float v = __half2float(ax[i]) - (float)aq[i] * a;
                        ar[i] = (signed char)max(-127, min(127, __float2int_rn(v / a2)));
                    }
                }
            }
        }
        __syncthreads();
        if constexpr (MODE == 0) {
            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int t = 0; t < G; t += 16) {
                wmma::load_matrix_sync(a, ax + wm * 16 * G + t, G);
                wmma::load_matrix_sync(b, wh + wn * 16 * G + t, G);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(partial[warp], acc, 16, wmma::mem_row_major);
        } else {
            wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::col_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,int> acc, racc;
            wmma::fill_fragment(acc, 0);
            if constexpr (MODE == 2) wmma::fill_fragment(racc, 0);
            #pragma unroll
            for (int t = 0; t < G; t += 16) {
                wmma::load_matrix_sync(a, aq + wm * 16 * G + t, G);
                wmma::load_matrix_sync(b, wi + wn * 16 * G + t, G);
                wmma::mma_sync(acc, a, b, acc);
                if constexpr (MODE == 2) {
                    wmma::load_matrix_sync(a, ar + wm * 16 * G + t, G);
                    wmma::mma_sync(racc, a, b, racc);
                }
            }
            wmma::store_matrix_sync(reinterpret_cast<int*>(partial[warp]), acc, 16, wmma::mem_row_major);
            if constexpr (MODE == 2)
                wmma::store_matrix_sync(residual_partial[warp], racc, 16, wmma::mem_row_major);
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int i = tid + j * 128, r = i / B, c = i % B;
            int w = (r / 16) * 2 + c / 16, off = (r % 16) * 16 + c % 16;
            float v;
            if constexpr (MODE == 0) v = partial[w][off];
            else {
                v = (float)reinterpret_cast<int*>(partial[w])[off] * sx[r];
                if constexpr (MODE == 2) v += (float)residual_partial[w][off] * sr[r];
            }
            sum[j] += v * sw[c];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int i = tid + j * 128, r = mb + i / B, c = nb + i % B;
        if (r < m && c < n) y[r * n + c] = __float2half_rn(sum[j]);
    }
}

extern "C" int launch_k5_group(int mode, const void *c, const void *s, const void *x, void *y,
                              int m, int n, int k, int cs, int ss, int xs,
                              float anchor, float levels, int coded, int logarithmic, void *stream) {
    dim3 grid((m + 31) / 32, (n + 31) / 32);
    auto st = reinterpret_cast<cudaStream_t>(stream);
    #define ARGS reinterpret_cast<const signed char*>(c), reinterpret_cast<const short*>(s), \
                 reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y), \
                 m,n,k,cs,ss,xs,anchor,levels,coded,logarithmic
    if (mode == 0) k5_group_cuda<0><<<grid,128,0,st>>>(ARGS);
    else if (mode == 1) k5_group_cuda<1><<<grid,128,0,st>>>(ARGS);
    else if (mode == 2) k5_group_cuda<2><<<grid,128,0,st>>>(ARGS);
    else return (int)cudaErrorInvalidValue;
    #undef ARGS
    return (int)cudaGetLastError();
}
