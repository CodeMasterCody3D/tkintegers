// One-dot INT8 K5 prefill kernel for sm89 (IMMA): the K5 code IS the s8 weight.
// The activation is quantized once per call (separate small kernel) to s8 with an
// affine scale per (row, SUB-element sub-group):  x ~ a * (A + c),  A in s8, c integer.
// Per 128-group g with SUB-wide sub-groups j, every sub-group dot is an exact s32
// tensor-core result D = dot(A, q); R = sum of codes in the sub-group (exact int16,
// precomputed from the unchanged codes). The FP32 flush computes
//   y[m,n] = sum_g S[n,g] * sum_j a[m,gj] * ( D[m,n,gj] + c[m,gj] * R[n,gj] )
// with S the unchanged g128 integer-log scale, so the weights are never converted.
// SUB=32 uses mma.m16n8k32.s8 (4 partials per group); SUB=16 uses mma.m16n8k16.s8
// (8 partials per group). Shared layout, tile (64x64, 8 warps, WN=32), XOR chunk
// permutation and ldmatrix fragment mapping follow k5_group_mma_swizzle_cuda.cu.
// Compile with --fmad=false; fmaf() is written explicitly where one rounding is meant.
// New file: no existing kernel, binary or default is changed.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int BM = 64, BN = 64, G = 128, STRIDE = 128, THREADS = 256, WN = 32;

template<int SUB> struct Layout {
    static constexpr int NSUB = G / SUB;
    static constexpr int A_OFF = 0;                         // s8 [BM][STRIDE], permuted 16-byte chunks
    static constexpr int W_OFF = BM * STRIDE;               // s8 [BN][STRIDE], permuted 16-byte chunks
    static constexpr int SW_OFF = W_OFF + BN * STRIDE;      // float [BN]      group scale / 121
    static constexpr int R_OFF = SW_OFF + BN * 4;           // float [NSUB][BN] code sums
    static constexpr int AC_OFF = R_OFF + NSUB * BN * 4;    // float2 [BM][NSUB] (a, c)
    static constexpr int BYTES = AC_OFF + BM * NSUB * 8;
};

__device__ __forceinline__ int sidx(int row, int col) {   // col is a multiple of 16
    return row * STRIDE + (col ^ ((row & 7) * 16));
}

__device__ __forceinline__ void ldsm4(unsigned (&v)[4], const void *ptr) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
}

__device__ __forceinline__ void imma32(int *d, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void imma16(int *d, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Exact s32 -> f32 for |v| < 2^22 (sub-group dots are bounded by 128*121*32 = 495,616).
__device__ __forceinline__ float exact_float(int v) {
    return __int_as_float(v + 0x4B400000) - 12582912.0f;
}

// sumg[h][jv] += a[row] * (D + c[row] * R[col]) for this thread's 2 rows x 8 columns.
// jv: 0 -> (row0,col0), 1 -> (row0,col1), 2 -> (row1,col0), 3 -> (row1,col1).
template<int NSUB>
__device__ __forceinline__ void flush(float (&sumg)[4][4], const int (&acc)[4][4], int j,
                                      const float2 *acs, const float *Rs, int wm, int wn, int lane) {
    const float2 ac0 = acs[(wm * 16 + lane / 4) * NSUB + j];
    const float2 ac1 = acs[(wm * 16 + lane / 4 + 8) * NSUB + j];
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float2 R = *reinterpret_cast<const float2*>(Rs + j * BN + wn * WN + h * 8 + (lane % 4) * 2);
        const float f0 = fmaf(ac0.y, R.x, exact_float(acc[h][0]));
        const float f1 = fmaf(ac0.y, R.y, exact_float(acc[h][1]));
        const float f2 = fmaf(ac1.y, R.x, exact_float(acc[h][2]));
        const float f3 = fmaf(ac1.y, R.y, exact_float(acc[h][3]));
        sumg[h][0] = fmaf(ac0.x, f0, sumg[h][0]);
        sumg[h][1] = fmaf(ac0.x, f1, sumg[h][1]);
        sumg[h][2] = fmaf(ac1.x, f2, sumg[h][2]);
        sumg[h][3] = fmaf(ac1.x, f3, sumg[h][3]);
    }
}

template<int SUB>
__global__ void __launch_bounds__(THREADS)
k5_group_imma_cuda(const int8_t *codes, const short *scales, const int8_t *aq, const float2 *ac,
                   const short *rsum, half *y, int m, int n, int k, int cs, int ss,
                   float anchor, float levels, int coded, int logarithmic) {
    using L = Layout<SUB>;
    constexpr int NSUB = L::NSUB;
    extern __shared__ __align__(16) unsigned char storage[];
    int8_t *As = reinterpret_cast<int8_t*>(storage + L::A_OFF);
    int8_t *Ws = reinterpret_cast<int8_t*>(storage + L::W_OFF);
    float *sw = reinterpret_cast<float*>(storage + L::SW_OFF);
    float *Rs = reinterpret_cast<float*>(storage + L::R_OFF);
    float2 *acs = reinterpret_cast<float2*>(storage + L::AC_OFF);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / 2, wn = warp % 2;
    const int mb = blockIdx.x * BM, nb = blockIdx.y * BN;
    const int nsub_k = k / SUB;
    float sum[4][4] = {};
    for (int g = 0; g < k / G; ++g) {
        #pragma unroll
        for (int i = tid; i < BM * (G / 16); i += THREADS) {
            const int r = i / (G / 16), c = (i % (G / 16)) * 16;
            const uint4 a = mb + r < m ? *reinterpret_cast<const uint4*>(aq + (size_t)(mb + r) * k + g * G + c)
                                      : make_uint4(0, 0, 0, 0);
            const uint4 w = nb + r < n ? *reinterpret_cast<const uint4*>(codes + (size_t)(nb + r) * cs + g * G + c)
                                      : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(As + sidx(r, c)) = a;
            *reinterpret_cast<uint4*>(Ws + sidx(r, c)) = w;
        }
        if (tid < BN) {
            float s = nb + tid < n ? (float)scales[(nb + tid) * ss + g] : 0.0f;
            if (coded) {
                s = (s / levels) * anchor;
                if (logarithmic) s = exp2f(s);
            }
            sw[tid] = s / 121.0f;
            #pragma unroll
            for (int j = 0; j < NSUB; ++j)
                Rs[j * BN + tid] = nb + tid < n ? (float)rsum[(size_t)(nb + tid) * nsub_k + g * NSUB + j] : 0.0f;
        } else if (tid < BN + BM) {
            const int r = tid - BN;
            #pragma unroll
            for (int j = 0; j < NSUB; ++j)
                acs[r * NSUB + j] = mb + r < m ? ac[(size_t)(mb + r) * nsub_k + g * NSUB + j] : make_float2(0.0f, 0.0f);
        }
        __syncthreads();
        float sumg[4][4] = {};
        #pragma unroll
        for (int t = 0; t < G; t += 32) {
            unsigned a[4], b[2][4];
            // Matrices 0..3: rows 0-7 / 8-15 at k bytes 0-15, then rows 0-7 / 8-15 at k bytes 16-31.
            ldsm4(a, As + sidx(wm * 16 + lane % 16, t + (lane / 16) * 16));
            ldsm4(b[0], Ws + sidx(wn * WN + lane % 16, t + (lane / 16) * 16));
            ldsm4(b[1], Ws + sidx(wn * WN + 16 + lane % 16, t + (lane / 16) * 16));
            if constexpr (SUB == 32) {
                int acc[4][4] = {};
                imma32(acc[0], a, b[0][0], b[0][2]);
                imma32(acc[1], a, b[0][1], b[0][3]);
                imma32(acc[2], a, b[1][0], b[1][2]);
                imma32(acc[3], a, b[1][1], b[1][3]);
                flush<NSUB>(sumg, acc, t / 32, acs, Rs, wm, wn, lane);
            } else {
                int acc[4][4] = {};
                imma16(acc[0], a[0], a[1], b[0][0]);
                imma16(acc[1], a[0], a[1], b[0][1]);
                imma16(acc[2], a[0], a[1], b[1][0]);
                imma16(acc[3], a[0], a[1], b[1][1]);
                flush<NSUB>(sumg, acc, t / 16, acs, Rs, wm, wn, lane);
                int acc2[4][4] = {};
                imma16(acc2[0], a[2], a[3], b[0][2]);
                imma16(acc2[1], a[2], a[3], b[0][3]);
                imma16(acc2[2], a[2], a[3], b[1][2]);
                imma16(acc2[3], a[2], a[3], b[1][3]);
                flush<NSUB>(sumg, acc2, t / 16 + 1, acs, Rs, wm, wn, lane);
            }
        }
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            #pragma unroll
            for (int jv = 0; jv < 4; ++jv) {
                const int c = wn * WN + h * 8 + (lane % 4) * 2 + jv % 2;
                sum[h][jv] += sumg[h][jv] * sw[c];   // separate multiply and add (--fmad=false)
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        #pragma unroll
        for (int jv = 0; jv < 4; ++jv) {
            const int r = mb + wm * 16 + lane / 4 + (jv / 2) * 8;
            const int c = nb + wn * WN + h * 8 + (lane % 4) * 2 + jv % 2;
            if (r < m && c < n) y[(size_t)r * n + c] = __float2half_rn(sum[h][jv]);
        }
    }
}

// One thread per (row, sub-group): x ~ a * (A + c), a = max((hi-lo)/255, amax/32767),
// c = round(lo/a) + 128, A = round(x/a) - c in [-128, 127]. The floor on a only
// affects nearly constant sub-groups (keeps c small) and costs at most amax/65534 per element.
template<int SUB>
__global__ void k5_quantize_activation(const half *x, int8_t *aq, float2 *ac, int m, int k, int xs) {
    const int nsub = k / SUB;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * nsub) return;
    const int r = idx / nsub, s = idx % nsub;
    const half *p = x + (size_t)r * xs + s * SUB;
    float v[SUB];
    #pragma unroll
    for (int i = 0; i < SUB / 8; ++i) {
        const uint4 u = *reinterpret_cast<const uint4*>(p + i * 8);
        const half *h = reinterpret_cast<const half*>(&u);
        #pragma unroll
        for (int e = 0; e < 8; ++e) v[i * 8 + e] = __half2float(h[e]);
    }
    float lo = v[0], hi = v[0];
    #pragma unroll
    for (int i = 1; i < SUB; ++i) { lo = fminf(lo, v[i]); hi = fmaxf(hi, v[i]); }
    float a = fmaxf((hi - lo) / 255.0f, fmaxf(fabsf(lo), fabsf(hi)) / 32767.0f);
    int z = 0;
    if (a > 0.0f) z = __float2int_rn(lo / a) + 128; else a = 1.0f;
    __align__(16) int8_t q[SUB];
    #pragma unroll
    for (int i = 0; i < SUB; ++i) {
        const int t = __float2int_rn(v[i] / a) - z;
        q[i] = (int8_t)max(-128, min(127, t));
    }
    #pragma unroll
    for (int i = 0; i < SUB / 16; ++i)
        *reinterpret_cast<uint4*>(aq + (size_t)r * k + s * SUB + i * 16) = *reinterpret_cast<const uint4*>(q + i * 16);
    ac[idx] = make_float2(a, (float)z);
}

template<int SUB>
static int initialize(int *info) {
    cudaError_t rc = cudaFuncSetAttribute(k5_group_imma_cuda<SUB>, cudaFuncAttributeMaxDynamicSharedMemorySize, Layout<SUB>::BYTES);
    if (rc != cudaSuccess) return (int)rc;
    cudaFuncAttributes attr;
    rc = cudaFuncGetAttributes(&attr, k5_group_imma_cuda<SUB>);
    if (rc != cudaSuccess) return (int)rc;
    int active;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, k5_group_imma_cuda<SUB>, THREADS, Layout<SUB>::BYTES);
    if (rc != cudaSuccess) return (int)rc;
    info[0] = BM; info[1] = BN; info[2] = THREADS; info[3] = Layout<SUB>::BYTES;
    info[4] = attr.numRegs; info[5] = (int)attr.localSizeBytes; info[6] = active;
    cudaFuncAttributes qattr;
    rc = cudaFuncGetAttributes(&qattr, k5_quantize_activation<SUB>);
    if (rc != cudaSuccess) return (int)rc;
    info[7] = qattr.numRegs; info[8] = (int)qattr.localSizeBytes;
    return active > 0 ? (int)cudaSuccess : (int)cudaErrorInvalidConfiguration;
}

// variant 0: SUB=32 (m16n8k32); variant 1: SUB=16 (m16n8k16). info has 9 ints.
extern "C" int initialize_k5_imma(int variant, int *info) {
    switch (variant) {
        case 0: return initialize<32>(info);
        case 1: return initialize<16>(info);
        default: return (int)cudaErrorInvalidValue;
    }
}

// codes s8 [n,k] (row stride cs), scales s16 [n,k/128] (row stride ss), x f16 [m,k] (row stride xs),
// rsum s16 [n,k/SUB] contiguous, aq s8 [m,k] and ac float2 [m,k/SUB] scratch (contiguous), y f16 [m,n].
extern "C" int launch_k5_imma(int variant, const void *codes, const void *scales, const void *x, const void *rsum,
                              void *aq, void *ac, void *y, int m, int n, int k, int cs, int ss, int xs,
                              float anchor, float levels, int coded, int logarithmic, void *stream) {
    if (k % G || cs % 16 || xs % 8 || m < 1 || n < 1 ||
        reinterpret_cast<uintptr_t>(codes) % 16 || reinterpret_cast<uintptr_t>(x) % 16 ||
        reinterpret_cast<uintptr_t>(aq) % 16 || reinterpret_cast<uintptr_t>(ac) % 8 ||
        reinterpret_cast<uintptr_t>(rsum) % 2)
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
    #define ARGS reinterpret_cast<const int8_t*>(codes), reinterpret_cast<const short*>(scales), \
                 reinterpret_cast<const int8_t*>(aq), reinterpret_cast<const float2*>(ac), \
                 reinterpret_cast<const short*>(rsum), reinterpret_cast<half*>(y), \
                 m, n, k, cs, ss, anchor, levels, coded, logarithmic
    switch (variant) {
        case 0:
            k5_quantize_activation<32><<<(m * (k / 32) + 127) / 128, 128, 0, st>>>(
                reinterpret_cast<const half*>(x), reinterpret_cast<int8_t*>(aq), reinterpret_cast<float2*>(ac), m, k, xs);
            k5_group_imma_cuda<32><<<grid, THREADS, Layout<32>::BYTES, st>>>(ARGS);
            break;
        case 1:
            k5_quantize_activation<16><<<(m * (k / 16) + 127) / 128, 128, 0, st>>>(
                reinterpret_cast<const half*>(x), reinterpret_cast<int8_t*>(aq), reinterpret_cast<float2*>(ac), m, k, xs);
            k5_group_imma_cuda<16><<<grid, THREADS, Layout<16>::BYTES, st>>>(ARGS);
            break;
        default: return (int)cudaErrorInvalidValue;
    }
    #undef ARGS
    return (int)cudaGetLastError();
}
