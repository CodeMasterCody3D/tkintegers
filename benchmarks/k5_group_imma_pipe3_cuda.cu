// Pipelined one-dot INT8 K5 prefill kernel, third version (sm89 IMMA + cp.async).
// Same algebra as k5_group_imma_cuda.cu / pipe v2:
//   y[m,n] = sum_g S[n,g] * sum_j a[m,gj] * ( D[m,n,gj] + c[m,gj] * R[n,gj] ).
// Changes versus pipe v2 (which was issue-bound: ~25M of ~35M warp-instructions were
// the per-sub-group flush, and the tensor pipe was ~20 percent busy):
//   * 4 warps per 64x64 CTA tile, each warp a 32x32 sub-tile: one ldmatrix.x4 now
//     feeds twice the IMMAs (shared-memory traffic per MAC halved);
//   * the IMMA C operand is the constant 0x4B400000 (12582912) in separate registers,
//     so every accumulator leaves the tensor core as float_bits(12582912 + D) for free;
//   * EXACT=1 flush (3 ops/value): f = fmaf(c, R, bits) - 12582912; sumg = fmaf(a, f, sumg).
//     Values are the same quantities as before, within one FP32 rounding.
//   * EXACT=0 flush (2 ops/value): sumg = fmaf(a, fmaf(c, R, bits), sumg) and the bias
//     12582912 * sum_j a_j is removed once per 128-group. This loses the low bits of
//     the per-sub-group sum (about 1e-4 relative) and is gated by the matrix test.
// Compile with --fmad=false; fmaf() written explicitly. New file: nothing existing changed.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int BM = 64, BN = 64, G = 128, KT = 64, THREADS = 128, WM = 32, WN = 32;
constexpr int MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.0f;

template<int SUB> struct Stage {
    static constexpr int NSUB_T = KT / SUB;
    static constexpr int A_OFF = 0;                             // s8 [BM][KT], permuted 16-byte chunks
    static constexpr int W_OFF = BM * KT;                       // s8 [BN][KT]
    static constexpr int R_OFF = W_OFF + BN * KT;               // float [NSUB_T][BN]
    static constexpr int AC_OFF = R_OFF + NSUB_T * BN * 4;      // float2 [BM][NSUB_T]
    static constexpr int S_OFF = AC_OFF + BM * NSUB_T * 8;      // float [BN]
    static constexpr int BYTES = S_OFF + BN * 4;
    static constexpr int A_CHUNKS = BM * KT / 16, W_CHUNKS = BN * KT / 16;
    static constexpr int R_CHUNKS = NSUB_T * BN * 4 / 16, AC_CHUNKS = BM * NSUB_T * 8 / 16;
    static constexpr int REGULAR = A_CHUNKS + W_CHUNKS + R_CHUNKS + AC_CHUNKS;   // 608 / 704
    static constexpr int S_CHUNKS = BN * 4 / 16;                                  // 16
    static constexpr int SLOTS = (REGULAR + THREADS - 1) / THREADS;               // 5 / 6
};

__device__ __forceinline__ int sidx64(int row, int col) {
    return row * KT + (((col >> 4) ^ ((row >> 1) & 3)) << 4);
}
__device__ __forceinline__ void cp_async16(unsigned smem, const void *gmem, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

__device__ __forceinline__ void ldsm4(unsigned (&v)[4], const void *ptr) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
}
// D = C + A*B with C the constant MAGIC in every lane: the result bits read as float(12582912 + D).
__device__ __forceinline__ void imma32m(int *d, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC_I));
}
__device__ __forceinline__ void imma16m(int *d, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC_I));
}

// acc[mt][h][v]: m-tile mt (rows wm*32 + mt*16 + lane/4 (+8 for v>=2)), n-tile h, v as in the PTX layout.
template<int NSUB_T, bool EXACT>
__device__ __forceinline__ void flush(float (&sumg)[2][4][4], float (&asum)[4], const int (&acc)[2][4][4], int j,
                                      const float2 *acs, const float *Rs, int wm, int wn, int lane) {
    float2 ac[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) ac[q] = acs[(wm * WM + q * 8 + lane / 4) * NSUB_T + j];   // rows mt*16 + half*8
    if constexpr (!EXACT) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) asum[q] += ac[q].x;
    }
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float2 R = *reinterpret_cast<const float2*>(Rs + j * BN + wn * WN + h * 8 + (lane % 4) * 2);
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const float2 a0 = ac[mt * 2], a1 = ac[mt * 2 + 1];
            float f0 = fmaf(a0.y, R.x, __int_as_float(acc[mt][h][0]));
            float f1 = fmaf(a0.y, R.y, __int_as_float(acc[mt][h][1]));
            float f2 = fmaf(a1.y, R.x, __int_as_float(acc[mt][h][2]));
            float f3 = fmaf(a1.y, R.y, __int_as_float(acc[mt][h][3]));
            if constexpr (EXACT) { f0 -= MAGIC_F; f1 -= MAGIC_F; f2 -= MAGIC_F; f3 -= MAGIC_F; }
            sumg[mt][h][0] = fmaf(a0.x, f0, sumg[mt][h][0]);
            sumg[mt][h][1] = fmaf(a0.x, f1, sumg[mt][h][1]);
            sumg[mt][h][2] = fmaf(a1.x, f2, sumg[mt][h][2]);
            sumg[mt][h][3] = fmaf(a1.x, f3, sumg[mt][h][3]);
        }
    }
}

struct Desc { const unsigned char *g; unsigned stride; unsigned soff; int bytes; };

template<int SUB, int STAGES, bool EXACT>
__global__ void __launch_bounds__(THREADS, 3)
k5_group_imma_pipe3_cuda(const int8_t *codes, const float *scales_f, const int8_t *aq, const float2 *ac,
                         const float *rsum_f, half *y, int m, int n, int k, int cs) {
    using St = Stage<SUB>;
    constexpr int NSUB_T = St::NSUB_T;
    extern __shared__ __align__(16) unsigned char storage[];
    const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / 2, wn = warp % 2;
    const int mb = blockIdx.x * BM, nb = blockIdx.y * BN;
    const int nsub_k = k / SUB, nstages = k / KT;

    Desc d[St::SLOTS];
    #pragma unroll
    for (int q = 0; q < St::SLOTS; ++q) {
        const int i = tid + q * THREADS;
        d[q].g = nullptr; d[q].stride = 0; d[q].soff = 0; d[q].bytes = 0;
        if (i < St::A_CHUNKS) {
            const int r = i / (KT / 16), c = (i % (KT / 16)) * 16;
            const bool valid = mb + r < m;
            d[q].g = reinterpret_cast<const unsigned char*>(aq + (size_t)(valid ? mb + r : 0) * k + c);
            d[q].stride = KT; d[q].soff = St::A_OFF + sidx64(r, c); d[q].bytes = valid ? 16 : 0;
        } else if (i < St::A_CHUNKS + St::W_CHUNKS) {
            const int p = i - St::A_CHUNKS, r = p / (KT / 16), c = (p % (KT / 16)) * 16;
            const bool valid = nb + r < n;
            d[q].g = reinterpret_cast<const unsigned char*>(codes + (size_t)(valid ? nb + r : 0) * cs + c);
            d[q].stride = KT; d[q].soff = St::W_OFF + sidx64(r, c); d[q].bytes = valid ? 16 : 0;
        } else if (i < St::A_CHUNKS + St::W_CHUNKS + St::R_CHUNKS) {
            const int p = i - St::A_CHUNKS - St::W_CHUNKS, j = p / (BN / 4), c = (p % (BN / 4)) * 4;
            const bool valid = nb + c < n;
            d[q].g = reinterpret_cast<const unsigned char*>(rsum_f + (size_t)j * n + (valid ? nb + c : 0));
            d[q].stride = NSUB_T * n * 4; d[q].soff = St::R_OFF + (j * BN + c) * 4; d[q].bytes = valid ? 16 : 0;
        } else if (i < St::REGULAR) {
            const int p = i - St::A_CHUNKS - St::W_CHUNKS - St::R_CHUNKS, r = p / (NSUB_T / 2), j = (p % (NSUB_T / 2)) * 2;
            const bool valid = mb + r < m;
            d[q].g = reinterpret_cast<const unsigned char*>(ac + (size_t)(valid ? mb + r : 0) * nsub_k + j);
            d[q].stride = NSUB_T * 8; d[q].soff = St::AC_OFF + (r * NSUB_T + j) * 8; d[q].bytes = valid ? 16 : 0;
        }
    }
    const bool s_thread = tid < St::S_CHUNKS;
    const int s_col = tid * 4;
    const bool s_valid = s_thread && nb + s_col < n;
    const unsigned char *s_g = reinterpret_cast<const unsigned char*>(scales_f + (s_valid ? nb + s_col : 0));

    auto issue = [&](int s, int buf) {
        const unsigned base = sbase + buf * St::BYTES;
        #pragma unroll
        for (int q = 0; q < St::SLOTS; ++q) {
            if (tid + q * THREADS < St::REGULAR) {
                cp_async16(base + d[q].soff, d[q].g, d[q].bytes);
                d[q].g += d[q].stride;
            }
        }
        if (s_thread) cp_async16(base + St::S_OFF + s_col * 4, s_g + (size_t)(s / 2) * n * 4, s_valid ? 16 : 0);
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nstages) issue(s, s);
        cp_async_commit();
    }
    float sum[2][4][4] = {}, sumg[2][4][4] = {}, asum[4] = {};
    for (int s = 0; s < nstages; ++s) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (s + STAGES - 1 < nstages) issue(s + STAGES - 1, (s + STAGES - 1) % STAGES);
        cp_async_commit();
        const unsigned char *base = storage + (s % STAGES) * St::BYTES;
        const int8_t *As = reinterpret_cast<const int8_t*>(base + St::A_OFF);
        const int8_t *Ws = reinterpret_cast<const int8_t*>(base + St::W_OFF);
        const float *Rs = reinterpret_cast<const float*>(base + St::R_OFF);
        const float2 *acs = reinterpret_cast<const float2*>(base + St::AC_OFF);
        const float *Ss = reinterpret_cast<const float*>(base + St::S_OFF);
        #pragma unroll
        for (int t = 0; t < KT; t += 32) {
            unsigned a[2][4], b[2][4];
            ldsm4(a[0], As + sidx64(wm * WM + lane % 16, t + (lane / 16) * 16));
            ldsm4(a[1], As + sidx64(wm * WM + 16 + lane % 16, t + (lane / 16) * 16));
            ldsm4(b[0], Ws + sidx64(wn * WN + lane % 16, t + (lane / 16) * 16));
            ldsm4(b[1], Ws + sidx64(wn * WN + 16 + lane % 16, t + (lane / 16) * 16));
            if constexpr (SUB == 32) {
                int acc[2][4][4];
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    imma32m(acc[mt][0], a[mt], b[0][0], b[0][2]);
                    imma32m(acc[mt][1], a[mt], b[0][1], b[0][3]);
                    imma32m(acc[mt][2], a[mt], b[1][0], b[1][2]);
                    imma32m(acc[mt][3], a[mt], b[1][1], b[1][3]);
                }
                flush<NSUB_T, EXACT>(sumg, asum, acc, t / 32, acs, Rs, wm, wn, lane);
            } else {
                int acc[2][4][4];
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    imma16m(acc[mt][0], a[mt][0], a[mt][1], b[0][0]);
                    imma16m(acc[mt][1], a[mt][0], a[mt][1], b[0][1]);
                    imma16m(acc[mt][2], a[mt][0], a[mt][1], b[1][0]);
                    imma16m(acc[mt][3], a[mt][0], a[mt][1], b[1][1]);
                }
                flush<NSUB_T, EXACT>(sumg, asum, acc, t / 16, acs, Rs, wm, wn, lane);
                int acc2[2][4][4];
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    imma16m(acc2[mt][0], a[mt][2], a[mt][3], b[0][2]);
                    imma16m(acc2[mt][1], a[mt][2], a[mt][3], b[0][3]);
                    imma16m(acc2[mt][2], a[mt][2], a[mt][3], b[1][2]);
                    imma16m(acc2[mt][3], a[mt][2], a[mt][3], b[1][3]);
                }
                flush<NSUB_T, EXACT>(sumg, asum, acc2, t / 16 + 1, acs, Rs, wm, wn, lane);
            }
        }
        if (s & 1) {                                       // end of a 128-group
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float2 sc = *reinterpret_cast<const float2*>(Ss + wn * WN + h * 8 + (lane % 4) * 2);
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    float g0 = sumg[mt][h][0], g1 = sumg[mt][h][1], g2 = sumg[mt][h][2], g3 = sumg[mt][h][3];
                    if constexpr (!EXACT) {                // remove the 12582912 * sum_j a_j bias once per group
                        g0 = fmaf(-MAGIC_F, asum[mt * 2], g0);
                        g1 = fmaf(-MAGIC_F, asum[mt * 2], g1);
                        g2 = fmaf(-MAGIC_F, asum[mt * 2 + 1], g2);
                        g3 = fmaf(-MAGIC_F, asum[mt * 2 + 1], g3);
                    }
                    sum[mt][h][0] += g0 * sc.x;             // separate multiply and add (--fmad=false)
                    sum[mt][h][1] += g1 * sc.y;
                    sum[mt][h][2] += g2 * sc.x;
                    sum[mt][h][3] += g3 * sc.y;
                    sumg[mt][h][0] = sumg[mt][h][1] = sumg[mt][h][2] = sumg[mt][h][3] = 0.0f;
                }
            }
            if constexpr (!EXACT) { asum[0] = asum[1] = asum[2] = asum[3] = 0.0f; }
        }
    }
    cp_async_wait<0>();
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            #pragma unroll
            for (int jv = 0; jv < 4; ++jv) {
                const int r = mb + wm * WM + mt * 16 + lane / 4 + (jv / 2) * 8;
                const int c = nb + wn * WN + h * 8 + (lane % 4) * 2 + jv % 2;
                if (r < m && c < n) y[(size_t)r * n + c] = __float2half_rn(sum[mt][h][jv]);
            }
        }
    }
}

__global__ void k5_prepare_scales_pipe3(const short *scales, float *scales_f, int n, int groups, int ss,
                                        float anchor, float levels, int coded, int logarithmic) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * groups) return;
    const int g = idx / n, col = idx % n;
    float s = (float)scales[(size_t)col * ss + g];
    if (coded) {
        s = (s / levels) * anchor;
        if (logarithmic) s = exp2f(s);
    }
    scales_f[(size_t)g * n + col] = s / 121.0f;
}

template<int SUB>
__global__ void k5_quantize_activation_pipe3(const half *x, int8_t *aq, float2 *ac, int m, int k, int xs) {
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

template<int SUB, int STAGES, bool EXACT>
static int initialize(int *info) {
    constexpr int BYTES = Stage<SUB>::BYTES * STAGES;
    cudaError_t rc = cudaFuncSetAttribute(k5_group_imma_pipe3_cuda<SUB, STAGES, EXACT>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
    if (rc != cudaSuccess) return (int)rc;
    cudaFuncAttributes attr;
    rc = cudaFuncGetAttributes(&attr, k5_group_imma_pipe3_cuda<SUB, STAGES, EXACT>);
    if (rc != cudaSuccess) return (int)rc;
    int active;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, k5_group_imma_pipe3_cuda<SUB, STAGES, EXACT>, THREADS, BYTES);
    if (rc != cudaSuccess) return (int)rc;
    info[0] = BM; info[1] = BN; info[2] = THREADS; info[3] = BYTES;
    info[4] = attr.numRegs; info[5] = (int)attr.localSizeBytes; info[6] = active;
    cudaFuncAttributes qattr;
    rc = cudaFuncGetAttributes(&qattr, k5_quantize_activation_pipe3<SUB>);
    if (rc != cudaSuccess) return (int)rc;
    info[7] = qattr.numRegs; info[8] = (int)qattr.localSizeBytes;
    return active > 0 ? (int)cudaSuccess : (int)cudaErrorInvalidConfiguration;
}

// variant = sub * 1000 + stages * 10 + exact
#define VARIANTS(X) X(32, 3, 1) X(32, 4, 1) X(32, 3, 0) X(32, 4, 0) X(32, 6, 0) X(16, 3, 1) X(16, 3, 0)

extern "C" int initialize_k5_imma_pipe3(int variant, int *info) {
    switch (variant) {
        #define CASE(S, T, E) case S * 1000 + T * 10 + E: return initialize<S, T, E>(info);
        VARIANTS(CASE)
        #undef CASE
        default: return (int)cudaErrorInvalidValue;
    }
}

extern "C" int prepare_k5_imma_pipe3(const void *scales, void *scales_f, int n, int k, int ss,
                                     float anchor, float levels, int coded, int logarithmic, void *stream) {
    if (k % G || n < 1) return (int)cudaErrorInvalidValue;
    const int groups = k / G;
    k5_prepare_scales_pipe3<<<(n * groups + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const short*>(scales), reinterpret_cast<float*>(scales_f), n, groups, ss, anchor, levels, coded, logarithmic);
    return (int)cudaGetLastError();
}

extern "C" int launch_k5_imma_pipe3(int variant, const void *codes, const void *scales_f, const void *x, const void *rsum_f,
                                    void *aq, void *ac, void *y, int m, int n, int k, int cs, int xs, void *stream) {
    if (k % G || cs % 16 || xs % 8 || n % 8 || m < 1 || n < 1 ||
        reinterpret_cast<uintptr_t>(codes) % 16 || reinterpret_cast<uintptr_t>(x) % 16 ||
        reinterpret_cast<uintptr_t>(aq) % 16 || reinterpret_cast<uintptr_t>(ac) % 16 ||
        reinterpret_cast<uintptr_t>(rsum_f) % 16 || reinterpret_cast<uintptr_t>(scales_f) % 16)
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
    const int sub = variant / 1000;
    if (sub == 32)
        k5_quantize_activation_pipe3<32><<<(m * (k / 32) + 127) / 128, 128, 0, st>>>(
            reinterpret_cast<const half*>(x), reinterpret_cast<int8_t*>(aq), reinterpret_cast<float2*>(ac), m, k, xs);
    else if (sub == 16)
        k5_quantize_activation_pipe3<16><<<(m * (k / 16) + 127) / 128, 128, 0, st>>>(
            reinterpret_cast<const half*>(x), reinterpret_cast<int8_t*>(aq), reinterpret_cast<float2*>(ac), m, k, xs);
    else return (int)cudaErrorInvalidValue;
    #define ARGS reinterpret_cast<const int8_t*>(codes), reinterpret_cast<const float*>(scales_f), \
                 reinterpret_cast<const int8_t*>(aq), reinterpret_cast<const float2*>(ac), \
                 reinterpret_cast<const float*>(rsum_f), reinterpret_cast<half*>(y), m, n, k, cs
    switch (variant) {
        #define CASE(S, T, E) case S * 1000 + T * 10 + E: \
            k5_group_imma_pipe3_cuda<S, T, E><<<grid, THREADS, Stage<S>::BYTES * T, st>>>(ARGS); break;
        VARIANTS(CASE)
        #undef CASE
        default: return (int)cudaErrorInvalidValue;
    }
    #undef ARGS
    return (int)cudaGetLastError();
}
