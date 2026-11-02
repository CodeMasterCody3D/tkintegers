// Pipelined one-dot INT8 K5 prefill kernel (sm89 IMMA + multi-stage cp.async).
// Arithmetic is identical to k5_group_imma_cuda.cu: the K5 code is the s8 weight,
// the activation is quantized once per call to x ~ a * (A + c) per (row, SUB-group),
// sub-group dots are exact s32 tensor-core results, FP32 flush
//   y[m,n] = sum_g S[n,g] * sum_j a[m,gj] * ( D[m,n,gj] + c[m,gj] * R[n,gj] ).
// What changes is only HOW bytes reach shared memory: K is staged 64 at a time,
// STAGES buffers deep, with cp.async.cg 16-byte copies (zero-fill for rows past
// m or n) issued STAGES-1 stages ahead of the tensor-core consumer. Scales and code
// sums arrive as int16 in column-contiguous (transposed) copies so every input is a
// 16-byte chunk; each thread converts the eight group scales it needs itself, so no
// synchronous global load or extra barrier sits in the K loop.
// Tile 64x64, 8 warps (4 along M x 2 along N, WN=32), m16n8k32.s8 (SUB=32) or
// m16n8k16.s8 (SUB=16). Shared rows are 64 bytes; 16-byte chunk index XOR ((row>>1)&3)
// keeps ldmatrix conflict-free. Compile with --fmad=false; fmaf() written explicitly.
// New file: nothing existing is changed.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int BM = 64, BN = 64, G = 128, KT = 64, THREADS = 256, WN = 32;

template<int SUB> struct Stage {
    static constexpr int NSUB_T = KT / SUB;                    // sub-groups per stage: 2 (SUB 32) or 4 (SUB 16)
    static constexpr int A_OFF = 0;                             // s8 [BM][KT]
    static constexpr int W_OFF = BM * KT;                       // s8 [BN][KT]
    static constexpr int R_OFF = W_OFF + BN * KT;               // s16 [NSUB_T][BN]   (transposed code sums)
    static constexpr int AC_OFF = R_OFF + NSUB_T * BN * 2;      // float2 [BM][NSUB_T]
    static constexpr int S_OFF = AC_OFF + BM * NSUB_T * 8;      // s16 [BN] group scales (transposed)
    static constexpr int BYTES = S_OFF + BN * 2;                // 9,600 (SUB 32) / 10,880 (SUB 16)
    static constexpr int CHUNKS = (BM * KT + BN * KT + NSUB_T * BN * 2 + BM * NSUB_T * 8 + BN * 2) / 16;
};

__device__ __forceinline__ int sidx64(int row, int col) {     // col multiple of 16
    return row * KT + (((col >> 4) ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, bool valid) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

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
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ float exact_float(int v) {        // |v| < 2^22
    return __int_as_float(v + 0x4B400000) - 12582912.0f;
}

// sumg += a[row] * (D + c[row] * R[col]) for the thread's 2 rows x 8 columns; j is the sub-group within the stage.
template<int NSUB_T>
__device__ __forceinline__ void flush(float (&sumg)[4][4], const int (&acc)[4][4], int j,
                                      const float2 *acs, const short *Rs, int wm, int wn, int lane) {
    const float2 ac0 = acs[(wm * 16 + lane / 4) * NSUB_T + j];
    const float2 ac1 = acs[(wm * 16 + lane / 4 + 8) * NSUB_T + j];
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        const int col = wn * WN + h * 8 + (lane % 4) * 2;
        const int rr = *reinterpret_cast<const int*>(Rs + j * BN + col);   // two int16 code sums
        const float R0 = (float)(short)(rr & 0xffff), R1 = (float)(short)(rr >> 16);
        const float f0 = fmaf(ac0.y, R0, exact_float(acc[h][0]));
        const float f1 = fmaf(ac0.y, R1, exact_float(acc[h][1]));
        const float f2 = fmaf(ac1.y, R0, exact_float(acc[h][2]));
        const float f3 = fmaf(ac1.y, R1, exact_float(acc[h][3]));
        sumg[h][0] = fmaf(ac0.x, f0, sumg[h][0]);
        sumg[h][1] = fmaf(ac0.x, f1, sumg[h][1]);
        sumg[h][2] = fmaf(ac1.x, f2, sumg[h][2]);
        sumg[h][3] = fmaf(ac1.x, f3, sumg[h][3]);
    }
}

template<int SUB, int STAGES>
__global__ void __launch_bounds__(THREADS, 2)
k5_group_imma_pipe_cuda(const int8_t *codes, const short *scales_t, const int8_t *aq, const float2 *ac,
                        const short *rsum_t, half *y, int m, int n, int k, int cs,
                        float anchor, float levels, int coded, int logarithmic) {
    using St = Stage<SUB>;
    constexpr int NSUB_T = St::NSUB_T;
    extern __shared__ __align__(16) unsigned char storage[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / 2, wn = warp % 2;
    const int mb = blockIdx.x * BM, nb = blockIdx.y * BN;
    const int nsub_k = k / SUB, nstages = k / KT;

    // Stage s covers k in [s*KT, s*KT+KT): group g = s/2, sub-groups (s%2)*NSUB_T + j.
    auto issue = [&](int s, int buf) {
        unsigned char *base = storage + buf * St::BYTES;
        const int ks = s * KT, g = s / 2;
        #pragma unroll
        for (int i = tid; i < St::CHUNKS; i += THREADS) {
            if (i < BM * (KT / 16)) {                                  // quantized activation tile
                const int r = i / (KT / 16), c = (i % (KT / 16)) * 16;
                const bool valid = mb + r < m;
                cp_async16(base + St::A_OFF + sidx64(r, c), aq + (size_t)(valid ? mb + r : 0) * k + ks + c, valid);
            } else if (i < (BM + BN) * (KT / 16)) {                    // code tile (the weights, unchanged bytes)
                const int q = i - BM * (KT / 16), r = q / (KT / 16), c = (q % (KT / 16)) * 16;
                const bool valid = nb + r < n;
                cp_async16(base + St::W_OFF + sidx64(r, c), codes + (size_t)(valid ? nb + r : 0) * cs + ks + c, valid);
            } else if (i < (BM + BN) * (KT / 16) + NSUB_T * BN / 8) {  // code sums, transposed: [k/SUB][n]
                const int q = i - (BM + BN) * (KT / 16), j = q / (BN / 8), c = (q % (BN / 8)) * 8;
                const bool valid = nb + c < n;                          // n is a multiple of 8 when the chunk is valid
                cp_async16(base + St::R_OFF + (j * BN + c) * 2, rsum_t + (size_t)(ks / SUB + j) * n + (valid ? nb + c : 0), valid);
            } else if (i < (BM + BN) * (KT / 16) + NSUB_T * BN / 8 + BM * NSUB_T / 2) {  // (a, c) pairs
                const int q = i - (BM + BN) * (KT / 16) - NSUB_T * BN / 8, r = q / (NSUB_T / 2), j = (q % (NSUB_T / 2)) * 2;
                const bool valid = mb + r < m;
                cp_async16(base + St::AC_OFF + (r * NSUB_T + j) * 8, ac + (size_t)(valid ? mb + r : 0) * nsub_k + ks / SUB + j, valid);
            } else {                                                   // group scales, transposed: [k/128][n]
                const int q = i - (BM + BN) * (KT / 16) - NSUB_T * BN / 8 - BM * NSUB_T / 2, c = q * 8;
                const bool valid = nb + c < n;
                cp_async16(base + St::S_OFF + c * 2, scales_t + (size_t)g * n + (valid ? nb + c : 0), valid);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nstages) issue(s, s);
        cp_async_commit();
    }
    float sum[4][4] = {}, sumg[4][4] = {};
    for (int s = 0; s < nstages; ++s) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();                                   // stage s landed; stage s-1's buffer is free
        if (s + STAGES - 1 < nstages) issue(s + STAGES - 1, (s + STAGES - 1) % STAGES);
        cp_async_commit();
        const unsigned char *base = storage + (s % STAGES) * St::BYTES;
        const int8_t *As = reinterpret_cast<const int8_t*>(base + St::A_OFF);
        const int8_t *Ws = reinterpret_cast<const int8_t*>(base + St::W_OFF);
        const short *Rs = reinterpret_cast<const short*>(base + St::R_OFF);
        const float2 *acs = reinterpret_cast<const float2*>(base + St::AC_OFF);
        const short *Ss = reinterpret_cast<const short*>(base + St::S_OFF);
        #pragma unroll
        for (int t = 0; t < KT; t += 32) {
            unsigned a[4], b[2][4];
            ldsm4(a, As + sidx64(wm * 16 + lane % 16, t + (lane / 16) * 16));
            ldsm4(b[0], Ws + sidx64(wn * WN + lane % 16, t + (lane / 16) * 16));
            ldsm4(b[1], Ws + sidx64(wn * WN + 16 + lane % 16, t + (lane / 16) * 16));
            if constexpr (SUB == 32) {
                int acc[4][4] = {};
                imma32(acc[0], a, b[0][0], b[0][2]);
                imma32(acc[1], a, b[0][1], b[0][3]);
                imma32(acc[2], a, b[1][0], b[1][2]);
                imma32(acc[3], a, b[1][1], b[1][3]);
                flush<NSUB_T>(sumg, acc, t / 32, acs, Rs, wm, wn, lane);
            } else {
                int acc[4][4] = {};
                imma16(acc[0], a[0], a[1], b[0][0]);
                imma16(acc[1], a[0], a[1], b[0][1]);
                imma16(acc[2], a[0], a[1], b[1][0]);
                imma16(acc[3], a[0], a[1], b[1][1]);
                flush<NSUB_T>(sumg, acc, t / 16, acs, Rs, wm, wn, lane);
                int acc2[4][4] = {};
                imma16(acc2[0], a[2], a[3], b[0][2]);
                imma16(acc2[1], a[2], a[3], b[0][3]);
                imma16(acc2[2], a[2], a[3], b[1][2]);
                imma16(acc2[3], a[2], a[3], b[1][3]);
                flush<NSUB_T>(sumg, acc2, t / 16 + 1, acs, Rs, wm, wn, lane);
            }
        }
        if (s & 1) {                                       // end of a 128-group: apply S[n,g] / 121
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {           // the thread's two columns in this n-tile
                    const int c = wn * WN + h * 8 + (lane % 4) * 2 + cc;
                    float sc = (float)Ss[c];
                    if (coded) {
                        sc = (sc / levels) * anchor;
                        if (logarithmic) sc = exp2f(sc);
                    }
                    sc = sc / 121.0f;
                    sum[h][cc] += sumg[h][cc] * sc;         // row 0; separate multiply and add (--fmad=false)
                    sum[h][cc + 2] += sumg[h][cc + 2] * sc; // row 1
                    sumg[h][cc] = 0.0f;
                    sumg[h][cc + 2] = 0.0f;
                }
            }
        }
    }
    cp_async_wait<0>();
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

// Same activation quantizer as k5_group_imma_cuda.cu (one thread per (row, sub-group)).
template<int SUB>
__global__ void k5_quantize_activation_pipe(const half *x, int8_t *aq, float2 *ac, int m, int k, int xs) {
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

template<int SUB, int STAGES>
static int initialize(int *info) {
    constexpr int BYTES = Stage<SUB>::BYTES * STAGES;
    cudaError_t rc = cudaFuncSetAttribute(k5_group_imma_pipe_cuda<SUB, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
    if (rc != cudaSuccess) return (int)rc;
    cudaFuncAttributes attr;
    rc = cudaFuncGetAttributes(&attr, k5_group_imma_pipe_cuda<SUB, STAGES>);
    if (rc != cudaSuccess) return (int)rc;
    int active;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, k5_group_imma_pipe_cuda<SUB, STAGES>, THREADS, BYTES);
    if (rc != cudaSuccess) return (int)rc;
    info[0] = BM; info[1] = BN; info[2] = THREADS; info[3] = BYTES;
    info[4] = attr.numRegs; info[5] = (int)attr.localSizeBytes; info[6] = active;
    cudaFuncAttributes qattr;
    rc = cudaFuncGetAttributes(&qattr, k5_quantize_activation_pipe<SUB>);
    if (rc != cudaSuccess) return (int)rc;
    info[7] = qattr.numRegs; info[8] = (int)qattr.localSizeBytes;
    return active > 0 ? (int)cudaSuccess : (int)cudaErrorInvalidConfiguration;
}

#define VARIANTS(X) X(32, 2) X(32, 3) X(32, 4) X(32, 5) X(16, 2) X(16, 3) X(16, 4)

// variant = sub * 100 + stages, e.g. 3204 = SUB 32, 4 stages. info has 9 ints.
extern "C" int initialize_k5_imma_pipe(int variant, int *info) {
    switch (variant) {
        #define CASE(S, T) case S * 100 + T: return initialize<S, T>(info);
        VARIANTS(CASE)
        #undef CASE
        default: return (int)cudaErrorInvalidValue;
    }
}

// codes s8 [n,k] (row stride cs, multiple of 16); scales_t s16 [k/128][n]; x f16 [m,k] (row stride xs);
// rsum_t s16 [k/SUB][n]; aq s8 [m,k] and ac float2 [m,k/SUB] scratch; y f16 [m,n]. n must be a multiple of 8.
extern "C" int launch_k5_imma_pipe(int variant, const void *codes, const void *scales_t, const void *x, const void *rsum_t,
                                   void *aq, void *ac, void *y, int m, int n, int k, int cs, int xs,
                                   float anchor, float levels, int coded, int logarithmic, void *stream) {
    if (k % G || cs % 16 || xs % 8 || n % 8 || m < 1 || n < 1 ||
        reinterpret_cast<uintptr_t>(codes) % 16 || reinterpret_cast<uintptr_t>(x) % 16 ||
        reinterpret_cast<uintptr_t>(aq) % 16 || reinterpret_cast<uintptr_t>(ac) % 16 ||
        reinterpret_cast<uintptr_t>(rsum_t) % 16 || reinterpret_cast<uintptr_t>(scales_t) % 16)
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
    const int sub = variant / 100;
    if (sub == 32)
        k5_quantize_activation_pipe<32><<<(m * (k / 32) + 127) / 128, 128, 0, st>>>(
            reinterpret_cast<const half*>(x), reinterpret_cast<int8_t*>(aq), reinterpret_cast<float2*>(ac), m, k, xs);
    else if (sub == 16)
        k5_quantize_activation_pipe<16><<<(m * (k / 16) + 127) / 128, 128, 0, st>>>(
            reinterpret_cast<const half*>(x), reinterpret_cast<int8_t*>(aq), reinterpret_cast<float2*>(ac), m, k, xs);
    else return (int)cudaErrorInvalidValue;
    #define ARGS reinterpret_cast<const int8_t*>(codes), reinterpret_cast<const short*>(scales_t), \
                 reinterpret_cast<const int8_t*>(aq), reinterpret_cast<const float2*>(ac), \
                 reinterpret_cast<const short*>(rsum_t), reinterpret_cast<half*>(y), \
                 m, n, k, cs, anchor, levels, coded, logarithmic
    switch (variant) {
        #define CASE(S, T) case S * 100 + T: \
            k5_group_imma_pipe_cuda<S, T><<<grid, THREADS, Stage<S>::BYTES * T, st>>>(ARGS); break;
        VARIANTS(CASE)
        #undef CASE
        default: return (int)cudaErrorInvalidValue;
    }
    #undef ARGS
    return (int)cudaGetLastError();
}
