# K5 one-dot INT8 prefill trials (Qwen3.5-9B, Colab L4)

These are experiment kernels and harnesses, recorded October 31 to November 1, 2026.
They are **not** a new default runtime. The accepted K5 prefill kernel and the
desktop converter are unchanged.

## Question

With CUDA graphs on for both formats, FP8 prefill beat K5 prefill on the L4.
The whole-model fair run used the same pinned Qwen3.5-9B and a 128-token prompt:

| Qwen3.5-9B, L4, 128-token prompt | K5 (`tk_k5_g128_ISk10log`) | TorchAO FP8 (E4M3 per-tensor) | K5 ÷ FP8 |
|---|---|---|---|
| Prefill, eager on both | 1048.6 tok/s | 466.1 tok/s | **2.25×** |
| Prefill, CUDA graphs on both | 1228.0 tok/s | 1959.1 tok/s | **0.63×** |
| Decode, eager, after graph prefill | 11.81 tok/s | 4.35 tok/s | **2.72×** |

The question was whether K5 could stay exact enough and run the prefill matrix
multiply on INT8 tensor cores fast enough to close that graph gap.

## Method: one-dot INT8 with affine activations

- The K5 code `q ∈ [-121, 121]` is the signed INT8 weight directly.
- Activations are quantized once per call, per row and per sub-group of
  `SUB` values: `x ≈ a·(A + c)`, where `A` is s8.
- The output is `y = Σ_g S[n,g] · Σ_j a·(D + c·R)`:
  - `D` is the exact s32 IMMA dot product;
  - `R` holds exact int16 sums of the codes, computed once per matrix.
- Accuracy gate: relative L2 ≤ 0.005 against an FP64 reference, on real 9B
  matrices and real prefill activations. The gate was not relaxed at any point.

## Accuracy (relative L2, lower is better)

| Activation scaling | 8192×4096 | 4096×4096 | 12288×4096 | 4096×12288 | Passes all 4 |
|---|---|---|---|---|---|
| Affine, scale per 128 | 0.00382 | 0.00456 | 0.00645 | 0.00977 | no |
| Affine, scale per 32 | 0.00212 | 0.00257 | 0.00408 | 0.00505 | no (3/4) |
| **Affine, scale per 16** | **0.00160** | **0.00192** | **0.00315** | **0.00353** | **yes** |
| FP8 E4M3 per-tensor (TorchAO scheme) | 0.0156–0.0309 across shapes | | | | no |

## Speed (ms per matrix, CUDA-graph timing, 128 tokens)

| Shape | Exact K5 swizzle kernel | Best INT8 (pipe v3) | FP8 cuBLAS + activation quant |
|---|---|---|---|
| 12288×4096 | 0.575 | 0.565 | 0.336 |
| 8192×4096 | 0.272 | **0.192** | 0.146 |
| 4096×4096 | 0.137 | **0.106** | 0.094 |
| 4096×12288 (sub-16) | **0.556** | 0.661 | 0.414 |

- **Pipe v3 beats the exact K5 kernel on three of four shapes, by up to 1.4×.**
- **FP8 cuBLAS stays 1.1–1.7× faster than INT8 on every shape.**

## Why it stopped: the 72 W power cap

- Nsight Compute shows the gap.
  - cuBLAS FP8 streams at 86% of DRAM bandwidth; the K5 kernels run at 33–36%.
  - Pipelining (cp.async, 3–6 stages) removed the memory-latency stalls: long-scoreboard fell from 2.76 to 0.01.
- On large shapes the L4 hits its 72 W cap and lowers the SM clock (to roughly 650–800 MHz, against 2040 MHz idle).
  - Pipe v3 needs about 29% fewer cycles than the swizzle kernel, but loses them to the clock drop.
  - Energy per 12288×4096 matrix: about **41 mJ for K5, about 24 mJ for FP8**.
- The per-sub-group scale flush (about 600M thread-ops per large matrix) dominates
  dynamic energy. It is built into sub-group activation scaling.

**Verdict:** one-dot INT8 is accuracy-viable under the unchanged gate. It does not
beat FP8 cuBLAS prefill on a power-capped L4 at 128 tokens. K5's measured whole-model
win remains decode, at 2.72×. Whole-model quality was not measured in these trials.

## Files

| File | Role | sha256 prefix |
|---|---|---|
| `probe_int8_affine_activation.py` | per-128 accuracy probe | 6c19fab5 |
| `probe_int8_subgroup_activation.py` | per-64/32 probe | 38450ed2 |
| `probe_int8_sub16_activation.py` | per-16/8 probe | 3b65a3e7 |
| `k5_group_imma_cuda.cu`, `test_k5_imma_matrix_l4.py` | INT8 v1, 64×64 tile, 8 warps | ba45a81b, a1c3c427 |
| `k5_group_imma_pipe_cuda.cu`, `test_k5_imma_pipe_matrix_l4.py` | pipe v1, cp.async stages | 71583ec7, 9e74a4f4 |
| `k5_group_imma_pipe2_cuda.cu`, `test_k5_imma_pipe2_matrix_l4.py` | pipe v2, prepared scales | 62c7022d, a5cddc4b |
| `k5_group_imma_pipe3_cuda.cu`, `test_k5_imma_pipe3_matrix_l4.py` | pipe v3, 32×32 warp tiles, constant-C flush | e69b2dca, 73fd0ed9 |
| `int8_prefill_results/` | each run's `result.json` plus Nsight text | — |

The harnesses expect the Colab layout `/content/tk-speed9-1029/` (9B K5 matrices,
captured prefill activations and the preserved swizzle control binary). They are
published as the record of what was run, not as a turnkey installer.
