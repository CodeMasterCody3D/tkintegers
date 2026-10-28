# Native k5 GPU kernels

This directory includes the exact faster native CUDA prefill source checked on
T4 and L4, together with its original decoder dependency. Publication does not
change the desktop converter or enable a new default runtime.

## Included implementations

- `k5_group_cuda.cu`: native group-WMMA prefill. Mode zero uses exact signed k5
  codes as FP16 operands, FP32 group accumulation and post-dot group scaling.
- `k5_prefill_cuda.py`: the original opt-in Kaggle T4/sm75 adapter. It defaults
  to `group_fp16`; its other variants are experimental, not selected defaults.
- `k5_prefill_cuda_l4.py`: the measured Colab L4/sm89 adapter, using the same
  CUDA arithmetic and mode zero.
- `k5_prefill_tiled.py`: the preserved earlier Triton prefill implementation.
  It also supplies the shared reference-module loader used by the adapters.
- `gemv_reference.py`: the frozen original Triton module, included unchanged so
  decode and scale reconstruction need no private source checkout.

`KERNEL_PROVENANCE.json` locks these five source files to the accepted L4 source
identities. The files are published without arithmetic or dispatch changes.
Historical comments in the original files are preserved; the tested scope below
takes precedence over broad capability descriptions in those comments.

## Representation and execution contract

The tested model format is **`tk_k5_g128_ISk10log`**: five-trit weight codes,
groups of 128 weights, and integer-encoded k10 logarithmic scales. K5 is a
243-level grid, not five binary bits or just three weight levels.

The native entry point takes:

- `codes`: contiguous-column CUDA `int8` matrix `[N, K]`, values from -121 to
  121. These are the saved unsigned bytes minus 121, not raw safetensors bytes.
- `scales`: CUDA `int16` matrix `[N, K/128]` containing the integer scale codes.
- `X`: CUDA FP16 activation matrix `[M, K]` on the same device.
- `meta`: CPU int64 tensor `[anchor_mantissa, anchor_exponent, 10, 1]`.
- Optional output-channel bias; output dtype FP16 for native prefill.

`K` must be divisible by 128. A scale reconstructs as
`2 ** (scale_code * (anchor_mantissa * 3 ** anchor_exponent) / 29524)`;
the weight reconstructs as `signed_code * scale / 121`.

The caller must validate the code range, descriptor and group identity before
invoking the low-level kernel. The wrappers check tensor geometry/dtypes but do
not validate every code or certify arbitrary scale policies. FS16 scales are not
the measured configuration. FP16 operands do not imply FP16 scale storage.

The adapters send one activation row to the original fused decoder, two through
sixteen rows to its small-batch path, and larger inputs to native prefill. The
full body is not retained as expanded FP16 weights. Changed scaling/rounding
locations mean output need not be bit-identical to a dense decoded-weight GEMM.

## Dependencies and use

Use an authorized remote GPU, not the laptop. Required components are CUDA
PyTorch, Triton and NVCC. The measured L4 environment used PyTorch 2.11.0+cu130,
Triton 3.6.0 and CUDA 13.0; this is a recorded environment, not a guarantee for
every package combination. The CUDA compilation uses `-O3` and `--fmad=false`.
First-use compilation/autotuning is not steady-state speed.

The unchanged adapters intentionally retain their tested environment guards:

- T4: compute capability 7.5 and a Kaggle filesystem; cache under `/kaggle/temp`.
- L4: compute capability 8.9 and mounted Colab Drive; build files under
  `/content/tk-fp8-l4-1027/build`.

On the assigned GPU, a caller with already validated resident tensors can use:

```python
import sys
sys.path.insert(0, "benchmarks")  # Run from the repository root.
import k5_prefill_cuda_l4 as kernel  # T4: import k5_prefill_cuda instead.

# codes, scales, X and meta must satisfy the contract above.
output = kernel.gemm_k5(codes, scales, X, meta=meta, out_dtype=X.dtype)
```

This is an operator integration example, not a model loader or evaluation script.
By default the reference loader finds the included `gemv_reference.py`. An
existing `TK_K5_REFERENCE` environment variable overrides that path; unset it to
use the bundled decoder. No dependency download or model run happens on import;
the CUDA library compiles when the native path is first called.

The adapters are not yet generalized to A100 or arbitrary accelerators. A source
publication does not establish Qwen 27B loader compatibility, GPU fit or answer
quality. The measured model was Qwen2.5-0.5B-Instruct. Benchmark drivers, private
artifacts and unpublished paper drafts are not part of this focused release.

## Measured scope

The matched T4 old/new whole-model comparison measured prefill medians of 843.46
and 1331.17 tokens/s: 57.82% more throughput. Decode medians were 23.13 and
23.21 tokens/s with overlapping ranges; this was a prefill improvement, not a new
faster single-token decoder.

The separate L4 comparison against installed eager TorchAO FP8 measured 3402.85
versus 1061.58 prefill tokens/s and 26.75 versus 8.63 fixed-sequence decode
tokens/s. These are 3.21x and 3.10x throughput ratios for those implementations;
the FP16 control remained faster. They are not an old/new L4 kernel ablation,
an intrinsic format ranking, or proof of task-quality equivalence.

The selected native FP16 path passed bounded loaded-matrix/model checks. The
direct INT8 activation variant failed the unchanged numerical criterion and must
not be described as accepted because it is present in the source. Residual INT8
passed the matrix checks but was not selected for model timing. Broader model,
context and reasoning evaluations remain separate work.
