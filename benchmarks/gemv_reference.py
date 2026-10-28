"""Fused GEMV kernels for the packed base-3 k1..k16 container and the special-cased k5
(one signed level per weight, byte-aligned — see its own note below). Never materializes
the dense dequantized weight: the kernel decodes trits and multiplies in the same pass.

Requires `triton` (bundled with a CUDA-build PyTorch on Linux; install separately
otherwise) and a CUDA device. `kernels/gemv_torch.py` is the dependency-free fallback with
identical semantics and no Triton/CUDA requirement — use it on CPU, non-NVIDIA GPUs, or
whenever the Triton toolchain is unavailable.

Scales may be plain floating-point tensors, or integer "tk scale codes" (int8/16/32/64)
decoded inside the kernel via `meta = [anchor_mantissa, anchor_exponent, k, is_log]` — a
compact encoding used elsewhere in the wider Tk-Integer pipeline this package draws from.
Pass `meta=None` (the default) to use plain float scales, which is what `quantizer.py`'s
`QTKLinear` produces.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _scale_args(scales, meta):
    coded = meta is not None
    if not coded:
        return False, 1.0, 1.0, False
    values = [int(v) for v in meta.detach().cpu().tolist()]
    if len(values) != 4:
        raise ValueError("scale metadata must be [anchor_mantissa, anchor_exponent, k, is_log]")
    am, ae, k, log = values
    return True, float(am) * (3.0 ** ae), float((3 ** k - 1) // 2), bool(log)


_POW3 = {}


def _pow3(device, count):
    key = (device.type, device.index, int(count))
    if key not in _POW3:
        _POW3[key] = torch.tensor([3 ** i for i in range(count)], device=device, dtype=torch.float32)
    return _POW3[key]


@triton.jit
def _k5_kernel(codes_ptr, scales_ptr, x_ptr, out_ptr,
              n_rows, n_cols, code_stride, scale_stride,
              scale_anchor, scale_levels,
              SCALE_CODED: tl.constexpr, SCALE_LOG: tl.constexpr,
              ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = rows < n_rows
    acc = tl.zeros((ROWS,), dtype=tl.float32)
    for base in range(0, n_cols, BLOCK):
        cols = base + tl.arange(0, BLOCK)
        cm = cols < n_cols
        code = tl.load(codes_ptr + rows[:, None] * code_stride + cols[None, :],
                      mask=rm[:, None] & cm[None, :], other=0).to(tl.float32)
        scale = tl.load(scales_ptr + rows[:, None] * scale_stride + cols[None, :] // 128,
                       mask=rm[:, None] & cm[None, :], other=0).to(tl.float32)
        if SCALE_CODED:
            scale = scale / scale_levels * scale_anchor
            if SCALE_LOG:
                scale = tl.exp2(scale)
        xv = tl.load(x_ptr + cols, mask=cm, other=0).to(tl.float32)
        acc += tl.sum(code * (1.0 / 121.0) * scale * xv[None, :], axis=1)
    tl.store(out_ptr + rows, acc, mask=rm)


def _k5_v2_configs():
    return [triton.Config({"ROWS": r, "BLOCK": b}, num_warps=w, num_stages=s)
            for r in (4, 8, 16) for b in (256, 512, 1024) for w in (4, 8) for s in (2, 3)]


@triton.autotune(configs=_k5_v2_configs(), key=["n_rows", "n_cols"])
@triton.jit
def _k5_kernel_v2(codes_ptr, scales_ptr, x_ptr, out_ptr,
                  n_rows, n_cols, code_stride, scale_stride,
                  scale_anchor, scale_levels,
                  SCALE_CODED: tl.constexpr, SCALE_LOG: tl.constexpr,
                  ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """SCALE-HOISTED k5 GEMV (2026-09-23; [[kernel-scale-hoisting]] measured 1.55x on the ternary kernel).
    v1 gathered one scale PER WEIGHT and did code*(1/121)*scale*x per element. Here each 128-wide group is
    reduced as sum(code*x) first, then multiplied by its ONE scale, and 1/121 is applied once at the end --
    the same sum, re-associated. BLOCK spans several groups per step; ROWS/BLOCK/warps/stages autotuned per
    shape."""
    G: tl.constexpr = 128
    NG: tl.constexpr = BLOCK // G
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = rows < n_rows
    n_groups = n_cols // G
    acc = tl.zeros((ROWS,), dtype=tl.float32)
    for base in range(0, n_cols, BLOCK):
        cols = base + tl.arange(0, BLOCK)
        cm = cols < n_cols
        code = tl.load(codes_ptr + rows[:, None] * code_stride + cols[None, :],
                       mask=rm[:, None] & cm[None, :], other=0).to(tl.float32)
        xv = tl.load(x_ptr + cols, mask=cm, other=0).to(tl.float32)
        part = tl.sum(tl.reshape(code * xv[None, :], (ROWS, NG, G)), axis=2)          # [ROWS, NG] per group
        gidx = base // G + tl.arange(0, NG)
        gm = gidx < n_groups
        scale = tl.load(scales_ptr + rows[:, None] * scale_stride + gidx[None, :],
                        mask=rm[:, None] & gm[None, :], other=0).to(tl.float32)
        if SCALE_CODED:
            scale = scale / scale_levels * scale_anchor
            if SCALE_LOG:
                scale = tl.exp2(scale)
        acc += tl.sum(part * scale, axis=1)
    tl.store(out_ptr + rows, (acc * (1.0 / 121.0)).to(out_ptr.dtype.element_ty), mask=rm)


@triton.jit
def _base3_kernel(packed_ptr, scales_ptr, x_ptr, out_ptr, powers_ptr,
                  n_rows, n_cols, packed_stride, scale_stride,
                  scale_anchor, scale_levels, weight_levels, group_size,
                  SCALE_CODED: tl.constexpr, SCALE_LOG: tl.constexpr,
                  K: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = rows < n_rows
    acc = tl.zeros((ROWS,), dtype=tl.float32)
    for base in range(0, n_cols, BLOCK):
        cols = base + tl.arange(0, BLOCK)
        cm = cols < n_cols
        level = tl.zeros((ROWS, BLOCK), dtype=tl.float32)
        for digit in range(0, K):
            ti = cols * K + digit
            byte = ti // 5
            offset = ti % 5
            raw = tl.load(packed_ptr + rows[:, None] * packed_stride + byte[None, :],
                         mask=rm[:, None] & cm[None, :], other=0).to(tl.int32)
            # Triton cannot exponentiate a compile-time scalar by a runtime tensor; offset
            # is always 0..4, so use the equivalent fixed lookup without a table load.
            pow_offset = tl.where(offset == 0, 1,
                         tl.where(offset == 1, 3,
                         tl.where(offset == 2, 9,
                         tl.where(offset == 3, 27, 81))))
            trit = (raw // pow_offset) % 3
            level += (trit.to(tl.float32) - 1.0) * tl.load(powers_ptr + digit)
        scale = tl.load(scales_ptr + rows[:, None] * scale_stride + cols[None, :] // group_size,
                       mask=rm[:, None] & cm[None, :], other=0).to(tl.float32)
        if SCALE_CODED:
            scale = scale / scale_levels * scale_anchor
            if SCALE_LOG:
                scale = tl.exp2(scale)
        xv = tl.load(x_ptr + cols, mask=cm, other=0).to(tl.float32)
        acc += tl.sum(level / weight_levels * scale * xv[None, :], axis=1)
    tl.store(out_ptr + rows, acc, mask=rm)


_K5_BEST = {}      # (rows, cols, coded, log) -> the autotuned Config: later calls skip the autotuner's CPU dispatch


def gemv_k5(codes, scales, x, meta=None, bias=None, rows_per_program=4, warps=2, v1=False, out_dtype=None):
    """Fused GEMV for one-signed-level-per-weight k5 (243 levels, byte-aligned — see
    `resident.py`'s note on why k5 is operationally int8, not a base-3 trit decode).
    Default = the scale-hoisted, autotuned v2 kernel; v1=True runs the original for A/B.
    out_dtype (v2): the kernel writes this dtype directly (e.g. bf16 -> no separate cast kernel per call);
    default fp32 as before. Accumulation is always fp32."""
    if not all(t.is_cuda for t in (codes, scales, x)) or x.ndim != 1:
        raise ValueError("fused GEMV requires CUDA tensors and a 1-D activation")
    rows, cols = map(int, codes.shape)
    if codes.ndim != 2 or scales.shape != (rows, cols // 128) or cols % 128:
        raise ValueError("invalid k5 codes/scales geometry")
    coded, anchor, levels, logarithmic = _scale_args(scales, meta)
    if not v1:
        out = torch.empty(rows, device=x.device, dtype=out_dtype or torch.float32)
        key = (rows, cols, coded, logarithmic)
        cfg = _K5_BEST.get(key)
        if cfg is None:                              # first call per shape: autotune, then remember the winner
            _k5_kernel_v2[lambda meta_: (triton.cdiv(rows, meta_["ROWS"]),)](
                codes, scales, x.contiguous(), out, rows, cols, codes.stride(0), scales.stride(0),
                anchor, levels, SCALE_CODED=coded, SCALE_LOG=logarithmic)
            _K5_BEST[key] = _k5_kernel_v2.best_config
        else:                                        # direct JIT launch with the tuned config
            kw = cfg.kwargs
            _k5_kernel_v2.fn[(triton.cdiv(rows, kw["ROWS"]),)](
                codes, scales, x.contiguous(), out, rows, cols, codes.stride(0), scales.stride(0),
                anchor, levels, SCALE_CODED=coded, SCALE_LOG=logarithmic, ROWS=kw["ROWS"], BLOCK=kw["BLOCK"],
                num_warps=cfg.num_warps, num_stages=cfg.num_stages)
        if bias is not None:
            out += bias.to(out.dtype)
        return out
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    _k5_kernel[(triton.cdiv(rows, rows_per_program),)](
        codes, scales, x.contiguous(), out, rows, cols, codes.stride(0), scales.stride(0),
        anchor, levels, SCALE_CODED=coded, SCALE_LOG=logarithmic, ROWS=rows_per_program,
        BLOCK=128, num_warps=warps)
    if bias is not None:
        out += bias.to(out.dtype)
    return out


def _k5_rows_configs():
    return [triton.Config({"ROWS": r}, num_warps=w, num_stages=s) for r in (16, 32, 64) for w in (4, 8) for s in (2, 3, 4)]


@triton.autotune(configs=_k5_rows_configs(), key=["n_rows", "n_cols"])
@triton.jit
def _k5_rows_kernel(codes_ptr, scales_ptr, x_ptr, out_ptr,
                    n_rows, n_cols, n_x, code_stride, scale_stride, x_stride, out_stride,
                    scale_anchor, scale_levels,
                    SCALE_CODED: tl.constexpr, SCALE_LOG: tl.constexpr, X_F16: tl.constexpr,
                    ROWS: tl.constexpr, MP: tl.constexpr):
    """BATCHED k5 decode (10-04, Cody: "apply that super fast kernel ... make it a gpu kernel" -- the k5 teacher's live traces):
    X [n_x, K] with 2 <= n_x <= 16. One program owns ROWS output rows and reads their int8 codes ONCE for every input row (the old
    path for >1 row dequantized the WHOLE matrix per forward -- the same defect the k1 kernel had until 10-02, 4 tok/s). Per 128-wide
    group: part[ROWS, n_x] = codes . X^T on the group (tl.dot; the codes |c| <= 121 are exact in fp16, so with fp16 activations the
    tensor cores multiply exact values and accumulate in fp32), times the group's ONE scale, accumulated in fp32; 1/121 at the end --
    v2's math, re-associated across rows (fp32 rounding differs from gemv_k5 per row only by summation order)."""
    G: tl.constexpr = 128
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = rows < n_rows
    ms = tl.arange(0, MP)
    mm = ms < n_x
    acc = tl.zeros((ROWS, MP), dtype=tl.float32)
    for g in range(0, n_cols // G):
        cols = g * G + tl.arange(0, G)
        code = tl.load(codes_ptr + rows[:, None] * code_stride + cols[None, :], mask=rm[:, None], other=0)
        xt = tl.load(x_ptr + ms[None, :] * x_stride + cols[:, None], mask=mm[None, :], other=0.0)
        if X_F16:
            part = tl.dot(code.to(tl.float16), xt.to(tl.float16), out_dtype=tl.float32)
        else:
            part = tl.dot(code.to(tl.float32), xt.to(tl.float32), input_precision="ieee")
        scale = tl.load(scales_ptr + rows * scale_stride + g, mask=rm, other=0).to(tl.float32)
        if SCALE_CODED:
            scale = scale / scale_levels * scale_anchor
            if SCALE_LOG:
                scale = tl.exp2(scale)
        acc += part * scale[:, None]
    acc = acc * (1.0 / 121.0)
    tl.store(out_ptr + ms[None, :] * out_stride + rows[:, None], acc.to(out_ptr.dtype.element_ty), mask=rm[:, None] & mm[None, :])


K5_ROWS_MAX = 16


def _k5_gemm_configs():
    return [triton.Config({"ROWS": r, "BM": m}, num_warps=w, num_stages=s)
            for r in (64, 128) for m in (64, 128) for w in (4, 8) for s in (2, 3)]


@triton.autotune(configs=_k5_gemm_configs(), key=["n_rows", "n_cols", "m_bucket"])
@triton.jit
def _k5_gemm_kernel(codes_ptr, scales_ptr, x_ptr, out_ptr,
                    n_rows, n_cols, n_x, m_bucket, code_stride, scale_stride, x_stride, out_stride,
                    scale_anchor, scale_levels,
                    SCALE_CODED: tl.constexpr, SCALE_LOG: tl.constexpr, X_F16: tl.constexpr,
                    ROWS: tl.constexpr, BM: tl.constexpr):
    """PREFILL k5 (10-04, Cody: "a smoke using the k5 teacher with the new fast kernel" -- the teacher cache walks 256-token rows, past
    gemv_k5_rows' 16): _k5_rows_kernel's math with a second program axis over BM-row tiles of X, so the int8 codes are read once per
    tile instead of the old prefill path's dense fp16 dequant of the whole matrix on every forward. Per 128-wide group:
    part[ROWS, BM] = codes . X_tile^T (exact codes, fp16 tensor cores, fp32 accumulate), times the group's scale; 1/121 at the end."""
    G: tl.constexpr = 128
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = rows < n_rows
    ms = tl.program_id(1) * BM + tl.arange(0, BM)
    mm = ms < n_x
    acc = tl.zeros((ROWS, BM), dtype=tl.float32)
    for g in range(0, n_cols // G):
        cols = g * G + tl.arange(0, G)
        code = tl.load(codes_ptr + rows[:, None] * code_stride + cols[None, :], mask=rm[:, None], other=0)
        xb = tl.load(x_ptr + ms[:, None] * x_stride + cols[None, :], mask=mm[:, None], other=0.0)
        if X_F16:
            part = tl.dot(code.to(tl.float16), tl.trans(xb.to(tl.float16)), out_dtype=tl.float32)
        else:
            part = tl.dot(code.to(tl.float32), tl.trans(xb.to(tl.float32)), input_precision="ieee")
        scale = tl.load(scales_ptr + rows * scale_stride + g, mask=rm, other=0).to(tl.float32)
        if SCALE_CODED:
            scale = scale / scale_levels * scale_anchor
            if SCALE_LOG:
                scale = tl.exp2(scale)
        acc += part * scale[:, None]
    acc = acc * (1.0 / 121.0)
    tl.store(out_ptr + ms[None, :] * out_stride + rows[:, None], acc.to(out_ptr.dtype.element_ty), mask=rm[:, None] & mm[None, :])


def gemm_k5(codes, scales, X, meta=None, bias=None, out_dtype=None):
    """Prefill for the k5 container: X [M, K], any M >= 1 -> [M, rows]; the same inputs and math as gemv_k5_rows, tiled over M."""
    if not all(t.is_cuda for t in (codes, scales, X)) or X.ndim != 2 or X.shape[0] < 1:
        raise ValueError("gemm_k5 needs CUDA tensors and X [M, K] with M >= 1")
    rows, cols = map(int, codes.shape)
    if codes.ndim != 2 or codes.dtype != torch.int8 or scales.shape != (rows, cols // 128) or cols % 128 or X.shape[1] != cols:
        raise ValueError("invalid k5 codes/scales/X geometry")
    coded, anchor, levels, logarithmic = _scale_args(scales, meta)
    X = X.contiguous()
    M = int(X.shape[0])
    out = torch.empty(M, rows, device=X.device, dtype=out_dtype or torch.float32)
    _k5_gemm_kernel[lambda meta_: (triton.cdiv(rows, meta_["ROWS"]), triton.cdiv(M, meta_["BM"]))](
        codes, scales, X, out, rows, cols, M, 1 << max(0, M - 1).bit_length(), codes.stride(0), scales.stride(0), X.stride(0),
        out.stride(0), anchor, levels, SCALE_CODED=coded, SCALE_LOG=logarithmic, X_F16=(X.dtype == torch.float16))
    if bias is not None:
        out += bias.to(out.dtype)
    return out


def gemv_k5_rows(codes, scales, X, meta=None, bias=None, out_dtype=None):
    """Batched decode for the k5 container: X [M, K], 2 <= M <= K5_ROWS_MAX -> [M, rows]. Same inputs as gemv_k5 (int8 codes [O, K],
    one scale per 128 columns). fp16 X runs the tensor cores on exact values; any other dtype runs an fp32 (ieee) dot. Accumulation fp32."""
    if not all(t.is_cuda for t in (codes, scales, X)) or X.ndim != 2 or not (2 <= X.shape[0] <= K5_ROWS_MAX):
        raise ValueError(f"gemv_k5_rows needs CUDA tensors and X [M, K] with 2 <= M <= {K5_ROWS_MAX}")
    rows, cols = map(int, codes.shape)
    if codes.ndim != 2 or codes.dtype != torch.int8 or scales.shape != (rows, cols // 128) or cols % 128 or X.shape[1] != cols:
        raise ValueError("invalid k5 codes/scales/X geometry")
    coded, anchor, levels, logarithmic = _scale_args(scales, meta)
    X = X.contiguous()
    M = int(X.shape[0])
    out = torch.empty(M, rows, device=X.device, dtype=out_dtype or torch.float32)
    _k5_rows_kernel[lambda meta_: (triton.cdiv(rows, meta_["ROWS"]),)](
        codes, scales, X, out, rows, cols, M, codes.stride(0), scales.stride(0), X.stride(0), out.stride(0),
        anchor, levels, SCALE_CODED=coded, SCALE_LOG=logarithmic, X_F16=(X.dtype == torch.float16), MP=16)
    if bias is not None:
        out += bias.to(out.dtype)
    return out


def gemv_base3(packed, scales, x, weight_k, group_size=128, meta=None, bias=None,
              rows_per_program=4, warps=2):
    """Fused GEMV for the packed base-3 k1..k16 trit-stream container."""
    if not all(t.is_cuda for t in (packed, scales, x)) or x.ndim != 1:
        raise ValueError("fused GEMV requires CUDA tensors and a 1-D activation")
    rows = int(packed.shape[0])
    cols = int(x.numel())
    k = int(weight_k)
    if packed.ndim != 2 or k not in range(1, 17):
        raise ValueError("packed base-3 GEMV supports k1 through k16")
    expected_bytes = (cols * k + 4) // 5
    if packed.shape[1] != expected_bytes or cols % group_size:
        raise ValueError("invalid packed base-3 geometry")
    if scales.shape != (rows, cols // group_size):
        raise ValueError("invalid scale geometry")
    coded, anchor, levels, logarithmic = _scale_args(scales, meta)
    powers = _pow3(x.device, k)
    weight_levels = float((3 ** k - 1) // 2)
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    _base3_kernel[(triton.cdiv(rows, rows_per_program),)](
        packed, scales, x.contiguous(), out, powers, rows, cols, packed.stride(0),
        scales.stride(0), anchor, levels, weight_levels, group_size, SCALE_CODED=coded,
        SCALE_LOG=logarithmic, K=k, ROWS=rows_per_program, BLOCK=128, num_warps=warps)
    if bias is not None:
        out += bias.to(out.dtype)
    return out
