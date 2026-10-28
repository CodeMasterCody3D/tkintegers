"""Opt-in packed k5 prefill experiment. Original codecs and decode stay intact.

Decode integer scales inside bounded weight tiles, round the reconstructed
weights to the declared FP16 execution dtype, and use tensor-core matmul.
No resident floating weight cache or full-matrix dequantization fallback.
Requires TK_K5_REFERENCE pointing to the frozen original kernel module.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


_REF = None
COUNTERS = {"prefill_tiled": 0, "batched_decode": 0, "original_decode": 0}


def reference():
    global _REF
    if _REF is None:
        path = Path(os.environ.get("TK_K5_REFERENCE", Path(__file__).with_name("gemv_reference.py")))
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            raise RuntimeError("Set TK_K5_REFERENCE to the preserved original gemv_triton module")
        spec = importlib.util.spec_from_file_location("tk_k5_prefill_reference", path)
        _REF = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_REF)
    return _REF


@triton.autotune(configs=[
    triton.Config({"BM": 16, "BN": 64, "BK": 64}, num_warps=4, num_stages=2),
    triton.Config({"BM": 32, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
    triton.Config({"BM": 32, "BN": 64, "BK": 128}, num_warps=4, num_stages=2),
    triton.Config({"BM": 32, "BN": 128, "BK": 64}, num_warps=8, num_stages=2),
], key=["M", "N", "K"])
@triton.jit
def _k5_prefill_tiled(C, S, X, Y, M, N, K, CS, SS, XS, YS, anchor, levels,
                      CODED: tl.constexpr, LOG: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, tl.cdiv(K, BK)):
        k = start * BK + rk
        x = tl.load(X + m[:, None] * XS + k[None, :],
                    mask=(m[:, None] < M) & (k[None, :] < K), other=0)
        q = tl.load(C + n[None, :] * CS + k[:, None],
                    mask=(n[None, :] < N) & (k[:, None] < K), other=0).to(tl.float32)
        s = tl.load(S + n[None, :] * SS + k[:, None] // 128,
                    mask=(n[None, :] < N) & (k[:, None] < K), other=0).to(tl.float32)
        if CODED:
            s = s / levels * anchor
            if LOG:
                s = tl.exp2(s)
        w = (tl.div_rn(q, 121.0) * s).to(tl.float16)
        acc += tl.dot(x, w, out_dtype=tl.float32)
    tl.store(Y + m[:, None] * YS + n[None, :], acc.to(Y.dtype.element_ty),
             mask=(m[:, None] < M) & (n[None, :] < N))


def gemv_k5(*args, **kwargs):
    COUNTERS["original_decode"] += 1
    return reference().gemv_k5(*args, **kwargs)


def gemv_k5_rows(*args, **kwargs):
    COUNTERS["batched_decode"] += 1
    return reference().gemv_k5_rows(*args, **kwargs)


def gemm_k5(codes, scales, X, meta=None, bias=None, out_dtype=None):
    if not all(t.is_cuda for t in (codes, scales, X)) or X.ndim != 2:
        raise ValueError("k5 prefill requires CUDA tensors and X [M, K]")
    n, k = map(int, codes.shape)
    m = int(X.shape[0])
    if (codes.dtype != torch.int8 or X.dtype != torch.float16 or k % 128 or m < 1 or
            X.shape[1] != k or scales.shape != (n, k // 128)):
        raise ValueError("This candidate requires signed k5/group128 and FP16 activations")
    if not (codes.device == scales.device == X.device):
        raise ValueError("k5 tensors must share one CUDA device")
    if m == 1:
        return gemv_k5(codes, scales, X[0], meta=meta, bias=bias, out_dtype=out_dtype).unsqueeze(0)
    if m <= 16:
        return gemv_k5_rows(codes, scales, X, meta=meta, bias=bias, out_dtype=out_dtype)
    coded, anchor, levels, log = reference()._scale_args(scales, meta)
    X = X.contiguous()
    out = torch.empty((m, n), device=X.device, dtype=out_dtype or torch.float32)
    with torch.cuda.device(X.device):
        _k5_prefill_tiled[lambda p: (triton.cdiv(m, p["BM"]), triton.cdiv(n, p["BN"]))](
            codes, scales, X, out, m, n, k, codes.stride(0), scales.stride(0), X.stride(0), out.stride(0),
            anchor, levels, CODED=coded, LOG=log)
    if bias is not None:
        out += bias.to(out.dtype)
    COUNTERS["prefill_tiled"] += 1
    return out
