"""Opt-in native CUDA/WMMA follow-up for the exact saved k5/g128 artifact."""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

import torch
from k5_prefill_tiled import reference

VARIANTS = ("group_fp16", "group_int8", "group_int8_residual")
COUNTERS = {v: 0 for v in VARIANTS}
COUNTERS["original_decode"] = 0
_LIB = None
_BUILD = None


def native_build():
    global _LIB, _BUILD
    if _LIB is not None: return _BUILD
    if not Path("/kaggle").is_dir() or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("This bounded CUDA experiment targets the assigned T4/sm75 only")
    src = Path(__file__).with_name("k5_group_cuda.cu")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    directory = Path("/kaggle/temp") / ("tk-wmma-" + digest[:16])
    directory.mkdir(exist_ok=True)
    so = directory / "kernel.so"
    ptx = directory / "kernel.ptx"
    commands = [["nvcc", "-std=c++17", "-O3", "-arch=sm_75", "--fmad=false", str(src),
                 "--shared", "-Xcompiler", "-fPIC", "-o", str(so)],
                ["nvcc", "-std=c++17", "-O3", "-arch=sm_75", "--fmad=false", str(src), "--ptx", "-o", str(ptx)]]
    if not (so.is_file() and ptx.is_file()):
        for i, cmd in enumerate(commands):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            (directory / f"build-{i}.log").write_text(result.stdout + result.stderr)
            if result.returncode: raise RuntimeError("Native WMMA compilation failed: " + result.stderr[-5000:])
    lib = ctypes.CDLL(str(so))
    lib.launch_k5_group.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 +
                                   [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.launch_k5_group.restype = ctypes.c_int
    _LIB = lib
    _BUILD = {"source_sha256": digest, "so": str(so), "ptx": str(ptx), "commands": commands,
              "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest(), "architecture": "sm_75"}
    return _BUILD


def gemv_k5(*args, **kwargs):
    COUNTERS["original_decode"] += 1
    return reference().gemv_k5(*args, **kwargs)


def gemv_k5_rows(*args, **kwargs): return reference().gemv_k5_rows(*args, **kwargs)


def gemm_k5(codes, scales, X, meta=None, bias=None, out_dtype=None, *, variant=None):
    variant = variant or os.environ.get("TK_K5_HW_VARIANT", "group_fp16")
    if variant not in VARIANTS: raise ValueError("Unknown native k5 variant")
    if not all(t.is_cuda for t in (codes, scales, X)) or codes.ndim != 2 or X.ndim != 2:
        raise ValueError("Requires CUDA signed k5 codes and FP16 X[M,K]")
    n, k = codes.shape
    m = X.shape[0]
    if (codes.dtype != torch.int8 or scales.dtype != torch.int16 or X.dtype != torch.float16 or
        k % 128 or m < 1 or X.shape[1] != k or scales.shape != (n, k // 128) or
        codes.stride(1) != 1 or scales.stride(1) != 1 or
        not (codes.device == scales.device == X.device)):
        raise ValueError("Native test supports this exact signed k5/g128/int16-scale artifact only")
    if m == 1: return gemv_k5(codes, scales, X[0], meta=meta, bias=bias, out_dtype=out_dtype)[None]
    if m <= 16: return gemv_k5_rows(codes, scales, X, meta=meta, bias=bias, out_dtype=out_dtype)
    if out_dtype is not None and out_dtype != torch.float16:
        raise ValueError("Native test output is FP16")
    coded, anchor, levels, log = reference()._scale_args(scales, meta)
    with torch.cuda.device(X.device):
        native_build()
        X = X.contiguous()
        y = torch.empty((m, n), device=X.device, dtype=torch.float16)
        rc = _LIB.launch_k5_group(VARIANTS.index(variant), codes.data_ptr(), scales.data_ptr(), X.data_ptr(), y.data_ptr(),
                m, n, k, codes.stride(0), scales.stride(0), X.stride(0), anchor, levels, coded, log,
                torch.cuda.current_stream(X.device).cuda_stream)
        if rc: raise RuntimeError(f"Native k5 launch refused with CUDA error {rc}")
    if bias is not None: y += bias.to(y.dtype)
    COUNTERS[variant] += 1
    return y
