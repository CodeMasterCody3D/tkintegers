"""Isolated L4/sm89 adapter; original T4 wrapper and CUDA arithmetic unchanged."""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess

import torch
from k5_prefill_tiled import reference

COUNTERS = {"native_prefill": 0, "original_decode": 0}
_LIB = None
_BUILD = None


def native_build():
    global _LIB, _BUILD
    if _LIB is not None: return _BUILD
    if not Path('/content/drive/MyDrive').is_dir() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError('This adapter requires the assigned Colab L4/sm89 and mounted Drive')
    src = Path(__file__).with_name('k5_group_cuda.cu')
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    directory = Path('/content/tk-fp8-l4-1027/build') / ('sm89-' + digest[:16])
    directory.mkdir(parents=True, exist_ok=True)
    so, ptx = directory / 'kernel.so', directory / 'kernel.ptx'
    commands = [['nvcc', '-std=c++17', '-O3', '-arch=sm_89', '--fmad=false', str(src),
                 '--shared', '-Xcompiler', '-fPIC', '-o', str(so)],
                ['nvcc', '-std=c++17', '-O3', '-arch=sm_89', '--fmad=false', str(src), '--ptx', '-o', str(ptx)]]
    if not (so.is_file() and ptx.is_file()):
        for i, command in enumerate(commands):
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            (directory / f'build-{i}.log').write_text(result.stdout + result.stderr)
            if result.returncode: raise RuntimeError('L4 compilation failed: ' + result.stderr[-3000:])
    lib = ctypes.CDLL(str(so))
    lib.launch_k5_group.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 +
                                   [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.launch_k5_group.restype = ctypes.c_int
    _LIB = lib
    _BUILD = {'source_sha256': digest, 'so': str(so), 'ptx': str(ptx), 'commands': commands,
              'so_sha256': hashlib.sha256(so.read_bytes()).hexdigest(), 'architecture': 'sm_89'}
    return _BUILD


def gemv_k5(*args, **kwargs):
    COUNTERS['original_decode'] += 1
    return reference().gemv_k5(*args, **kwargs)


def gemv_k5_rows(*args, **kwargs): return reference().gemv_k5_rows(*args, **kwargs)


def gemm_k5(codes, scales, X, meta=None, bias=None, out_dtype=None):
    if not all(t.is_cuda for t in (codes, scales, X)) or codes.ndim != 2 or X.ndim != 2:
        raise ValueError('Requires CUDA signed k5 codes and FP16 X[M,K]')
    n, k = codes.shape
    m = X.shape[0]
    if (codes.dtype != torch.int8 or scales.dtype != torch.int16 or X.dtype != torch.float16 or
        k % 128 or m < 1 or X.shape[1] != k or scales.shape != (n, k // 128) or
        codes.stride(1) != 1 or scales.stride(1) != 1 or
        not (codes.device == scales.device == X.device)):
        raise ValueError('This adapter supports the saved k5/g128/int16-scale artifact only')
    if m == 1: return gemv_k5(codes, scales, X[0], meta=meta, bias=bias, out_dtype=out_dtype)[None]
    if m <= 16: return gemv_k5_rows(codes, scales, X, meta=meta, bias=bias, out_dtype=out_dtype)
    if out_dtype is not None and out_dtype != torch.float16: raise ValueError('FP16 output required')
    coded, anchor, levels, log = reference()._scale_args(scales, meta)
    with torch.cuda.device(X.device):
        native_build()
        X = X.contiguous()
        y = torch.empty((m, n), device=X.device, dtype=torch.float16)
        rc = _LIB.launch_k5_group(0, codes.data_ptr(), scales.data_ptr(), X.data_ptr(), y.data_ptr(),
                m, n, k, codes.stride(0), scales.stride(0), X.stride(0), anchor, levels, coded, log,
                torch.cuda.current_stream(X.device).cuda_stream)
        if rc: raise RuntimeError(f'L4 k5 launch returned CUDA error {rc}')
    if bias is not None: y += bias.to(y.dtype)
    COUNTERS['native_prefill'] += 1
    return y
