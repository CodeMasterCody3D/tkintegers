#!/usr/bin/env python3
"""Isolated K5 operator benchmark around the 16-row dispatch boundary.

Run on the assigned Colab L4 or Kaggle T4 from the repository root:
  python benchmarks/luna_two_perf_baseline.py --adapter l4
  python benchmarks/luna_two_perf_baseline.py --adapter t4

Default inputs are deterministic synthetic K5 tensors. This measures operator
dispatch and overhead, not whole-model speed or answer quality. For representative
weights, pass --fixture with a tensor-only .pt dictionary containing codes,
scales, meta, and optional bias. Activations are generated deterministically.

Each first-call sample runs in its own fresh process, so one batch size cannot
warm another batch size in Python/Triton state. Persistent compiler caches are
left intact; first-call timing includes Triton/native setup only when uncached.
Warm timing is synchronized Python wall time after warmup and includes copies,
bias addition, and kernel completion. No kernel source is changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
BATCHES_DEFAULT = "1,2,15,16,17,32"
MARKER = "K5_BASELINE_JSON="


def parse_batches(value: str) -> list[int]:
    try:
        batches = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from exc
    if not batches or any(batch < 1 for batch in batches):
        raise argparse.ArgumentTypeError("batch sizes must all be positive")
    return batches


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("l4", "t4"), default="l4")
    parser.add_argument("--batches", type=parse_batches, default=parse_batches(BATCHES_DEFAULT))
    parser.add_argument("--n", type=int, default=4096, help="output channels for synthetic inputs")
    parser.add_argument("--k", type=int, default=4096, help="input width; must be divisible by 128")
    parser.add_argument("--fixture", type=Path, help="tensor-only .pt dict: codes, scales, meta, optional bias")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, help="write JSON results here instead of stdout")
    parser.add_argument("--cold-child", type=int, help=argparse.SUPPRESS)
    return parser


def load_weights(args: argparse.Namespace, device: torch.device):
    if args.fixture:
        fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
        required = {"codes", "scales", "meta"}
        missing = required.difference(fixture)
        if missing:
            raise ValueError(f"fixture is missing keys: {sorted(missing)}")
        codes = fixture["codes"].to(device=device, dtype=torch.int8).contiguous()
        scales = fixture["scales"].to(device=device, dtype=torch.int16).contiguous()
        meta = fixture["meta"].to(device="cpu", dtype=torch.int64).reshape(-1)
        bias_cpu = fixture.get("bias")
        bias = None if bias_cpu is None else bias_cpu.to(device=device, dtype=torch.float16).contiguous()
    else:
        if args.k % 128:
            raise ValueError("--k must be divisible by 128")
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        codes = torch.randint(-121, 122, (args.n, args.k), generator=generator,
                              dtype=torch.int16).to(torch.int8).to(device).contiguous()
        scale_codes = torch.randint(-29524, 29525, (args.n, args.k // 128),
                                    generator=generator, dtype=torch.int32)
        scales = scale_codes.to(torch.int16).to(device).contiguous()
        meta = torch.tensor([1, 0, 10, 1], dtype=torch.int64)
        bias = None

    if codes.ndim != 2 or scales.shape != (codes.shape[0], codes.shape[1] // 128):
        raise ValueError("fixture shapes must be codes[N,K] and scales[N,K/128]")
    if codes.shape[1] % 128:
        raise ValueError("K must be divisible by 128")
    if meta.numel() != 4:
        raise ValueError("meta must contain [anchor_mantissa, anchor_exponent, k, is_log]")
    if codes.numel() and (int(codes.min().item()) < -121 or int(codes.max().item()) > 121):
        raise ValueError("signed K5 codes must be in [-121, 121]")
    return codes, scales, meta, bias


def make_activation(batch: int, width: int, device: torch.device, seed: int,
                    strided: bool = False) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((batch, width), generator=generator, dtype=torch.float32).to(torch.float16)
    x = x.to(device)
    if not strided:
        return x.contiguous()
    storage = torch.empty((batch, width * 2), device=device, dtype=torch.float16)
    storage[:, ::2].copy_(x)
    return storage[:, ::2]


def load_adapter(name: str):
    os.environ["TK_K5_REFERENCE"] = str(HERE / "gemv_reference.py")
    if name == "t4":
        os.environ["TK_K5_HW_VARIANT"] = "group_fp16"
        import k5_prefill_cuda as kernel
        expected = (7, 5)
    else:
        import k5_prefill_cuda_l4 as kernel
        expected = (8, 9)
    actual = torch.cuda.get_device_capability()
    if actual != expected:
        raise RuntimeError(f"{name.upper()} adapter requires compute capability {expected}; found {actual}")
    return kernel


def call_kernel(kernel, codes, scales, x, meta, bias, out_dtype):
    return kernel.gemm_k5(codes, scales, x, meta=meta, bias=bias, out_dtype=out_dtype)


def output_error(kernel, codes, scales, x, meta, bias, out, out_dtype):
    reference = kernel.reference().gemm_k5(
        codes, scales, x, meta=meta, bias=bias, out_dtype=out_dtype or out.dtype
    )
    torch.cuda.synchronize()
    delta = out.float() - reference.float()
    ref_norm = torch.linalg.vector_norm(reference.float()).item()
    return {
        "max_abs": delta.abs().max().item(),
        "rms": torch.sqrt(torch.mean(delta.square())).item(),
        "relative_l2": torch.linalg.vector_norm(delta).item() / max(ref_norm, 1e-30),
    }


def path_for(batch: int) -> str:
    if batch == 1:
        return "original_single_row_decode"
    if batch <= 16:
        return "original_batched_decode"
    return "native_prefill"


def environment(kernel, device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "adapter": kernel.__name__,
    }


def first_call(args: argparse.Namespace, batch: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on the assigned remote GPU")
    device = torch.device("cuda:0")
    kernel = load_adapter(args.adapter)
    codes, scales, meta, _ = load_weights(args, device)
    x = make_activation(batch, codes.shape[1], device, args.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = call_kernel(kernel, codes, scales, x, meta, None, None)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    error = output_error(kernel, codes, scales, x, meta, None, out, None)
    build = getattr(kernel, "_BUILD", None)
    return {
        "batch_rows_M": batch,
        "path": path_for(batch),
        "first_call_ms": elapsed_ms,
        "output_dtype": str(out.dtype),
        "output_error_vs_frozen_reference": error,
        "native_build": None if build is None else {
            key: build.get(key) for key in ("architecture", "source_sha256", "so_sha256")
        },
    }


def warm_case(kernel, args, codes, scales, meta, fixture_bias, batch: int,
              name: str, use_bias: bool = False, strided: bool = False,
              fp16_output: bool = False) -> dict:
    device = codes.device
    x = make_activation(batch, codes.shape[1], device, args.seed, strided=strided)
    bias = fixture_bias
    if use_bias and bias is None:
        bias = torch.zeros(codes.shape[0], device=device, dtype=torch.float16)
    out_dtype = torch.float16 if fp16_output else None

    for _ in range(args.warmup):
        call_kernel(kernel, codes, scales, x, meta, bias, out_dtype)
    torch.cuda.synchronize()

    samples = []
    out = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = call_kernel(kernel, codes, scales, x, meta, bias, out_dtype)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

    assert out is not None
    samples.sort()
    p95 = samples[max(0, math.ceil(0.95 * len(samples)) - 1)]
    error = output_error(kernel, codes, scales, x, meta, bias, out, out_dtype)
    return {
        "case": name,
        "batch_rows_M": batch,
        "path": path_for(batch),
        "input_contiguous": x.is_contiguous(),
        "bias": bias is not None,
        "requested_output_dtype": "float16" if fp16_output else "adapter_default",
        "output_dtype": str(out.dtype),
        "warm_wall_ms_median": statistics.median(samples),
        "warm_wall_ms_p95": p95,
        "warm_wall_ms_min": samples[0],
        "output_error_vs_frozen_reference": error,
    }


def run_parent(args: argparse.Namespace) -> dict:
    script = str(Path(__file__).resolve())
    cold_rows = []
    for batch in args.batches:
        command = [
            sys.executable, script, "--adapter", args.adapter, "--batches", str(batch),
            "--n", str(args.n), "--k", str(args.k), "--seed", str(args.seed),
            "--cold-child", str(batch),
        ]
        if args.fixture:
            command += ["--fixture", str(args.fixture.resolve())]
        child = subprocess.run(command, capture_output=True, text=True)
        if child.returncode:
            raise RuntimeError(
                f"first-call subprocess failed for M={batch}:\n{child.stderr or child.stdout}"
            )
        line = next((line for line in reversed(child.stdout.splitlines())
                     if line.startswith(MARKER)), None)
        if line is None:
            raise RuntimeError(f"first-call subprocess returned no result for M={batch}")
        cold_rows.append(json.loads(line[len(MARKER):]))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on the assigned remote GPU")
    device = torch.device("cuda:0")
    kernel = load_adapter(args.adapter)
    codes, scales, meta, fixture_bias = load_weights(args, device)
    warm_rows = []
    for batch in args.batches:
        warm_rows.append(warm_case(kernel, args, codes, scales, meta, fixture_bias,
                                   batch, "default_contiguous_no_bias"))
        if batch in (16, 17):
            warm_rows.append(warm_case(kernel, args, codes, scales, meta, fixture_bias,
                                       batch, "with_bias", use_bias=True))
            warm_rows.append(warm_case(kernel, args, codes, scales, meta, fixture_bias,
                                       batch, "strided_input", strided=True))
            warm_rows.append(warm_case(kernel, args, codes, scales, meta, fixture_bias,
                                       batch, "uniform_fp16_output", fp16_output=True))

    result = {
        "description": "K5 operator microbenchmark; synthetic by default, not whole-model performance",
        "dispatch_note": "The adapter dispatches on activation rows M: M<=16 uses the frozen decoder; M>16 uses native prefill.",
        "first_call_definition": "one adapter call in a fresh process after input setup; persistent compiler caches are not cleared",
        "warm_definition": f"{args.warmup} warmups followed by {args.repeats} synchronized calls; median and p95 wall time",
        "weight_shape_NK": list(codes.shape),
        "cold_first_calls": cold_rows,
        "warm_cases": warm_rows,
        "environment": environment(kernel, device),
    }
    return result


def main() -> None:
    args = make_parser().parse_args()
    if args.k % 128:
        raise SystemExit("--k must be divisible by 128")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("--warmup must be nonnegative and --repeats must be positive")
    if args.cold_child is not None:
        row = first_call(args, args.cold_child)
        print(MARKER + json.dumps(row, separators=(",", ":")))
        return
    result = run_parent(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
