"""Matrix trial (remote L4 only): one-dot INT8 IMMA kernel versus the EXACT preserved swizzle binary and cuBLAS FP8.

Candidate: benchmarks/k5_group_imma_cuda.cu (new). Variant 0 = SUB 32 (m16n8k32),
variant 1 = SUB 16 (m16n8k16). Predeclared selection from the three activation
probes: SUB 32 for 8192x4096, 4096x4096 and 12288x4096; SUB 16 for 4096x12288.
Both variants are measured on every shape for the record.

Control: the preserved swizzle binary (sha f5d15c22...), opened read-only from
its recorded path; never rebuilt or copied over. Competitor: torch._scaled_mm
with E4M3 weights and dynamic per-tensor E4M3 activations (the TorchAO per-tensor
scheme used in the whole-model comparisons), timed with and without the
activation quantization.

Per shape: FP64 exact reference over all columns on the hash-checked real
9B matrices and saved prefill activations; relative L2 (gate <= 0.005 for the
candidate, unchanged), worst row, determinism (two runs byte-equal); then graph
timing: 20 calls per graph, 5 replays per sample, 9 rounds in ABCDE/EDCBA order,
GPU clock and power sampled around every sample. Medians, paired ratios, IQR.
Nothing existing is modified; outputs go to a new directory.
"""
from collections import Counter
from pathlib import Path
import argparse
import ctypes
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import probe_int8_affine_activation as base  # unchanged probe module: CASES, ACTIVATIONS, sha, unpack_hi

CONTROL_MAP = Path('/content/tk-speed9-1029/qwen35-9b-swizzle64-kernel-0929am-1031-2026/selection.candidate.json')
CONTROL_MAP_SHA = '4de7702c5b29e8f7cc554ffa90c8cd4d307ceae8ee078671f6da92b438869ddf'
CONTROL_SO_SHA = 'f5d15c220be4ec1669ffa87ebc0550226fcd72cf96fad811b8ec0cf4645b37bf'
CONTROL_VARIANT = 1  # swizzle64-8warps, the selection for all four shapes
SOURCE = HERE / 'k5_group_imma_cuda.cu'
GATE = 0.005
ORDER = ['12288x4096', '8192x4096', '4096x4096', '4096x12288']
SELECTED = {'12288x4096': 0, '8192x4096': 0, '4096x4096': 0, '4096x12288': 1}
VARIANT_NAME = {0: 'imma_sub32', 1: 'imma_sub16'}
SASS_OPS = ('IMMA.16832', 'IMMA.16816', 'HMMA', 'LDSM', 'LDG.E.128', 'STS.128', 'BAR.SYNC', 'FFMA', 'I2F', 'STL', 'LDL')
BANDWIDTH = 300e9  # L4 GDDR6, for the floor estimate only


def build(out):
    digest = base.sha(SOURCE)
    directory = Path('/content/tk-speed9-1029/build') / ('imma-sm89-' + digest[:16])
    directory.mkdir(parents=True, exist_ok=True)
    so, ptx = directory / 'kernel.so', directory / 'kernel.ptx'
    flags = ['nvcc', '-std=c++17', '-O3', '-arch=sm_89', '--fmad=false', str(SOURCE)]
    commands = [flags + ['--shared', '-Xcompiler', '-fPIC', '-Xptxas=-v', '-o', str(so)], flags + ['--ptx', '-o', str(ptx)]]
    logs = []
    for i, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        text = result.stdout + result.stderr
        (out / f'build-{i}.log').write_text(text)
        logs.append(text)
        if result.returncode:
            raise RuntimeError('Compilation failed: ' + text[-4000:])
    resources = {}
    current = None
    for line in logs[0].splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            current = m.group(1)
        m = re.search(r'Used (\d+) registers', line)
        if m and current:
            spill = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', logs[0][logs[0].find(current):])
            resources[current] = {'registers': int(m.group(1)),
                                  'spill_stores': int(spill.group(1)) if spill else None,
                                  'spill_loads': int(spill.group(2)) if spill else None}
    shutil.copyfile(so, out / 'candidate.so')
    shutil.copyfile(ptx, out / 'candidate.ptx')
    disasm = subprocess.run(['cuobjdump', '--dump-sass', str(so)], capture_output=True, text=True, timeout=120)
    (out / 'candidate.sass').write_text(disasm.stdout)
    counts = {op: disasm.stdout.count(op) for op in SASS_OPS}
    if counts['IMMA.16832'] == 0 or counts['IMMA.16816'] == 0:
        raise RuntimeError('Candidate SASS lacks IMMA instructions: ' + json.dumps(counts))
    return {'source_sha256': digest, 'so': str(so), 'so_sha256': base.sha(so), 'ptx_sha256': base.sha(ptx),
            'commands': commands, 'ptxas': resources, 'sass_counts': counts}


def open_control():
    if base.sha(CONTROL_MAP) != CONTROL_MAP_SHA:
        raise RuntimeError('Preserved swizzle map differs')
    record = json.loads(CONTROL_MAP.read_text())
    so = record['candidate_build']['so']
    if record['candidate_build']['so_sha256'] != CONTROL_SO_SHA or base.sha(so) != CONTROL_SO_SHA:
        raise RuntimeError('Preserved swizzle binary differs')
    lib = ctypes.CDLL(so)
    lib.initialize_k5_shapes.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.initialize_k5_shapes.restype = ctypes.c_int
    lib.launch_k5_shapes.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 +
                                     [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.launch_k5_shapes.restype = ctypes.c_int
    info = (ctypes.c_int * 7)()
    rc = lib.initialize_k5_shapes(CONTROL_VARIANT, info)
    if rc:
        raise RuntimeError('Control preflight failed: ' + str(rc))
    return lib, {'so': so, 'resources': list(info), 'selection': record['selection']}


def open_candidate(so):
    lib = ctypes.CDLL(so)
    lib.initialize_k5_imma.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.initialize_k5_imma.restype = ctypes.c_int
    lib.launch_k5_imma.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 7 + [ctypes.c_int] * 6 +
                                   [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.launch_k5_imma.restype = ctypes.c_int
    names = ['tile_m', 'tile_n', 'threads', 'shared_bytes', 'registers', 'local_bytes', 'active_blocks_per_sm',
             'quantize_registers', 'quantize_local_bytes']
    resources = {}
    for variant in (0, 1):
        info = (ctypes.c_int * 9)()
        rc = lib.initialize_k5_imma(variant, info)
        if rc:
            raise RuntimeError(f'Candidate preflight failed for variant {variant}: CUDA {rc}')
        resources[VARIANT_NAME[variant]] = dict(zip(names, info))
    return lib, resources


def scale_args(meta):
    mantissa, exponent, k, log = (int(v) for v in meta.tolist())
    return 1, float(mantissa) * 3.0 ** exponent, float((3 ** k - 1) // 2), int(log)


def control_launcher(torch, lib, codes, scales, x, out, args):
    coded, anchor, levels, log = args
    m, k = x.shape
    n = codes.shape[0]
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        rc = lib.launch_k5_shapes(CONTROL_VARIANT, codes.data_ptr(), scales.data_ptr(), x.data_ptr(), out.data_ptr(),
                                  m, n, k, codes.stride(0), scales.stride(0), x.stride(0), anchor, levels, coded, log,
                                  torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError('Control launch failed: ' + str(rc))
        return out
    return run


def candidate_launcher(torch, lib, variant, codes, scales, x, rsum, aq, ac, out, args):
    coded, anchor, levels, log = args
    m, k = x.shape
    n = codes.shape[0]

    def run():
        rc = lib.launch_k5_imma(variant, codes.data_ptr(), scales.data_ptr(), x.data_ptr(), rsum.data_ptr(),
                                aq.data_ptr(), ac.data_ptr(), out.data_ptr(), m, n, k, codes.stride(0),
                                scales.stride(0), x.stride(0), anchor, levels, coded, log,
                                torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError('Candidate launch failed: ' + str(rc))
        return out
    return run


def fp8_arms(torch, W, x, out):
    """TorchAO per-tensor scheme: E4M3 weights with amax/448 scale; dynamic per-tensor E4M3 activations."""
    wmax = float(W.abs().max())
    w_scale = torch.tensor(wmax / 448.0, device='cuda', dtype=torch.float32)
    wq = (W.float() / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()  # [n, k]
    wt = wq.t()  # [k, n] column-major view, as scaled_mm requires
    shapes = [(), (1,), (1, 1)]
    chosen = None
    for shape in shapes:
        try:
            sa = (x.float().abs().amax() / 448.0).reshape(shape)
            xq = (x.float() / sa).clamp(-448, 448).to(torch.float8_e4m3fn)
            torch._scaled_mm(xq, wt, scale_a=sa, scale_b=w_scale.reshape(shape), out_dtype=torch.float16, use_fast_accum=True)
            chosen = shape
            break
        except Exception as error:  # noqa: BLE001
            last = error
    if chosen is None:
        raise RuntimeError('torch._scaled_mm per-tensor call failed: ' + str(last))
    wscale = w_scale.reshape(chosen)
    xq_fixed = (x.float() / (x.float().abs().amax() / 448.0).reshape(chosen)).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    sa_fixed = (x.float().abs().amax() / 448.0).reshape(chosen).contiguous()

    def full():
        sa = (x.float().abs().amax() / 448.0).reshape(chosen)
        xq = (x.float() / sa).clamp(-448, 448).to(torch.float8_e4m3fn)
        out.copy_(torch._scaled_mm(xq, wt, scale_a=sa, scale_b=wscale, out_dtype=torch.float16, use_fast_accum=True))
        return out

    def mm_only():
        out.copy_(torch._scaled_mm(xq_fixed, wt, scale_a=sa_fixed, scale_b=wscale, out_dtype=torch.float16, use_fast_accum=True))
        return out
    return full, mm_only, {'scale_shape': list(chosen), 'weight_scale': float(w_scale), 'use_fast_accum': True,
                           'weight_bytes': wq.numel()}


def capture(torch, fn, calls=20):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    return graph


def gpu_state(nvml):
    if nvml is None:
        return None
    h, pynvml = nvml
    return {'clock_mhz': pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            'power_w': pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            'temp_c': pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)}


def timing(torch, arms, nvml, rounds=9):
    graphs = {name: capture(torch, fn) for name, fn in arms.items()}
    names = list(arms)
    order = names + names[::-1]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rows = []
    for round_id in range(rounds):
        for name in order:
            before = gpu_state(nvml)
            for _ in range(3):
                graphs[name].replay()
            start.record()
            for _ in range(5):
                graphs[name].replay()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / 100
            after = gpu_state(nvml)
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError('Invalid timing')
            rows.append({'round': round_id, 'arm': name, 'ms_per_matrix': ms, 'before': before, 'after': after})
    per_round = {name: [statistics.mean(r['ms_per_matrix'] for r in rows if r['arm'] == name and r['round'] == i)
                        for i in range(rounds)] for name in names}
    medians = {name: statistics.median(r['ms_per_matrix'] for r in rows if r['arm'] == name) for name in names}

    def ratio(a, b):
        values = sorted(x / y for x, y in zip(per_round[a], per_round[b]))
        return {'median': statistics.median(values), 'iqr': [values[len(values) // 4], values[-1 - len(values) // 4]]}
    clocks = {name: statistics.mean(r[s]['clock_mhz'] for r in rows if r['arm'] == name for s in ('before', 'after'))
              for name in names} if nvml else None
    return {'scope': '20 calls per graph; 5 replays per sample; 9 rounds; order ' + ','.join(order),
            'medians_ms': medians, 'mean_clock_mhz': clocks, 'raw': rows,
            'ratios': {f'{a}/{b}': ratio(a, b) for a in names for b in names if a != b and b in ('swizzle', 'fp8_full')}}


def main(out):
    import torch
    from safetensors.torch import load_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    if torch.cuda.get_device_capability() != (8, 9) or 'L4' not in torch.cuda.get_device_name():
        raise RuntimeError('Requires the assigned L4')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = (pynvml.nvmlDeviceGetHandleByIndex(0), pynvml)
    except Exception:  # noqa: BLE001
        pass
    if base.sha(base.ACTIVATIONS) != base.ACTIVATIONS_SHA:
        raise RuntimeError('Activation fixture hash differs')
    acts = load_file(base.ACTIVATIONS)
    build_record = build(out)
    control_lib, control_record = open_control()
    candidate_lib, candidate_resources = open_candidate(build_record['so'])
    result = {'scope': __doc__, 'gate_relative_l2': GATE, 'selected_variant': {k: VARIANT_NAME[v] for k, v in SELECTED.items()},
              'candidate_build': build_record, 'candidate_resources': candidate_resources, 'control': control_record,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'cases': {}}
    print('BUILD', json.dumps({'ptxas': build_record['ptxas'], 'resources': candidate_resources, 'sass': build_record['sass_counts']}), flush=True)
    with torch.inference_mode():
        for label in ORDER:
            path, want_sha, occurrences = base.CASES[label]
            if base.sha(path) != want_sha:
                raise RuntimeError('Matrix hash differs: ' + label)
            data = load_file(path)
            n, k = (int(v) for v in label.split('x'))
            raw = data['codes'].reshape(n, k)
            if raw.dtype != torch.uint8 or int(raw.max()) > 242:
                raise RuntimeError('Invalid K5 codes: ' + label)
            codes = (raw.to(torch.int16) - 121).to(torch.int8).cuda().contiguous()
            scales = data['scales'].reshape(n, k // 128).cuda().contiguous()
            args = scale_args(data['meta'])
            S = base.unpack_hi(data['scales'].reshape(n, k // 128), data['meta']).cuda()
            W = (codes.double().reshape(n, -1, 128) / 121.0 * S[..., None]).reshape(n, k)
            x = acts['shape-' + label].cuda().contiguous()
            if list(x.shape) != [128, k] or x.dtype != torch.float16:
                raise RuntimeError('Invalid activation: ' + label)
            m = x.shape[0]
            exact = torch.nn.functional.linear(x.double(), W)
            norm = exact.norm()
            rsum = {sub: codes.to(torch.int32).reshape(n, k // sub, sub).sum(-1).to(torch.int16).contiguous() for sub in (32, 16)}
            if any(int(v.to(torch.int32).abs().max()) > 121 * s for s, v in rsum.items()):
                raise RuntimeError('Code sum out of range')
            outputs = {name: torch.empty((m, n), device='cuda', dtype=torch.float16) for name in
                       ('swizzle', 'imma_sub32', 'imma_sub16', 'fp8_full', 'fp8_mm')}
            aq = {sub: torch.empty((m, k), device='cuda', dtype=torch.int8) for sub in (32, 16)}
            ac = {sub: torch.empty((m, k // sub, 2), device='cuda', dtype=torch.float32) for sub in (32, 16)}
            arms = {'swizzle': control_launcher(torch, control_lib, codes, scales, x, outputs['swizzle'], args),
                    'imma_sub32': candidate_launcher(torch, candidate_lib, 0, codes, scales, x, rsum[32], aq[32], ac[32], outputs['imma_sub32'], args),
                    'imma_sub16': candidate_launcher(torch, candidate_lib, 1, codes, scales, x, rsum[16], aq[16], ac[16], outputs['imma_sub16'], args)}
            full, mm_only, fp8_info = fp8_arms(torch, W, x, outputs['fp8_full'])
            arms['fp8_full'] = full
            arms['fp8_mm'] = lambda: mm_only()
            accuracy = {}
            for name, fn in arms.items():
                first = fn().clone()
                torch.cuda.synchronize()
                second = fn().clone()
                torch.cuda.synchronize()
                y = first.double()
                rel = float((y - exact).norm() / norm)
                worst = float(((y - exact).norm(dim=1) / exact.norm(dim=1).clamp_min(1e-30)).max())
                accuracy[name] = {'relative_l2': rel, 'worst_row_relative_l2': worst, 'passes_gate': rel <= GATE,
                                  'deterministic': bool(torch.equal(first.view(torch.uint8), second.view(torch.uint8))),
                                  'finite': bool(torch.isfinite(first).all())}
            fp8_info['activation_fp8_relative_l2_vs_fp64'] = accuracy['fp8_full']['relative_l2']
            # Quantized activation sanity: reconstruct x from (A, a, c) and compare with the probe's method.
            recon = {}
            for sub in (32, 16):
                arms[f'imma_sub{32 if sub == 32 else 16}']()
                torch.cuda.synchronize()
                a_ = ac[sub][..., 0].double()[..., None]
                c_ = ac[sub][..., 1].double()[..., None]
                xhat = ((aq[sub].double().reshape(m, k // sub, sub) + c_) * a_).reshape(m, k)
                recon[f'sub{sub}'] = {'activation_relative_l2': float((xhat - x.double()).norm() / x.double().norm()),
                                      'max_abs_c': float(ac[sub][..., 1].abs().max()), 'min_a': float(ac[sub][..., 0].min())}
            torch.cuda.synchronize()
            timed = timing(torch, arms, nvml)
            selected = VARIANT_NAME[SELECTED[label]]
            floor_us = {'codes_bytes_over_300GBps_us': n * k / BANDWIDTH * 1e6,
                        'imma_bytes_us': (n * k + m * k + 2 * n * k // 32 + 2 * n * k // 128 + m * k // 32 * 8) / BANDWIDTH * 1e6,
                        'swizzle_bytes_us': (n * k + 2 * m * k + 2 * n * k // 128) / BANDWIDTH * 1e6}
            case = {'model_occurrences': occurrences, 'selected': selected, 'accuracy': accuracy, 'fp8': fp8_info,
                    'activation_reconstruction': recon, 'timing': timed, 'floor': floor_us}
            result['cases'][label] = case
            med = timed['medians_ms']
            print('CASE', label, json.dumps({
                'rel_l2': {k_: round(v['relative_l2'], 6) for k_, v in accuracy.items()},
                'det': {k_: v['deterministic'] for k_, v in accuracy.items()},
                'ms': {k_: round(v, 4) for k_, v in med.items()},
                'ratios': {k_: round(v['median'], 4) for k_, v in timed['ratios'].items()},
                'clock': timed['mean_clock_mhz'], 'floor_us': {k_: round(v, 1) for k_, v in floor_us.items()}}), flush=True)
            del W, exact, data, outputs, aq, ac, rsum, arms, full, mm_only
            torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    result['script_sha256'] = base.sha(__file__)
    result['base_script_sha256'] = base.sha(base.__file__)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    verdict = {label: {'selected': c['selected'], 'selected_passes_gate': c['accuracy'][c['selected']]['passes_gate'],
                       'selected_over_swizzle': c['timing']['ratios'][c['selected'] + '/swizzle']['median'],
                       'selected_over_fp8_full': c['timing']['ratios'][c['selected'] + '/fp8_full']['median']}
               for label, c in result['cases'].items()}
    print('IMMA_MATRIX_DONE', json.dumps(verdict), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    main(parser.parse_args().out)
