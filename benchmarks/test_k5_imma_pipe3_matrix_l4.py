"""Matrix trial (remote L4 only): INT8 IMMA pipe v3 (32x32 warp tiles, constant-C flush) versus swizzle, INT8 v1, pipe v2 and cuBLAS FP8.

Candidate: benchmarks/k5_group_imma_pipe3_cuda.cu (new). Variants: SUB 32 with
3/4 stages EXACT flush (3 ops/value), SUB 32 with 3/4/6 stages FAST flush
(2 ops/value, bias removed per group), SUB 16 with 3 stages EXACT and FAST.
Predeclared primary: SUB 32 x 3 stages EXACT (3 CTAs/SM); FAST variants are
reported with their own gate result and are not assumed byte-equal to anything.

Controls: preserved swizzle binary f5d15c22..., INT8 v1 binary, pipe v2 binary
(SUB 32 x 5 stages), all hash-checked. Competitor: torch._scaled_mm E4M3
per-tensor with and without activation quantization. Same accuracy checks and
graph timing as the earlier trials. Nothing existing is modified.
"""
from pathlib import Path
import argparse
import ctypes
import json
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import probe_int8_affine_activation as base
import test_k5_imma_matrix_l4 as T
import test_k5_imma_pipe2_matrix_l4 as P2

SOURCE = HERE / 'k5_group_imma_pipe3_cuda.cu'
V1_RESULT = Path('/content/tk-speed9-1029/int8-imma-matrix-v1/result.json')
P2_RESULT = Path('/content/tk-speed9-1029/int8-imma-pipe2-matrix-v1/result.json')
VARIANTS = {'pipe3_32_s3_exact': 32031, 'pipe3_32_s4_exact': 32041, 'pipe3_32_s3_fast': 32030, 'pipe3_32_s4_fast': 32040,
            'pipe3_32_s6_fast': 32060, 'pipe3_16_s3_exact': 16031, 'pipe3_16_s3_fast': 16030}
PRIMARY = {'12288x4096': 'pipe3_32_s3_exact', '8192x4096': 'pipe3_32_s3_exact', '4096x4096': 'pipe3_32_s3_exact',
           '4096x12288': 'pipe3_16_s3_exact'}
SASS_OPS = ('IMMA.16832', 'IMMA.16816', 'HMMA', 'LDSM', 'LDGSTS', 'BAR.SYNC', 'DEPBAR', 'FFMA', 'FADD', 'IADD3', 'STL', 'LDL')


def build(out):
    digest = base.sha(SOURCE)
    directory = Path('/content/tk-speed9-1029/build') / ('imma-pipe3-sm89-' + digest[:16])
    directory.mkdir(parents=True, exist_ok=True)
    so, ptx = directory / 'kernel.so', directory / 'kernel.ptx'
    flags = ['nvcc', '-std=c++17', '-O3', '-arch=sm_89', '--fmad=false', str(SOURCE)]
    commands = [flags + ['--shared', '-Xcompiler', '-fPIC', '-Xptxas=-v', '-o', str(so)], flags + ['--ptx', '-o', str(ptx)]]
    logs = []
    for i, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        text = result.stdout + result.stderr
        (out / f'build-{i}.log').write_text(text)
        logs.append(text)
        if result.returncode:
            raise RuntimeError('Compilation failed: ' + text[-4000:])
    resources = {}
    for block in logs[0].split("Compiling entry function '")[1:]:
        name = block.split("'")[0]
        regs = re.search(r'Used (\d+) registers', block)
        spill = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', block)
        resources[name] = {'registers': int(regs.group(1)) if regs else None,
                           'spill_stores': int(spill.group(1)) if spill else None,
                           'spill_loads': int(spill.group(2)) if spill else None}
    shutil.copyfile(so, out / 'candidate.so')
    shutil.copyfile(ptx, out / 'candidate.ptx')
    disasm = subprocess.run(['cuobjdump', '--dump-sass', str(so)], capture_output=True, text=True, timeout=120)
    (out / 'candidate.sass').write_text(disasm.stdout)
    counts = {op: disasm.stdout.count(op) for op in SASS_OPS}
    if counts['LDGSTS'] == 0 or counts['IMMA.16832'] == 0:
        raise RuntimeError('Candidate SASS lacks cp.async (LDGSTS) or IMMA: ' + json.dumps(counts))
    return {'source_sha256': digest, 'so': str(so), 'so_sha256': base.sha(so), 'ptx_sha256': base.sha(ptx),
            'commands': commands, 'ptxas': resources, 'sass_counts': counts}


def open_pipe3(so):
    lib = ctypes.CDLL(so)
    lib.initialize_k5_imma_pipe3.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.initialize_k5_imma_pipe3.restype = ctypes.c_int
    lib.prepare_k5_imma_pipe3.argtypes = ([ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.prepare_k5_imma_pipe3.restype = ctypes.c_int
    lib.launch_k5_imma_pipe3.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p])
    lib.launch_k5_imma_pipe3.restype = ctypes.c_int
    names = ['tile_m', 'tile_n', 'threads', 'shared_bytes', 'registers', 'local_bytes', 'active_blocks_per_sm',
             'quantize_registers', 'quantize_local_bytes']
    resources = {}
    for name, variant in VARIANTS.items():
        info = (ctypes.c_int * 9)()
        rc = lib.initialize_k5_imma_pipe3(variant, info)
        if rc:
            raise RuntimeError(f'Pipe3 preflight failed for {name}: CUDA {rc}')
        resources[name] = dict(zip(names, info))
    return lib, resources


def prepare_scales(torch, lib, scales, k, args):
    coded, anchor, levels, log = args
    n = scales.shape[0]
    scales_f = torch.empty((k // 128, n), device='cuda', dtype=torch.float32)
    rc = lib.prepare_k5_imma_pipe3(scales.data_ptr(), scales_f.data_ptr(), n, k, scales.stride(0), anchor, levels, coded, log,
                                   torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError('prepare failed: ' + str(rc))
    torch.cuda.synchronize()
    return scales_f


def pipe3_launcher(torch, lib, variant, codes, scales_f, x, rsum_f, aq, ac, out):
    m, k = x.shape
    n = codes.shape[0]

    def run():
        rc = lib.launch_k5_imma_pipe3(variant, codes.data_ptr(), scales_f.data_ptr(), x.data_ptr(), rsum_f.data_ptr(),
                                      aq.data_ptr(), ac.data_ptr(), out.data_ptr(), m, n, k, codes.stride(0), x.stride(0),
                                      torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError('Pipe3 launch failed: ' + str(rc))
        return out
    return run


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
    v1 = json.loads(V1_RESULT.read_text())['candidate_build']
    p2 = json.loads(P2_RESULT.read_text())['candidate_build']
    for rec in (v1, p2):
        if base.sha(rec['so']) != rec['so_sha256']:
            raise RuntimeError('Earlier binary differs from its record: ' + rec['so'])
    build_record = build(out)
    control_lib, control_record = T.open_control()
    v1_lib, _ = T.open_candidate(v1['so'])
    p2_lib, _ = P2.open_pipe2(p2['so'])
    lib, resources = open_pipe3(build_record['so'])
    result = {'scope': __doc__, 'gate_relative_l2': T.GATE, 'primary_variant': PRIMARY, 'variants': VARIANTS,
              'candidate_build': build_record, 'candidate_resources': resources,
              'v1_build': {k_: v1[k_] for k_ in ('so', 'so_sha256', 'source_sha256')},
              'pipe2_build': {k_: p2[k_] for k_ in ('so', 'so_sha256', 'source_sha256')},
              'control': control_record, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'cases': {}}
    print('BUILD', json.dumps({'ptxas': build_record['ptxas'], 'resources': resources, 'sass': build_record['sass_counts']}), flush=True)
    with torch.inference_mode():
        for label in T.ORDER:
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
            args = T.scale_args(data['meta'])
            scales_f = prepare_scales(torch, lib, scales, k, args)
            scales_f2 = P2.prepare_scales(torch, p2_lib, scales, k, args)
            if not torch.equal(scales_f, scales_f2):
                raise RuntimeError('Prepared scales differ between pipe v2 and v3')
            S = base.unpack_hi(data['scales'].reshape(n, k // 128), data['meta']).cuda()
            W = (codes.double().reshape(n, -1, 128) / 121.0 * S[..., None]).reshape(n, k)
            x = acts['shape-' + label].cuda().contiguous()
            if list(x.shape) != [128, k] or x.dtype != torch.float16:
                raise RuntimeError('Invalid activation: ' + label)
            m = x.shape[0]
            exact = torch.nn.functional.linear(x.double(), W)
            norm = exact.norm()
            rsum = {sub: codes.to(torch.int32).reshape(n, k // sub, sub).sum(-1).to(torch.int16).contiguous() for sub in (32, 16)}
            rsum_f = {sub: v.t().float().contiguous() for sub, v in rsum.items()}
            names = ['swizzle', 'imma_v1_sub32', 'imma_v1_sub16', 'pipe2_32_s5'] + list(VARIANTS) + ['fp8_full', 'fp8_mm']
            outputs = {name: torch.empty((m, n), device='cuda', dtype=torch.float16) for name in names}
            aq = {name: torch.empty((m, k), device='cuda', dtype=torch.int8) for name in names}
            ac = {name: torch.empty((m, k // (16 if '16' in name else 32), 2), device='cuda', dtype=torch.float32) for name in names}
            arms = {'swizzle': T.control_launcher(torch, control_lib, codes, scales, x, outputs['swizzle'], args),
                    'imma_v1_sub32': T.candidate_launcher(torch, v1_lib, 0, codes, scales, x, rsum[32], aq['imma_v1_sub32'], ac['imma_v1_sub32'], outputs['imma_v1_sub32'], args),
                    'imma_v1_sub16': T.candidate_launcher(torch, v1_lib, 1, codes, scales, x, rsum[16], aq['imma_v1_sub16'], ac['imma_v1_sub16'], outputs['imma_v1_sub16'], args),
                    'pipe2_32_s5': P2.pipe2_launcher(torch, p2_lib, 3205, codes, scales_f, x, rsum_f[32], aq['pipe2_32_s5'], ac['pipe2_32_s5'], outputs['pipe2_32_s5'])}
            for name, variant in VARIANTS.items():
                sub = variant // 1000
                arms[name] = pipe3_launcher(torch, lib, variant, codes, scales_f, x, rsum_f[sub], aq[name], ac[name], outputs[name])
            full, mm_only, fp8_info = T.fp8_arms(torch, W, x, outputs['fp8_full'])
            arms['fp8_full'] = full
            arms['fp8_mm'] = lambda: mm_only()
            accuracy, first_outputs = {}, {}
            for name, fn in arms.items():
                first = fn().clone()
                torch.cuda.synchronize()
                second = fn().clone()
                torch.cuda.synchronize()
                first_outputs[name] = first
                y = first.double()
                rel = float((y - exact).norm() / norm)
                worst = float(((y - exact).norm(dim=1) / exact.norm(dim=1).clamp_min(1e-30)).max())
                accuracy[name] = {'relative_l2': rel, 'worst_row_relative_l2': worst, 'passes_gate': rel <= T.GATE,
                                  'deterministic': bool(torch.equal(first.view(torch.uint8), second.view(torch.uint8))),
                                  'finite': bool(torch.isfinite(first).all())}
            for name in list(VARIANTS) + ['pipe2_32_s5']:
                ref = 'imma_v1_sub16' if '_16_' in name else 'imma_v1_sub32'
                diff = (first_outputs[name].double() - first_outputs[ref].double())
                accuracy[name]['byte_equal_to_' + ref] = bool(torch.equal(first_outputs[name].view(torch.uint8), first_outputs[ref].view(torch.uint8)))
                accuracy[name]['relative_l2_vs_' + ref] = float(diff.norm() / first_outputs[ref].double().norm())
            del first_outputs
            torch.cuda.synchronize()
            timed = T.timing(torch, arms, nvml)
            floor_us = {'codes_bytes_over_300GBps_us': n * k / T.BANDWIDTH * 1e6}
            case = {'model_occurrences': occurrences, 'primary': PRIMARY[label], 'accuracy': accuracy, 'fp8': fp8_info,
                    'timing': timed, 'floor': floor_us}
            result['cases'][label] = case
            med = timed['medians_ms']
            print('CASE', label, json.dumps({
                'rel_l2': {k_: round(v['relative_l2'], 6) for k_, v in accuracy.items()},
                'det': all(v['deterministic'] for v in accuracy.values()),
                'vs_v1': {k_: round([vv for kk, vv in v.items() if kk.startswith('relative_l2_vs')][0], 7) for k_, v in accuracy.items() if k_ in VARIANTS or k_ == 'pipe2_32_s5'},
                'ms': {k_: round(v, 4) for k_, v in med.items()},
                'over_swizzle': {k_: round(v['median'], 4) for k_, v in timed['ratios'].items() if k_.endswith('/swizzle')},
                'over_fp8_full': {k_: round(v['median'], 4) for k_, v in timed['ratios'].items() if k_.endswith('/fp8_full')},
                'clock': {k_: round(v) for k_, v in timed['mean_clock_mhz'].items()}, 'floor_us': {k_: round(v, 1) for k_, v in floor_us.items()}}), flush=True)
            del W, exact, data, outputs, aq, ac, rsum, rsum_f, arms, full, mm_only, scales_f, scales_f2
            torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    result['script_sha256'] = base.sha(__file__)
    result['helper_script_sha256'] = {'test_k5_imma_matrix_l4.py': base.sha(T.__file__), 'test_k5_imma_pipe2_matrix_l4.py': base.sha(P2.__file__)}
    result['base_script_sha256'] = base.sha(base.__file__)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    verdict = {label: {'primary': c['primary'], 'primary_passes_gate': c['accuracy'][c['primary']]['passes_gate'],
                       'primary_over_swizzle': c['timing']['ratios'][c['primary'] + '/swizzle']['median'],
                       'primary_over_fp8_full': c['timing']['ratios'][c['primary'] + '/fp8_full']['median'],
                       'best_variant': min(VARIANTS, key=lambda v: c['timing']['medians_ms'][v]),
                       'best_passes_gate': c['accuracy'][min(VARIANTS, key=lambda v: c['timing']['medians_ms'][v])]['passes_gate'],
                       'best_over_fp8_full': min(c['timing']['ratios'][v + '/fp8_full']['median'] for v in VARIANTS)}
               for label, c in result['cases'].items()}
    print('IMMA_PIPE3_MATRIX_DONE', json.dumps(verdict), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    main(parser.parse_args().out)
