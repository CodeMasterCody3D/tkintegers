"""Matrix trial (remote L4 only): pipelined INT8 IMMA kernel versus the exact swizzle binary, the first INT8 kernel and cuBLAS FP8.

Candidate: benchmarks/k5_group_imma_pipe_cuda.cu (new). Same arithmetic as
k5_group_imma_cuda.cu; only the staging changes (cp.async, K staged 64 at a
time, STAGES buffers). Variants swept: SUB 32 with 2/3/4/5 stages, SUB 16 with
3/4 stages. Predeclared primary: SUB 32 x 4 stages (38.4 KB shared, 2 CTAs/SM).

Controls: the preserved swizzle binary f5d15c22... (read-only) and the first
INT8 binary from int8-imma-matrix-v1 (hash-checked). Competitor: torch._scaled_mm
E4M3 per-tensor (TorchAO scheme), with and without its activation quantization.

Per shape: FP64 exact reference, relative L2 (gate <= 0.005), worst row,
determinism, and byte equality of each pipelined output with the first INT8
kernel of the same SUB (same arithmetic in the same order). Then graph timing:
20 calls/graph, 5 replays/sample, 9 rounds, forward/reverse order, clocks and
power sampled around every sample. Nothing existing is modified.
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

SOURCE = HERE / 'k5_group_imma_pipe_cuda.cu'
V1_RESULT = Path('/content/tk-speed9-1029/int8-imma-matrix-v1/result.json')
VARIANTS = {'pipe32_s2': 3202, 'pipe32_s3': 3203, 'pipe32_s4': 3204, 'pipe32_s5': 3205, 'pipe16_s3': 1603, 'pipe16_s4': 1604}
PRIMARY = {'12288x4096': 'pipe32_s4', '8192x4096': 'pipe32_s4', '4096x4096': 'pipe32_s4', '4096x12288': 'pipe16_s4'}
SASS_OPS = ('IMMA.16832', 'IMMA.16816', 'HMMA', 'LDSM', 'LDGSTS', 'LDG.E.128', 'STS.128', 'BAR.SYNC', 'DEPBAR', 'FFMA', 'MUFU.EX2', 'STL', 'LDL')


def build(out):
    digest = base.sha(SOURCE)
    directory = Path('/content/tk-speed9-1029/build') / ('imma-pipe-sm89-' + digest[:16])
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


def open_pipe(so):
    lib = ctypes.CDLL(so)
    lib.initialize_k5_imma_pipe.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.initialize_k5_imma_pipe.restype = ctypes.c_int
    lib.launch_k5_imma_pipe.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 +
                                        [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.launch_k5_imma_pipe.restype = ctypes.c_int
    names = ['tile_m', 'tile_n', 'threads', 'shared_bytes', 'registers', 'local_bytes', 'active_blocks_per_sm',
             'quantize_registers', 'quantize_local_bytes']
    resources = {}
    for name, variant in VARIANTS.items():
        info = (ctypes.c_int * 9)()
        rc = lib.initialize_k5_imma_pipe(variant, info)
        if rc:
            raise RuntimeError(f'Pipe preflight failed for {name}: CUDA {rc}')
        resources[name] = dict(zip(names, info))
    return lib, resources


def pipe_launcher(torch, lib, variant, codes, scales_t, x, rsum_t, aq, ac, out, args):
    coded, anchor, levels, log = args
    m, k = x.shape
    n = codes.shape[0]

    def run():
        rc = lib.launch_k5_imma_pipe(variant, codes.data_ptr(), scales_t.data_ptr(), x.data_ptr(), rsum_t.data_ptr(),
                                     aq.data_ptr(), ac.data_ptr(), out.data_ptr(), m, n, k, codes.stride(0), x.stride(0),
                                     anchor, levels, coded, log, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError('Pipe launch failed: ' + str(rc))
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
    if base.sha(v1['so']) != v1['so_sha256']:
        raise RuntimeError('First INT8 binary differs from its record')
    build_record = build(out)
    control_lib, control_record = T.open_control()
    v1_lib, v1_resources = T.open_candidate(v1['so'])
    pipe_lib, pipe_resources = open_pipe(build_record['so'])
    result = {'scope': __doc__, 'gate_relative_l2': T.GATE, 'primary_variant': PRIMARY, 'variants': VARIANTS,
              'candidate_build': build_record, 'candidate_resources': pipe_resources,
              'v1_build': {'so': v1['so'], 'so_sha256': v1['so_sha256'], 'source_sha256': v1['source_sha256']},
              'control': control_record, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'cases': {}}
    print('BUILD', json.dumps({'ptxas': build_record['ptxas'], 'resources': pipe_resources, 'sass': build_record['sass_counts']}), flush=True)
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
            scales_t = scales.t().contiguous()                      # [k/128, n], same int16 values
            args = T.scale_args(data['meta'])
            S = base.unpack_hi(data['scales'].reshape(n, k // 128), data['meta']).cuda()
            W = (codes.double().reshape(n, -1, 128) / 121.0 * S[..., None]).reshape(n, k)
            x = acts['shape-' + label].cuda().contiguous()
            if list(x.shape) != [128, k] or x.dtype != torch.float16:
                raise RuntimeError('Invalid activation: ' + label)
            m = x.shape[0]
            exact = torch.nn.functional.linear(x.double(), W)
            norm = exact.norm()
            rsum = {sub: codes.to(torch.int32).reshape(n, k // sub, sub).sum(-1).to(torch.int16).contiguous() for sub in (32, 16)}
            rsum_t = {sub: v.t().contiguous() for sub, v in rsum.items()}   # [k/sub, n]
            names = ['swizzle', 'imma_v1_sub32', 'imma_v1_sub16'] + list(VARIANTS) + ['fp8_full', 'fp8_mm']
            outputs = {name: torch.empty((m, n), device='cuda', dtype=torch.float16) for name in names}
            aq = {name: torch.empty((m, k), device='cuda', dtype=torch.int8) for name in names}
            ac = {name: torch.empty((m, k // (16 if '16' in name else 32), 2), device='cuda', dtype=torch.float32) for name in names}
            arms = {'swizzle': T.control_launcher(torch, control_lib, codes, scales, x, outputs['swizzle'], args),
                    'imma_v1_sub32': T.candidate_launcher(torch, v1_lib, 0, codes, scales, x, rsum[32], aq['imma_v1_sub32'], ac['imma_v1_sub32'], outputs['imma_v1_sub32'], args),
                    'imma_v1_sub16': T.candidate_launcher(torch, v1_lib, 1, codes, scales, x, rsum[16], aq['imma_v1_sub16'], ac['imma_v1_sub16'], outputs['imma_v1_sub16'], args)}
            for name, variant in VARIANTS.items():
                sub = variant // 100
                arms[name] = pipe_launcher(torch, pipe_lib, variant, codes, scales_t, x, rsum_t[sub], aq[name], ac[name], outputs[name], args)
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
            for name in VARIANTS:
                ref = 'imma_v1_sub16' if '16' in name else 'imma_v1_sub32'
                accuracy[name]['byte_equal_to_' + ref] = bool(torch.equal(first_outputs[name].view(torch.uint8),
                                                                           first_outputs[ref].view(torch.uint8)))
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
                'byte_equal_v1': {k_: v.get('byte_equal_to_imma_v1_sub32', v.get('byte_equal_to_imma_v1_sub16')) for k_, v in accuracy.items() if k_ in VARIANTS},
                'ms': {k_: round(v, 4) for k_, v in med.items()},
                'over_swizzle': {k_: round(v['median'], 4) for k_, v in timed['ratios'].items() if k_.endswith('/swizzle')},
                'over_fp8_full': {k_: round(v['median'], 4) for k_, v in timed['ratios'].items() if k_.endswith('/fp8_full')},
                'clock': timed['mean_clock_mhz'], 'floor_us': {k_: round(v, 1) for k_, v in floor_us.items()}}), flush=True)
            del W, exact, data, outputs, aq, ac, rsum, rsum_t, arms, full, mm_only
            torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    result['script_sha256'] = base.sha(__file__)
    result['helper_script_sha256'] = base.sha(T.__file__)
    result['base_script_sha256'] = base.sha(base.__file__)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    verdict = {label: {'primary': c['primary'], 'primary_passes_gate': c['accuracy'][c['primary']]['passes_gate'],
                       'primary_over_swizzle': c['timing']['ratios'][c['primary'] + '/swizzle']['median'],
                       'primary_over_fp8_full': c['timing']['ratios'][c['primary'] + '/fp8_full']['median'],
                       'best_variant': min(VARIANTS, key=lambda v: c['timing']['medians_ms'][v]),
                       'best_over_fp8_full': min(c['timing']['ratios'][v + '/fp8_full']['median'] for v in VARIANTS)}
               for label, c in result['cases'].items()}
    print('IMMA_PIPE_MATRIX_DONE', json.dumps(verdict), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    main(parser.parse_args().out)
