"""Follow-up to probe_int8_affine_activation.py (unchanged): activation scale per SUB-group.

One-dot INT8 with the K5 code as the weight. The activation scale (and offset)
is per (row, sub-group of SUB elements) inside each 128 weight group. The
integer dot splits into 128/SUB exact s32 partials:

  y = sum_g S[n,g] * sum_j (a[m,g,j]*dot(A,q)[j] + b[m,g,j]*R[n,g,j])

R[n,g,j] = sum of codes in sub-group j, exact in int16. Same method as the
first probe: y = linear(xhat, W) versus the exact FP64 y over all columns, on
the same hash-checked real inputs. Error only; no kernel, speed or model claim.
"""
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import probe_int8_affine_activation as base

GATE = base.GATE


def sub(x, size):
    return x.reshape(x.shape[0], -1, size)


def affine(size):
    def apply(x):
        xg = sub(x.float(), size)
        lo, hi = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
        a = (hi - lo) / 255.0
        flat = a == 0
        a_safe = torch.where(flat, torch.ones_like(a), a)
        b = torch.where(flat, lo, lo + 128.0 * a)
        A = torch.where(flat, torch.zeros_like(xg), torch.clamp(torch.round((xg - b) / a_safe), -128, 127))
        return (A.double() * a.double() + b.double()).reshape(x.shape)
    return apply


def symmetric(size):
    def apply(x):
        xg = sub(x.float(), size)
        s = xg.abs().amax(-1, keepdim=True) / 127.0
        s = torch.where(s > 0, s, torch.ones_like(s))
        return (torch.clamp(torch.round(xg / s), -127, 127).double() * s.double()).reshape(x.shape)
    return apply


VARIANTS = {
    'affine_int8_sub32 (1 dot, 4 partials/group)': affine(32),
    'sym_int8_sub32 (1 dot, 4 partials/group)': symmetric(32),
    'affine_int8_sub64 (1 dot, 2 partials/group)': affine(64),
    'affine_int8_sub128 (repeat of first probe)': affine(128),
}


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if base.sha(base.ACTIVATIONS) != base.ACTIVATIONS_SHA:
        raise RuntimeError('Activation fixture hash differs')
    acts = load_file(base.ACTIVATIONS)
    result = {'scope': 'Activation sub-group scale error only; same inputs/reference as the first probe.',
              'gate_relative_l2': GATE, 'cases': {}}
    with torch.inference_mode():
        for label, (path, want_sha, occurrences) in base.CASES.items():
            if base.sha(path) != want_sha:
                raise RuntimeError('Matrix hash differs: ' + label)
            data = load_file(path)
            n, k = (int(v) for v in label.split('x'))
            q = (data['codes'].reshape(n, k).to(torch.int16) - 121).cuda().double()
            S = base.unpack_hi(data['scales'].reshape(n, k // 128), data['meta']).cuda()
            W = (q.reshape(n, -1, 128) / 121.0 * S[..., None]).reshape(n, k)
            del q
            x = acts['shape-' + label].cuda()
            exact = torch.nn.functional.linear(x.double(), W)
            case = {'model_occurrences': occurrences, 'variants': {}}
            for name, fn in VARIANTS.items():
                y = torch.nn.functional.linear(fn(x), W)
                r = float((y - exact).norm() / exact.norm())
                worst = float(((y - exact).norm(dim=1) / exact.norm(dim=1).clamp_min(1e-30)).max())
                case['variants'][name] = {'relative_l2': r, 'worst_row_relative_l2': worst, 'passes_gate': r <= GATE}
            result['cases'][label] = case
            print(label, json.dumps({k_: round(v['relative_l2'], 6) for k_, v in case['variants'].items()}), flush=True)
            del W, exact, data
            torch.cuda.empty_cache()
    result['all_four_pass'] = {name: all(c['variants'][name]['passes_gate'] for c in result['cases'].values())
                               for name in VARIANTS}
    result['seconds'] = time.time() - started
    result['script_sha256'] = base.sha(__file__)
    result['base_script_sha256'] = base.sha(base.__file__)
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PROBE_DONE', json.dumps(result['all_four_pass']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
