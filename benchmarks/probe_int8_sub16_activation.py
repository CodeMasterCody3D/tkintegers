"""Third activation probe: affine INT8 scale per 16-element sub-group (8 exact s32 partials per 128 group).

Same inputs, reference and method as probe_int8_affine_activation.py and
probe_int8_subgroup_activation.py (both unchanged, imported). Adds sub-16 and
sub-8 so the error-versus-sub-group-size trend is measured instead of
extrapolated. mma.m16n8k16.s8 exists on sm89, so 16-wide scales are a kernel
option; sub-8 is measured only as a trend point. Error only; no speed claim.
"""
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import probe_int8_affine_activation as base
import probe_int8_subgroup_activation as sub

GATE = base.GATE

VARIANTS = {
    'affine_int8_sub8 (trend point only)': sub.affine(8),
    'affine_int8_sub16 (1 dot, 8 partials/group, m16n8k16)': sub.affine(16),
    'affine_int8_sub32 (repeat of second probe)': sub.affine(32),
    'sym_int8_sub16 (1 dot, 8 partials/group)': sub.symmetric(16),
}


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if base.sha(base.ACTIVATIONS) != base.ACTIVATIONS_SHA:
        raise RuntimeError('Activation fixture hash differs')
    acts = load_file(base.ACTIVATIONS)
    result = {'scope': 'Activation sub-group scale error only (16 and 8 wide); same inputs/reference as the first probe.',
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
    result['subgroup_script_sha256'] = base.sha(sub.__file__)
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PROBE_DONE', json.dumps(result['all_four_pass']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
