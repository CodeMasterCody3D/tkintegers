"""Activation-quantization error probe on the real Qwen3.5-9B K5 matrices (remote L4 only).

Question: can a ONE-dot integer path (K5 code q used directly as the INT8 weight)
meet the existing 0.005 relative-L2 matrix gate if the activation is quantized
per (row, 128-group)? Prior evidence (Qwen2.5-0.5B, T4): symmetric amax/127 INT8
failed (0/8, max 0.00655); a two-dot residual passed but was slower than FP16.

Every variant is written as y = linear(xhat, W), W = q/121 * S (FP64), versus the
exact y = linear(x, W). Because the integer path computes exactly
sum_g S_g * (a*dot(A,q) + b*R) with exact s32 group dots, this measures the
activation-quantization error exactly; only the per-group FP32 flush rounding
(order 1e-7 relative) is not modeled. Weights, codes and scales are unchanged.

No kernel, timing, model or default change. Inputs are hash-checked.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ACTIVATIONS = '/content/tk-speed9-1029/qwen35-9b-prefill1-profile-1037pm-1030-2026/prefill-activations.safetensors'
ACTIVATIONS_SHA = '559ec638235bb460fc8b2bcd11b9cfd6ff0ab318907af408bdad927e932372c6'
CASES = {
    '8192x4096': ('/content/tk-speed9-1029/artifacts/0008-tk.safetensors',
                  '6ed4ef7e51bc3fafb45cd438a1fd83e90c27398042d5555779b33106f9e0bdc6', 32),
    '4096x4096': ('/content/tk-speed9-1029/artifacts/0009-tk.safetensors',
                  'a22fddfed8c42c9b74e6c3ebec78cb6209e78847b654877bd1f20db54193b4c0', 56),
    '12288x4096': ('/content/tk-speed9-1029/artifacts/0013-tk.safetensors',
                   'fef3eb134193c07aaff78f3edc8a07d0e0d5ffd9cbd37ebed25cd9ea5ebc8b4f', 64),
    '4096x12288': ('/content/tk-speed9-1029/artifacts/0012-tk.safetensors',
                   '7b56c1e45f83148ddbd9be539622135d7d6b9268fd4be48dfba97a8e6a6585ca', 32),
}
GATE = 0.005
G = 128


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack_hi(codes, meta):
    mantissa, exponent, k, log = (int(v) for v in meta.tolist())
    levels = (3 ** k - 1) // 2
    values = codes.to(torch.float64) / levels * (float(mantissa) * 3.0 ** exponent)
    return torch.pow(values.new_tensor(2.0), values) if log else values


def grouped(x):
    return x.reshape(x.shape[0], -1, G)


def sym_int8(x):
    """Prior method: per (row, group) s = amax/127 (FP32), A = RNE(x/s) in [-127, 127]."""
    xg = grouped(x.float())
    s = xg.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    a = torch.clamp(torch.round(xg / s), -127, 127)
    return (a.double() * s.double()).reshape(x.shape)


def sym_int8_row(x):
    """Per-row scale over all of K (one flush multiply fewer, less accurate)."""
    xf = x.float()
    s = xf.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    a = torch.clamp(torch.round(xf / s), -127, 127)
    return a.double() * s.double()


def affine_int8(x):
    """Proposed: x ~ a*A + b, a = (max-min)/255, b = min + 128a, A = RNE((x-b)/a) in [-128, 127].
    Kernel form: sum_g S_g*(a*dot(A,q) + b*R_g) with R_g = sum_k q (precomputed, exact)."""
    xg = grouped(x.float())
    lo, hi = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
    a = (hi - lo) / 255.0
    flat = a == 0
    a_safe = torch.where(flat, torch.ones_like(a), a)
    b = torch.where(flat, lo, lo + 128.0 * a)
    A = torch.where(flat, torch.zeros_like(xg), torch.clamp(torch.round((xg - b) / a_safe), -128, 127))
    return (A.double() * a.double() + b.double()).reshape(x.shape)


def e4m3(x):
    """FP8 activation analog: per (row, group) s = amax/448, RNE to E4M3."""
    xg = grouped(x.float())
    s = xg.abs().amax(-1, keepdim=True) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    q = (xg / s).to(torch.float8_e4m3fn).float()
    return (q.double() * s.double()).reshape(x.shape)


def with_residual(fn):
    def apply(x):
        first = fn(x)
        return first + fn((x.double() - first).float())
    return apply


VARIANTS = {
    'sym_int8_group (prior, 1 dot)': sym_int8,
    'affine_int8_group (proposed, 1 dot + rank-1)': affine_int8,
    'sym_int8_row (1 dot)': sym_int8_row,
    'e4m3_group (1 FP8 dot)': e4m3,
    'sym_int8_group + residual (prior, 2 dots)': with_residual(sym_int8),
    'e4m3_group + residual (2 FP8 dots)': with_residual(e4m3),
}


def stats(x):
    xg = grouped(x.double())
    lo, hi = xg.amin(-1), xg.amax(-1)
    rms = xg.pow(2).mean(-1).sqrt().clamp_min(1e-30)
    peak = xg.abs().amax(-1) / rms
    skew = ((hi + lo) / (hi - lo).clamp_min(1e-30)).abs()
    q = torch.tensor([0.5, 0.99], dtype=torch.float64, device=x.device)
    return {'amax_over_rms_median_p99_max': [*torch.quantile(peak.flatten(), q).tolist(), float(peak.max())],
            'range_asymmetry_|hi+lo|/(hi-lo)_median_p99': torch.quantile(skew.flatten(), q).tolist()}


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if sha(ACTIVATIONS) != ACTIVATIONS_SHA:
        raise RuntimeError('Activation fixture hash differs')
    acts = load_file(ACTIVATIONS)
    result = {'scope': ('Activation-quantization error only, on saved real 128-row prefill activations '
                        'and original K5 matrices; FP64 exact reference over ALL output columns. '
                        'No kernel, speed, model or quality claim.'),
              'gate_relative_l2': GATE, 'cases': {}}
    with torch.inference_mode():
        for label, (path, want_sha, occurrences) in CASES.items():
            if sha(path) != want_sha:
                raise RuntimeError('Matrix hash differs: ' + label)
            data = load_file(path)
            n, k = (int(v) for v in label.split('x'))
            raw = data['codes'].reshape(n, k)
            if raw.dtype != torch.uint8 or int(raw.max()) > 242:
                raise RuntimeError('Invalid K5 codes: ' + label)
            q = (raw.to(torch.int16) - 121).cuda().double()
            S = unpack_hi(data['scales'].reshape(n, k // G), data['meta']).cuda()
            W = (q.reshape(n, -1, G) / 121.0 * S[..., None]).reshape(n, k)
            del q
            x = acts['shape-' + label].cuda()
            if list(x.shape) != [128, k] or x.dtype != torch.float16:
                raise RuntimeError('Invalid activation: ' + label)
            exact = torch.nn.functional.linear(x.double(), W)
            norm = exact.norm()

            def rel(y):
                return float((y - exact).norm() / norm)

            def row_max(y):
                return float(((y - exact).norm(dim=1) / exact.norm(dim=1).clamp_min(1e-30)).max())

            # Today's exact-code FP16 path: FP16 x times integer q, FP32 group sums, FP32 scale flush.
            qf = (data['codes'].reshape(n, k).to(torch.int16) - 121).cuda().float()
            part = torch.einsum('mgk,ngk->gmn', grouped(x.float()), qf.reshape(n, -1, G))
            fp16_path = (part * (S.float().T[:, None, :] / 121.0)).sum(0).double()
            del part, qf
            case = {'model_occurrences': occurrences, 'activation_stats': stats(x),
                    'fp16_path_relative_l2': rel(fp16_path), 'variants': {}}
            for name, fn in VARIANTS.items():
                y = torch.nn.functional.linear(fn(x).double(), W)
                r = rel(y)
                case['variants'][name] = {'relative_l2': r, 'worst_row_relative_l2': row_max(y),
                                          'passes_gate': r <= GATE}
            result['cases'][label] = case
            print(label, json.dumps({'fp16': case['fp16_path_relative_l2'],
                                     **{n_: v['relative_l2'] for n_, v in case['variants'].items()}}), flush=True)
            del W, exact, data
            torch.cuda.empty_cache()
    result['all_four_pass'] = {name: all(c['variants'][name]['passes_gate'] for c in result['cases'].values())
                               for name in VARIANTS}
    result['seconds'] = time.time() - started
    result['gpu'] = torch.cuda.get_device_name(0)
    result['torch'] = torch.__version__
    result['script_sha256'] = sha(__file__)
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PROBE_DONE', json.dumps(result['all_four_pass']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
