"""Qwen3.5 AWQ folds extracted from the successful r2 controller.

Only dependencies are injected: packed_linear(payload) returns the actual K5
runtime module; check_stop() implements the caller's existing cancellation check.
The numerical function bodies are checked against the frozen controller by
verify.py. The caller uses torch.no_grad(), as in the measured controller.
"""
import torch
from tk_codec import pack_k5_hiscale


def awq_loss(weights, names, x, scaling, packed_linear):
    """AWQ linear-output search, measured using the actual shipping K5 kernel.

    Shared-norm consumers share one scale and their squared output errors sum.
    This is the upstream linear-output objective, not a nonlinear block search
    or the upstream clipping stage. Every one of the 16 examples contributes.
    """
    loss = torch.zeros((), device='cuda', dtype=torch.float64)
    for name in names:
        w = weights[name]
        module = packed_linear(pack_k5_hiscale(w * scaling[None, :]))
        for start in range(0, x.shape[0], 256):
            xx = x[start:start + 256].cuda()
            reference = torch.nn.functional.linear(xx, w.half())
            actual = module((xx.float() / scaling).half())
            loss += (actual.float() - reference.float()).square().sum(dtype=torch.float64)
        del module
    result = float(loss)
    if not 0 <= result < float('inf'):
        raise RuntimeError('AWQ search produced nonfinite energy')
    return result


def fold_awq(block, weights, original_inputs, *, packed_linear, check_stop):
    is_delta = block.block_type == 'linear_attention'
    first = ['linear_attn.' + n for n in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a']] if is_delta else ['self_attn.' + n for n in ['q_proj', 'k_proj', 'v_proj']]
    groups = [('input_layernorm', first),
              ('post_attention_layernorm', ['mlp.gate_proj', 'mlp.up_proj']),
              ('mlp.up_proj', ['mlp.down_proj'])]
    input_scalings = {n: torch.ones(w.shape[1], device='cuda') for n, w in weights.items()}
    report = []
    for previous, names in groups:
        check_stop()
        x = original_inputs[names[0]]
        act = x.float().abs().mean(0).cuda()
        prev = block.get_submodule(previous)
        norm = previous.endswith('layernorm')
        old_norm = prev.weight.detach().clone() if norm else None
        best = None
        for i in range(20):
            check_stop()
            ratio = i / 20
            scaling = act.pow(ratio).clamp_min(0.0001)
            scaling /= (scaling.max() * scaling.min()).sqrt()
            new_norm = None
            if norm:
                # Respect the actual zero-centered gamma and its stored FP16 value.
                new_norm = ((1 + old_norm.float()) / scaling - 1).half()
                if i == 0:
                    new_norm = old_norm.clone()
                gamma = 1 + old_norm.float()
                gamma_new = 1 + new_norm.float()
                if bool(((gamma != 0) & (gamma_new == 0)).any()):
                    continue
                scaling = torch.where(gamma == 0, 1, gamma / gamma_new)
                if not torch.isfinite(scaling).all() or bool((scaling <= 0).any()):
                    continue
            value = awq_loss(weights, names, x, scaling, packed_linear)
            if best is None or value < best['loss']:
                best = {'loss': value, 'ratio': ratio, 'scale': scaling.clone(),
                        'norm': new_norm.clone() if norm else None}
        if best is None:
            raise RuntimeError('No finite AWQ candidate, including identity')
        scaling = best.pop('scale')
        new_norm = best.pop('norm')
        if norm:
            prev.weight.copy_(new_norm)
        else:
            # SiLU(gate) * (up / S), followed by down*S: exact in real arithmetic.
            weights[previous] /= scaling[:, None]
            prev.weight.copy_(weights[previous].half())
        for name in names:
            weights[name] *= scaling[None, :]
            block.get_submodule(name).weight.copy_(weights[name].half())
            input_scalings[name] *= scaling
        report.append({'previous': previous, 'consumers': names, **best,
                       'minimum_scaling': float(scaling.min()), 'maximum_scaling': float(scaling.max()),
                       'zero_centered_norm': norm})
        print('__ANCHOR_STEP__ AWQ', previous, 'ratio', best['ratio'], flush=True)
    return input_scalings, report
