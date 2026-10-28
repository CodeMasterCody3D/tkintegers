"""K5 stored-grid adapter for Forge's QEP and gated DAPTQ equations.

Sources: ternary_gptq.py::_qep_iter, joint_scale_solve_, and
_flip_polish_incremental_. Alphabet is generalized to [-121,121]; every
scale proposal is encoded by the unchanged integer-k10-log codec before
acceptance. Joint scale systems use activation factors of H, not a diagonal
approximation. They are evaluated in bounded output-row tiles.
"""
import torch
from tk_codec import pack_hi, unpack_hi, pack_k5_hiscale

GROUP = 128
LEVELS = 121


def decode(codes, scales, meta):
    s = unpack_hi(scales, meta).to(codes.device)
    return (codes.float().reshape(*s.shape, GROUP) / LEVELS * s[..., None]).flatten(1)


def energy(w, q, x, tile=128):
    """Stable E = tr((W-Q) H (W-Q)^T), H = X^T X / token count."""
    value = torch.zeros((), device=w.device, dtype=torch.float64)
    for a in range(0, w.shape[0], tile):
        err = (w[a:a + tile] - q[a:a + tile]) @ x.T
        value += err.square().sum(dtype=torch.float64)
    result = float(value / x.shape[0])
    if not 0 <= result < float('inf'):
        raise RuntimeError('Nonfinite/negative stored-grid energy')
    return result


def statistics(xf, xq, damp=0.01):
    if xf.shape != xq.shape or xf.ndim != 2 or not xf.shape[0]:
        raise RuntimeError('QEP token alignment mismatch')
    xf, xq = xf.float(), xq.float()
    h = xq.T @ xq / xq.shape[0]
    cross = xf.T @ xq / xq.shape[0]
    if not torch.isfinite(h).all() or not torch.isfinite(cross).all():
        raise RuntimeError('Nonfinite incoming statistics; never sanitize silently')
    hd = h.clone()
    dead = hd.diagonal() == 0
    hd.diagonal()[dead] = 1
    hd.diagonal().add_(damp * hd.diagonal().mean())
    chol = torch.linalg.cholesky(hd)
    return h, cross, chol


def qep_target(w, h, cross, chol, alpha=0.5):
    # Same solve as W + alpha W(C-H)H_damped^-1, without storing a dense inverse.
    rhs = w @ (cross - h)
    correction = torch.cholesky_solve(rhs.T.contiguous(), chol).T
    target = w + alpha * correction
    if not torch.isfinite(target).all():
        raise RuntimeError('Nonfinite QEP target')
    return target, float((target - w).norm() / w.norm().clamp_min(0.000000000001))


def joint_scales(w, codes, x, tile=32):
    """Exact off-diagonal joint solve via B[r,t,g] = X[t,g] dot C[r,g].

    B^T B / T = C_g^T H C_h, B^T (X W^T) / T = C_g^T H W.
    This is Forge's joint system, factored to avoid one full H GEMM per group.
    """
    rows, cols = codes.shape
    ng = cols // GROUP
    xg = x.reshape(x.shape[0], ng, GROUP)
    result = torch.empty((rows, ng), device=w.device)
    fallbacks = 0
    for a in range(0, rows, tile):
        c = codes[a:a + tile].float().reshape(-1, ng, GROUP) / LEVELS
        bfactor = torch.einsum('tgi,rgi->rtg', xg, c)
        target = x @ w[a:a + tile].T
        system = bfactor.transpose(1, 2) @ bfactor / x.shape[0]
        rhs = torch.einsum('rtg,tr->rg', bfactor, target) / x.shape[0]
        system.diagonal(dim1=1, dim2=2).add_(0.00000001)
        sol, info = torch.linalg.solve_ex(system, rhs[..., None])
        sol = sol.squeeze(-1)
        residual = ((system @ sol[..., None]).squeeze(-1) - rhs).norm(dim=1) / rhs.norm(dim=1).clamp_min(0.000000000000000000000000000001)
        bad = (info != 0) | ~torch.isfinite(sol).all(1) | (residual > 0.001)
        if bool(bad.any()):
            fallbacks += int(bad.sum())
            sol[bad] = torch.linalg.solve(system[bad].double(), rhs[bad, :, None].double()).squeeze(-1).float()
            residual = ((system @ sol[..., None]).squeeze(-1) - rhs).norm(dim=1) / rhs.norm(dim=1).clamp_min(0.000000000000000000000000000001)
        if not torch.isfinite(sol).all() or float(residual.max()) > 0.001:
            raise RuntimeError('Joint-scale solve residual failed')
        result[a:a + tile] = sol.clamp_min(0.000000000001)
    return result, fallbacks


def polish(w, codes, scales, meta, h, steps=10):
    """One best legal code change per output row per step; no interacting flips.

    The scalar quadratic gives the best of all 243 levels directly. Forge's
    incremental gradient update is refreshed every eight steps, then the caller
    remeasures the actual stored candidate on all calibration tokens.
    """
    c = codes.to(torch.int16).clone()
    unit = (unpack_hi(scales, meta).to(w.device) / LEVELS).repeat_interleave(GROUP, 1)
    q = c.float() * unit
    grad = (q - w) @ h
    hd = h.diagonal().clamp_min(0)
    row = torch.arange(w.shape[0], device=w.device)
    changed = 0
    for step in range(steps):
        if step and step % 8 == 0:
            grad = (q - w) @ h
        denominator = unit * hd
        moves = torch.where(denominator > 0, -grad / denominator.clamp_min(0.000000000000000000000000000001), 0).round()
        new = (c.float() + moves).clamp(-LEVELS, LEVELS)
        delta = (new - c.float()) * unit
        gain = 2 * delta * grad + delta.square() * hd
        col = gain.argmin(1)
        use = gain[row, col] < 0
        if not bool(use.any()):
            break
        rr, cc = row[use], col[use]
        dd = delta[rr, cc]
        c[rr, cc] = new[rr, cc].to(torch.int16)
        q[rr, cc] = c[rr, cc].float() * unit[rr, cc]
        grad[rr] += dd[:, None] * h[cc]
        changed += int(use.sum())
    return c.to(torch.int8), changed


def refine(w, h, x, rounds=2, steps=10):
    # Existing K5 rounding is the initializer, not another independently scored method.
    codes, scales, meta = pack_k5_hiscale(w, k=5, scale_k=10, group_size=GROUP)
    q = decode(codes, scales, meta)
    start = best = energy(w, q, x)
    history = []
    for iteration in range(rounds):
        before = best
        proposed, fallbacks = joint_scales(w, codes, x)
        sc1, meta1 = pack_hi(proposed, 10, log=True)
        q1 = decode(codes, sc1, meta1)
        e1 = energy(w, q1, x)
        accepted_scale = e1 < best
        if accepted_scale:
            scales, meta, q, best = sc1, meta1, q1, e1
        c1, changes = polish(w, codes, scales, meta, h, steps=steps)
        q1 = decode(c1, scales, meta)
        e2 = energy(w, q1, x)
        accepted_codes = e2 < best
        if accepted_codes:
            codes, q, best = c1, q1, e2
        history.append({'round': iteration + 1, 'start': before,
                        'scale_trial': e1, 'scale_kept': accepted_scale,
                        'code_trial': e2, 'codes_kept': accepted_codes,
                        'proposed_coordinate_moves': changes,
                        'scale_solve_double_fallback_rows': fallbacks, 'end': best})
        if not best < before * (1 - 0.000001):
            break
    # The values judged here are decoded from the actual integer payload again.
    check = energy(w, decode(codes, scales, meta), x)
    if check > start * (1 + 0.000001) or abs(check - best) > max(0.00000001, abs(best) * 0.000001):
        raise RuntimeError('Stored-grid acceptance was not reproducible')
    return (codes.contiguous(), scales.contiguous(), meta), {
        'energy_start': start, 'energy_final': check, 'rounds': history,
        'objective': 'QEP target versus decoded stored K5 grid on all real calibration tokens',
        'initializer': 'Unchanged pack_k5_hiscale; no additional quantizer sweep'}


def algebra_check(w, x):
    """Small same-model mirror checks before the expensive first block."""
    w, x = w[:4, :256].float(), x[:, :256].float()
    h, c, chol = statistics(x, x)
    same, movement = qep_target(w, h, c, chol)
    if not torch.equal(same, w) or movement != 0:
        raise RuntimeError('QEP changes an identical-input control')
    s = torch.linspace(0.5, 2, 256, device=x.device)
    hs = (x / s).T @ (x / s) / x.shape[0]
    expected = h / s[:, None] / s[None, :]
    relative = float((hs - expected).norm() / expected.norm())
    if relative > 0.00001:
        raise RuntimeError('AWQ Hessian basis transformation failed')
    codes, sc, meta = pack_k5_hiscale(w, group_size=GROUP)
    cf = codes.float() / LEVELS
    bg = torch.einsum('tgi,rgi->rtg', x.reshape(-1, 2, GROUP), cf.reshape(4, 2, GROUP))
    system = bg.transpose(1, 2) @ bg / x.shape[0]
    direct = torch.empty_like(system)
    for g in range(2):
        for j in range(2):
            direct[:, g, j] = (cf[:, g*GROUP:(g+1)*GROUP] @ h[g*GROUP:(g+1)*GROUP, j*GROUP:(j+1)*GROUP] * cf[:, j*GROUP:(j+1)*GROUP]).sum(1)
    system_relative = float((direct - system).norm() / direct.norm())
    if system_relative > 0.00001:
        raise RuntimeError('Factored joint scale system differs from full-H equation')
    payload, result = refine(w, h, x, rounds=2, steps=10)
    return {'same_input_qep_exact': True, 'scaled_hessian_relative_l2': relative,
            'joint_system_relative_l2': system_relative, 'stored_grid_refinement': result}
