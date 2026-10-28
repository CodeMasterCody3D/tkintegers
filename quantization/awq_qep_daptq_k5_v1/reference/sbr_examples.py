"""Complete-example walker + checkpointing for shadow_body_rot_08b (SBR_EXAMPLES=1).

Cody, 2026-09-20 (via ChatGPT), the requirements this file exists for:
  * "Keep 256 as the processing chunk size. My requirement is to process each complete saved prompt + thinking +
    answer with continuous context. ... Do not silently truncate them or reset their context at row boundaries."
    -> each example is walked in `chunk` pieces with a CARRIED DynamicCache for student AND teacher; the cache is
       reset only between examples; every token is processed (the last chunk is simply short) and that is asserted.
  * "Show the original token count, processed token count, and whether the final answer is included."
    -> ExamplesReport writes one JSONL row per walked example; the console prints the running totals.
  * "clarify whether bs 8 means eight live sequences or gradient accumulation."
    -> ONE live example; the trainer accumulates `accum` examples per optimizer step (loss scaled 1/(n_tok*accum)).
  * "Use FLA and causal-conv1d where they support this reconstruction correctly ... Don't remove or bypass any
    model operations."  -> the HF forward is used as-is; backend_report/require_backend assert what is bound.
  * "stop on failure or OOM rather than shortening examples."  -> the trainer catches OOM, writes an emergency
    checkpoint and EXITS with the example id and counts; nothing is clipped.

Mechanism (verified in transformers 5.14.1, memory qwen35-carried-state-chunking): a populated DynamicCache carries
the conv tail, the DeltaNet recurrent state, the attention KV and the RoPE positions into a multi-token chunk. Under
grad the cache tensors are detached between chunks (truncated BPTT): activation memory is per chunk, the forward
context is the whole example, every chunk still updates every shared weight.
"""
from __future__ import annotations

import json
import os
import random
import time

import torch
from transformers import DynamicCache


# ----------------------------------------------------------------------------------------------- backends
# transformers <= 5.14 bound the kernels PER INSTANCE (`self.chunk_gated_delta_rule = fla_fn or torch_fn`).
# transformers >= 5.16 dispatches through MODULE-LEVEL wrappers: hub_kernels.use_kernel_func_from_hub_with_fallback
# resolves the local `fla` / `causal_conv1d` package at import time (priority hub kernel > local package > torch) and
# keeps it in the wrapper's closure (`implementation`, `is_new_implementation`) under functools.wraps. Both are
# handled here; the report says which is in force so a silent torch fallback can never pass as accelerated
# (Cody 2026-09-20: "this explains why the recon and gptq-rot been SOOO slow" -- neither package was ever installed).
_KFN = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")


def _q():
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    return Q


def dn_layers(model):
    return [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]


def resolve_impl(fn):
    """(implementation name, is_accelerated) for a >=5.16 kernel wrapper; walks the functools.wraps chain."""
    import inspect
    f, depth = fn, 0
    while f is not None and depth < 6:
        try:
            nl = inspect.getclosurevars(f).nonlocals
            if "implementation" in nl:
                impl = nl["implementation"]
                new = bool(nl.get("is_new_implementation", impl is not nl.get("torch_function")))
                return f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', '?')}", new
        except (TypeError, ValueError):
            pass
        f = getattr(f, "__wrapped__", None)
        depth += 1
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}", False


def backend_report(model, tag, log=print):
    """Which DeltaNet kernels are in force for `model`; asserts every layer agrees; never guesses."""
    ls = dn_layers(model)
    if not ls:
        log(f"[{tag}] no GatedDeltaNet layers found")
        return {}
    Q = _q()

    def name(f):
        return f"{getattr(f, '__module__', '?')}.{getattr(f, '__name__', type(f).__name__)}"

    if hasattr(Q, "chunk_gated_delta_rule"):                       # <= 5.14: per-instance bindings
        b = [(name(d.chunk_gated_delta_rule), name(d.recurrent_gated_delta_rule),
              "None(torch conv)" if d.causal_conv1d_fn is None else d.causal_conv1d_fn.__module__,
              type(d.norm).__name__) for d in ls]
        if len(set(b)) != 1:
            raise SystemExit(f"[{tag}] DeltaNet layers bind DIFFERENT kernels: {set(b)}")
        chunk_rule, rec_rule, conv, norm = b[0]
        accel = ("fla" in chunk_rule) and not conv.startswith("None")
        out = {"chunk": chunk_rule, "recurrent": rec_rule, "conv": conv, "norm": norm, "accel": accel,
               "style": "per-instance (<=5.14)"}
    else:                                                          # >= 5.16: module-level wrappers
        res = {k: resolve_impl(getattr(Q, k)) for k in _KFN}
        accel = all(new for _, new in res.values())
        out = {"chunk": res["torch_chunk_gated_delta_rule"][0], "recurrent": res["torch_recurrent_gated_delta_rule"][0],
               "conv": res["causal_conv1d_fn"][0], "conv_update": res["causal_conv1d_update"][0],
               "norm": type(ls[0].norm).__name__, "accel": accel, "style": "module wrappers (>=5.16)",
               "hub_kernels": os.environ.get("USE_HUB_KERNELS", "YES")}
    log(f"[{tag}] DeltaNet backend x{len(ls)} [{out['style']}]: chunk={out['chunk']} recurrent={out['recurrent']} "
        f"conv={out['conv']} norm={out['norm']} -> {'ACCELERATED (fla + causal_conv1d)' if accel else 'PARTIAL / TORCH FALLBACK'}")
    return out


class torch_backend:
    """Context manager: force the pure-torch DeltaNet path on the SAME model (>=5.16 module dispatch only)."""

    def __enter__(self):
        Q = _q()
        assert not hasattr(Q, "chunk_gated_delta_rule"), "torch_backend() is for transformers>=5.16 module dispatch"
        self.saved = {k: getattr(Q, k) for k in _KFN}
        for k, fn in self.saved.items():
            f = fn
            while getattr(f, "__wrapped__", None) is not None:
                f = f.__wrapped__
            setattr(Q, k, f)
        return self

    def __exit__(self, *exc):
        Q = _q()
        for k, v in self.saved.items():
            setattr(Q, k, v)
        return False


def require_backend(binds, allow_torch, log=print):
    if binds and not binds["accel"] and not allow_torch:
        raise SystemExit(
            "DeltaNet is NOT on the accelerated path (FLA + causal_conv1d). Cody's rule: the torch fallback may only "
            "be used after it is verified against the accelerated path on the same weights and inputs "
            "(chunk_carry_equiv.py). Install fla + causal-conv1d, or set SBR_ALLOW_TORCH_FALLBACK=1 after that check.")


# ----------------------------------------------------------------------------------------------- the walk
def detach_cache_(cache):
    """Truncated BPTT: cut the autograd graph at the chunk boundary, keep every state VALUE."""
    for L in cache.layers:
        for a in ("keys", "values"):
            t = getattr(L, a, None)
            if torch.is_tensor(t) and t.requires_grad:
                setattr(L, a, t.detach())
        for dn in ("conv_states", "recurrent_states"):
            d = getattr(L, dn, None)
            if isinstance(d, dict):
                for k, v in d.items():
                    if torch.is_tensor(v) and v.requires_grad:
                        d[k] = v.detach()


_CARRY_CHECK = {"v": None}


def deltanet_carry_check():
    """'ok ...' if the installed Qwen3.5 DeltaNet layer READS its cached state for a multi-token chunk, else a 'REFUSING: ...' text.

    10-03 (Cody's research question: "in the pinned Transformers version, Qwen's DeltaNet path reuses its recurrent state only for
    single-token continuation"): TRUE for transformers 5.2.0-5.6.2 -- `use_precomputed_states` also requires `seq_len == 1`, so a
    multi-token chunk runs the chunk kernel with initial_state=None and convolves the chunk alone: a ZERO DeltaNet state + conv tail at
    every chunk boundary even with a cache passed (measured, 4-layer Qwen3.5 model, torch path: chunk 2 9.3% off the full forward; the
    attention layer still carried). Fixed in 5.7.0; 5.16.1 / 5.17.0 / 5.18.0 measured carrying (chunk 2 within 3 parts in 10 million), and
    our carried walk (carry_stream.carried_chunks + this cache) too. assert_cache_live cannot see the bug -- the state is WRITTEN, just
    never READ -- so the check reads the layer's own source."""
    if _CARRY_CHECK["v"] is not None:
        return _CARRY_CHECK["v"]
    import inspect
    import transformers
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        src = inspect.getsource(M.Qwen3_5GatedDeltaNet.forward)
    except Exception as ex:                                   # noqa: BLE001
        _CARRY_CHECK["v"] = f"ok (no Qwen3.5 DeltaNet to check in transformers {transformers.__version__}: {type(ex).__name__})"
        return _CARRY_CHECK["v"]
    i = src.find("use_precomputed_states =")
    j = src.find("\n\n", i) if i >= 0 else -1
    expr = " ".join(src[i:j if j > i else i + 400].split()) if i >= 0 else ""
    if i < 0:
        res = (f"REFUSING: transformers {transformers.__version__}: no `use_precomputed_states` in Qwen3_5GatedDeltaNet.forward -- the "
               f"layer changed; re-run chunk_carry_equiv.py before trusting a carried walk")
    elif "seq_len" in expr:
        res = (f"REFUSING: transformers {transformers.__version__}'s Qwen3.5 DeltaNet reads its cached state only for 1-token steps "
               f"({expr[:170]}) -- every multi-token chunk would restart from a zero state. Install transformers>=5.7 (the repo pins 5.17.0).")
    else:
        res = f"ok (transformers {transformers.__version__}: the DeltaNet reads the cached state for multi-token chunks)"
    _CARRY_CHECK["v"] = res
    return res


def make_train_cache(config):
    """A DynamicCache whose linear-attention layers REBIND the recurrent state on update instead of copy_()ing.

    HF's `LinearAttentionLayer.update_recurrent_state` does `self.recurrent_states[i].copy_(new)` ("to preserve the
    static address for cudagraphs"). FLA's chunk kernel puts `initial_state` -- that very buffer -- in
    ctx.save_for_backward, so under grad the saved tensor is stale by backward time. Measured 2026-09-20 on Cody's
    A100: "modified by an inplace operation [1,16,128,128] ... version 36; expected 35" at
    fla/ops/gated_delta_rule/chunk.py:361. Rebinding changes storage policy only: every model op and every value is
    identical (the same `last_recurrent_state` tensor the model produced is what the cache holds). The conv tail is
    left on HF's path -- the conv consumes a fresh torch.cat, not the cached buffer, so it is safe for backward.
    """
    import types
    _chk = deltanet_carry_check()                             # 10-03: a carried walk on a 5.2-5.6 transformers would silently reset
    if _chk.startswith("REFUSING"):
        raise SystemExit(_chk)
    cache = DynamicCache(config=config)

    def _rebind(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return self.recurrent_states[state_idx]

    # NO REFERENCE CYCLE (2026-09-23, 27B shadow run OOM): `types.MethodType(_rebind, L)` stored ON L made L -> bound
    # method -> L, so a finished example's cache (DeltaNet states + the whole example's KV) was freed only by Python's
    # rare gen-2 GC -- caches piled up ~0.3 GB/step until CUDA OOM. The closure holds a WEAK reference instead; the
    # method body and every value are unchanged.
    import weakref

    def _bind(layer):
        ref = weakref.ref(layer)

        def update_recurrent_state(recurrent_states, state_idx=0, **kwargs):
            return _rebind(ref(), recurrent_states, state_idx=state_idx, **kwargs)
        return update_recurrent_state

    n = 0
    for L in cache.layers:
        if hasattr(L, "recurrent_states") and hasattr(L, "update_recurrent_state"):
            L.update_recurrent_state = _bind(L)
            n += 1
    cache._sbx_rebound_layers = n
    return cache


def chunk_bounds(n: int, chunk: int):
    """[(s0, s1)] covering tokens 0..n exactly once. HF dispatches a 1-token chunk with a carried cache to the
    RECURRENT (decode) kernel, and FLA's fused_recurrent has NO backward (NotImplementedError, measured 2026-09-20
    after 1,645 walks -- an example of length 1 mod 256 ends in a 1-token chunk). A 1-token tail is folded into the
    previous chunk (chunk+1 tokens; the chunk kernel takes any length). Nothing is dropped, nothing is padded."""
    b = list(range(0, n, chunk)) + [n]
    if len(b) >= 3 and b[-1] - b[-2] == 1:
        b.pop(-2)
    return list(zip(b[:-1], b[1:]))


def _backbone_hidden(model, x, cache):
    """10-11 forge kernels: the model's final hidden state (after its final norm) and the updated cache, WITHOUT the lm_head -- the
    same call the CausalLM wrapper makes before its head (rotation.resolve_inner finds the text model in any wrapper)."""
    from rotation import resolve_inner
    out = resolve_inner(model)(input_ids=x, past_key_values=cache, use_cache=True)
    return out.last_hidden_state, out.past_key_values


class Walker:
    def __init__(self, student, teacher, chunk: int, dev: str, log=print):
        self.student, self.teacher, self.chunk, self.dev, self.log = student, teacher, int(chunk), dev, log
        self.single_token_examples = 0            # n == 1: forwarded without grad (nothing to backprop through)
        # 10-11 forge kernels (docs/KERNEL_RAM_PLAN_1011.md); both OFF by default = the old path bit for bit:
        #   FORGE_FUSED_KL=1 -> kl_example / held_kl_lanes take each model's final hidden state and fk_head_kl.head_kl does head + KL in
        #     token tiles (no whole [tokens, V] logits; tiles sized from free memory, halved on an OOM -- kernel-level OOM protection);
        #   FORGE_TEACHER_HCACHE=1 -> held_kl_lanes stores the TEACHER's final hidden state per (example, chunk) on the host the first time
        #     (4 KB a token on the 0.8B, 20 KB on the 27B -- vs 1 MB for the log-prob cache) and later passes rerun only its head.
        self.fused_kl = os.environ.get("FORGE_FUSED_KL", "0") == "1"
        self.teacher_hcache = {} if os.environ.get("FORGE_TEACHER_HCACHE", "0") == "1" else None
        if self.fused_kl:
            from fk_head_kl import self_check            # the auto-kernel check: a fused path that does not match switches itself off
            self.fused_kl = self_check(student.get_output_embeddings().weight, teacher.get_output_embeddings().weight, log=log)
        if self.fused_kl or self.teacher_hcache is not None:
            log(f"FORGE KERNELS in the walker: fused head-KL {'ON' if self.fused_kl else 'off'} | teacher hidden-state cache "
                f"{'ON' if self.teacher_hcache is not None else 'off'} (docs/KERNEL_RAM_PLAN_1011.md)")

    @staticmethod
    def chunk_weights(mask, s0, s1, like):
        """Position s0+j predicts ids[s0+j+1]: its weight is mask[s0+j+1] (the row's last position predicts nothing -> 0).
        `mask` is indexed by TARGET token (1 = counts). None -> every position counts."""
        if mask is None:
            return None
        w = mask[s0 + 1:s1 + 1].to(like.device, like.dtype)
        if w.numel() < like.numel():
            w = torch.cat([w, torch.zeros(like.numel() - w.numel(), device=w.device, dtype=w.dtype)])
        return w

    def kl_example(self, ids: torch.Tensor, grad: bool = False, scale: float = 1.0, mask=None):
        """Walk ONE complete example ids[1, n] in `chunk` pieces with carried caches. Returns (kl_sum, counted, processed) --
        counted = the positions the KL summed over (n without a mask). If grad: backward per chunk with loss = kl_chunk_sum * scale,
        student cache detached between chunks. `mask` (09-29, probe #5): a [n] 0/1 tensor by target token -- chat-templated rows
        count only the teacher's OWN turns (assistant tokens): a post-trained teacher does not model user / tool text (it predicts
        <|im_end|> mid-text with ~100% confidence there), so the KL on those positions is not a target. The FORWARD still walks every
        token (the state carries through user / tool text), only the loss skips them."""
        if self.fused_kl:
            return self._kl_example_fused(ids, grad, scale, mask)
        n = int(ids.shape[1])
        cs = make_train_cache(self.student.config)
        ct = make_train_cache(self.teacher.config)
        kl_sum, processed, counted = 0.0, 0, 0.0
        for s0, s1 in chunk_bounds(n, self.chunk):
            x = ids[:, s0:s1].to(self.dev)
            with torch.no_grad():
                ot = self.teacher(x, past_key_values=ct, use_cache=True)
                ct = ot.past_key_values
                lp_t = torch.log_softmax(ot.logits.float(), -1)
                del ot
            use_grad = grad and (s1 - s0) > 1            # a 1-token chunk would hit the no-backward recurrent kernel
            if grad and not use_grad:
                self.single_token_examples += 1
            ctx = torch.enable_grad() if use_grad else torch.no_grad()
            with ctx:
                os_ = self.student(x, past_key_values=cs, use_cache=True)
                cs = os_.past_key_values
                lp_s = torch.log_softmax(os_.logits.float(), -1)
                kl_pos = (lp_t.exp() * (lp_t - lp_s)).sum(-1)[0]
                w = self.chunk_weights(mask, s0, s1, kl_pos)
                if w is None:
                    kl_c = kl_pos.sum()
                    counted += float(kl_pos.numel())
                else:
                    kl_c = (kl_pos * w).sum()
                    counted += float(w.sum())
                if use_grad:
                    (kl_c * scale).backward()
            kl_sum += float(kl_c.detach())
            processed += int(x.shape[1])
            if grad:
                detach_cache_(cs)
            del os_, lp_s, lp_t, kl_c, kl_pos
        assert processed == n, f"walker processed {processed} of {n} tokens -- never allowed"
        return kl_sum, int(counted), processed

    @torch.no_grad()
    @torch.no_grad()                                      # EVAL: never build a graph (76 GB OOM on 2026-09-25 with shadow latents)
    def held_kl(self, examples, edges=(256, 512, 1024, 2048, 4096, 8192)):
        """Per-token mean KL over every held token, plus per-position-bucket means (the window diagnostic)."""
        edges_t = torch.tensor(edges)
        tot, ntok = 0.0, 0
        bsum = [0.0] * (len(edges) + 1)
        bcnt = [0] * (len(edges) + 1)
        for e in examples:
            ids = e["ids"].unsqueeze(0)
            n = int(ids.shape[1])
            cs = make_train_cache(self.student.config)
            ct = make_train_cache(self.teacher.config)
            for s, s1 in chunk_bounds(n, self.chunk):
                x = ids[:, s:s1].to(self.dev)
                ot = self.teacher(x, past_key_values=ct, use_cache=True)
                ct = ot.past_key_values
                os_ = self.student(x, past_key_values=cs, use_cache=True)
                cs = os_.past_key_values
                lp_t = torch.log_softmax(ot.logits.float(), -1)
                lp_s = torch.log_softmax(os_.logits.float(), -1)
                kl = (lp_t.exp() * (lp_t - lp_s)).sum(-1)[0].cpu()          # [chunk_len]
                tot += float(kl.sum())
                ntok += kl.numel()
                b = torch.bucketize(torch.arange(s, s + kl.numel()), edges_t)
                for q in range(len(edges) + 1):
                    sel = kl[b == q]
                    if sel.numel():
                        bsum[q] += float(sel.sum())
                        bcnt[q] += int(sel.numel())
                del ot, os_, lp_t, lp_s
        labels = [f"0-{edges[0]-1}"] + [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 1)] + [f"{edges[-1]}+"]
        buckets = {lab: (bsum[q] / bcnt[q]) for q, lab in enumerate(labels) if bcnt[q]}
        return {"kl": tot / max(1, ntok), "tokens": ntok, "examples": len(examples), "buckets": buckets}


    @torch.no_grad()                                      # EVAL: never build a graph (76 GB OOM on 2026-09-25 with shadow latents)
    def held_kl_lanes(self, examples, lane_of, cache=None):
        """ONE pass -> {"kl": mean over every held token, "lanes": {lane: token-weighted mean KL}, "examples": [(id,
        lane, tokens, kl)]}. Cody 2026-09-25 (lane-gate-rollback): the keep-best decision needs the lanes at every
        check, not only the mean; computing them lane-by-lane would walk the held set eleven times.
        `cache` (dict, 09-28 Cody "do the two speedups"): the TEACHER is frozen, so its fp32 log-softmax per (example, chunk)
        is stored on the host the first time and reused -- every later held-out pass runs only the student. Bit-identical
        (the same fp32 tensor); ~38 GB host RAM for the 27B's 38,607-token held set. All-or-nothing per example, because the
        teacher's carried state must be walked from chunk 0."""
        if self.fused_kl or self.teacher_hcache is not None:
            return self._held_kl_lanes_fused(examples, lane_of)
        tot, ntok = 0.0, 0
        lsum, lcnt, rows = {}, {}, []
        asum, lasum = 0.0, {}                     # top-20 agreement (Cody 09-29 12:50 AM: "keep the top 20 like qwen is supposed to do")
        for e in examples:
            ids = e["ids"].unsqueeze(0)
            n = int(ids.shape[1])
            eid = e.get("i", e.get("id"))
            bounds = list(chunk_bounds(n, self.chunk))
            hit = cache is not None and all((eid, ci) in cache for ci in range(len(bounds)))
            cs = make_train_cache(self.student.config)
            ct = None if hit else make_train_cache(self.teacher.config)
            esum, ecnt, easum = 0.0, 0, 0.0
            for ci, (s, s1) in enumerate(bounds):
                x = ids[:, s:s1].to(self.dev)
                if hit:
                    lp_t = cache[(eid, ci)].to(self.dev, non_blocking=True)
                else:
                    ot = self.teacher(x, past_key_values=ct, use_cache=True)
                    ct = ot.past_key_values
                    lp_t = torch.log_softmax(ot.logits.float(), -1)
                    if cache is not None:
                        cache[(eid, ci)] = lp_t.to("cpu")
                    del ot
                os_ = self.student(x, past_key_values=cs, use_cache=True)
                cs = os_.past_key_values
                lp_s = torch.log_softmax(os_.logits.float(), -1)
                kl = (lp_t.exp() * (lp_t - lp_s)).sum(-1)[0]
                # the share of the teacher's top-20 tokens the student also ranks in ITS top-20, per position (Qwen's top-20 view
                # of the distribution; the mean KL alone can improve while the ranking under the argmax drifts)
                t20 = lp_t[0].topk(20, -1).indices
                s20 = lp_s[0].topk(20, -1).indices
                agree = (t20.unsqueeze(-1) == s20.unsqueeze(-2)).any(-1).float().mean(-1)
                w = self.chunk_weights(e.get("mask") if getattr(self, "assistant_only", True) else None, s, s1, kl)
                if w is None:
                    esum += float(kl.sum()); ecnt += int(kl.numel()); easum += float(agree.sum())
                else:
                    esum += float((kl * w).sum()); ecnt += int(w.sum()); easum += float((agree * w).sum())
                del os_, lp_t, lp_s, kl, t20, s20, agree
            ln = lane_of(e)
            lsum[ln] = lsum.get(ln, 0.0) + esum
            lcnt[ln] = lcnt.get(ln, 0) + ecnt
            lasum[ln] = lasum.get(ln, 0.0) + easum
            tot += esum
            ntok += ecnt
            asum += easum
            rows.append((e.get("i", e.get("id")), ln, ecnt, esum / max(1, ecnt)))
        return {"kl": tot / max(1, ntok), "tokens": ntok,
                "lanes": {ln: lsum[ln] / max(1, lcnt[ln]) for ln in lsum},
                "lane_tokens": dict(lcnt), "examples": rows,
                "top20": asum / max(1, ntok), "lane_top20": {ln: lasum[ln] / max(1, lcnt[ln]) for ln in lasum}}

    # ---- 10-11 forge kernels: the same two walks on hidden states + fk_head_kl (see __init__); the old methods above are unchanged ----
    def _kl_example_fused(self, ids, grad, scale, mask):
        """kl_example with the head + KL fused in token tiles: the same carried walk, chunk bounds, weights, scale and returns."""
        from fk_head_kl import head_kl
        n = int(ids.shape[1])
        cs = make_train_cache(self.student.config)
        ct = make_train_cache(self.teacher.config)
        Ws = self.student.get_output_embeddings().weight
        Wt = self.teacher.get_output_embeddings().weight
        kl_sum, processed, counted = 0.0, 0, 0.0
        for s0, s1 in chunk_bounds(n, self.chunk):
            x = ids[:, s0:s1].to(self.dev)
            with torch.no_grad():
                ht, ct = _backbone_hidden(self.teacher, x, ct)
            use_grad = grad and (s1 - s0) > 1            # a 1-token chunk would hit the no-backward recurrent kernel
            if grad and not use_grad:
                self.single_token_examples += 1
            ctx = torch.enable_grad() if use_grad else torch.no_grad()
            with ctx:
                hs, cs = _backbone_hidden(self.student, x, cs)
                like = torch.empty(int(x.shape[1]), device=hs.device, dtype=torch.float32)
                w = self.chunk_weights(mask, s0, s1, like)
                loss, kl_pos, _ = head_kl(hs[0], Ws, ht[0], Wt, tw=w)
                counted += float(kl_pos.numel()) if w is None else float(w.sum())
                if use_grad:
                    (loss * scale).backward()
            kl_sum += float(loss.detach())
            processed += int(x.shape[1])
            if grad:
                detach_cache_(cs)
            del hs, ht, loss, kl_pos
        return kl_sum, counted, processed

    def _held_kl_lanes_fused(self, examples, lane_of):
        """held_kl_lanes with the head + KL (and the top-20 agreement) fused in token tiles; with FORGE_TEACHER_HCACHE the teacher's
        final hidden state per (example, chunk) is kept on the host after the first pass (all-or-nothing per example, because the
        teacher's carried state must be walked from chunk 0) and later passes run only the student + the two heads."""
        from fk_head_kl import head_kl, reference_head_kl
        Ws = self.student.get_output_embeddings().weight
        Wt = self.teacher.get_output_embeddings().weight
        kl_fn = head_kl if self.fused_kl else reference_head_kl
        hc = self.teacher_hcache
        tot, ntok = 0.0, 0
        lsum, lcnt, rows = {}, {}, []
        asum, lasum = 0.0, {}
        with torch.no_grad():
            for e in examples:
                ids = e["ids"].unsqueeze(0)
                n = int(ids.shape[1])
                eid = e.get("i", e.get("id"))
                bounds = list(chunk_bounds(n, self.chunk))
                hit = hc is not None and all((eid, ci) in hc for ci in range(len(bounds)))
                cs = make_train_cache(self.student.config)
                ct = None if hit else make_train_cache(self.teacher.config)
                esum, ecnt, easum = 0.0, 0, 0.0
                for ci, (s, s1) in enumerate(bounds):
                    x = ids[:, s:s1].to(self.dev)
                    if hit:
                        ht = hc[(eid, ci)].to(self.dev, non_blocking=True)
                    else:
                        ht, ct = _backbone_hidden(self.teacher, x, ct)
                        if hc is not None:
                            hc[(eid, ci)] = ht.to("cpu")
                    hs, cs = _backbone_hidden(self.student, x, cs)
                    _l, kl, agree = kl_fn(hs[0], Ws, ht[0], Wt, None, topk=20)
                    w = self.chunk_weights(e.get("mask") if getattr(self, "assistant_only", True) else None, s, s1, kl)
                    if w is None:
                        esum += float(kl.sum()); ecnt += int(kl.numel()); easum += float(agree.sum())
                    else:
                        esum += float((kl * w).sum()); ecnt += int(w.sum()); easum += float((agree * w).sum())
                    del hs, ht, kl, agree, _l
                ln = lane_of(e)
                lsum[ln] = lsum.get(ln, 0.0) + esum
                lcnt[ln] = lcnt.get(ln, 0) + ecnt
                lasum[ln] = lasum.get(ln, 0.0) + easum
                tot += esum
                ntok += ecnt
                asum += easum
                rows.append((e.get("i", e.get("id")), ln, ecnt, esum / max(1, ecnt)))
        return {"kl": tot / max(1, ntok), "tokens": ntok,
                "lanes": {ln: lsum[ln] / max(1, lcnt[ln]) for ln in lsum},
                "lane_tokens": dict(lcnt), "examples": rows,
                "top20": asum / max(1, ntok), "lane_top20": {ln: lasum[ln] / max(1, lcnt[ln]) for ln in lasum}}


# ----------------------------------------------------------------------------------------------- report
class ExamplesReport:
    """One JSONL row per walked example: original / processed tokens, coverage, think + final-answer flags."""

    def __init__(self, path):
        self.path = path
        self.n = self.tok = self.think = self.closed = self.ans = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "a").close()

    def add(self, e, kl_sum, processed, phase, step):
        included = processed == e["n_tokens"] and ((not e["has_think"]) or e["answer_after_think"])
        row = {"step": step, "phase": phase, "i": e["i"], "lane": e["topic"], "orig_tokens": e["n_tokens"],
               "processed_tokens": processed, "coverage": processed / max(1, e["n_tokens"]),
               "has_think": e["has_think"], "think_closed": e["think_closed"],
               "answer_after_think": e["answer_after_think"], "final_answer_included": bool(included),
               "kl_per_token": kl_sum / max(1, processed)}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.n += 1
        self.tok += processed
        self.think += int(e["has_think"])
        self.closed += int(e["think_closed"])
        self.ans += int(included)

    def line(self):
        return (f"examples walked {self.n} | tokens {self.tok:,} | with think {self.think} | think closed "
                f"{self.closed} | final answer included {self.ans}/{self.n} | every example processed 100% (asserted)")


# ----------------------------------------------------------------------------------------------- checkpoints
def save_ckpt(path, lats, opt, state: dict, log=print):
    """Atomic: latents + optimizer + counters + RNG. Resumable with load_ckpt (memory checkpoint-every-long-run)."""
    tmp = path + ".tmp"
    torch.save({"latents": [l.detach().cpu() for l in lats],
                "opt": opt.state_dict() if opt is not None else None,
                "state": state,
                "rng": {"py": random.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                "saved_at": time.time()}, tmp)
    os.replace(tmp, path)
    log(f"    ckpt -> {path} (step {state.get('step')}, best {state.get('best', float('nan')):.4f}, "
        f"{os.path.getsize(path) / 2**30:.2f} GB)")


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng(rng):
    try:
        random.setstate(rng["py"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:  # RNG restore is best-effort; the latents/optimizer are what matter
        print(f"    rng restore skipped: {e}", flush=True)


def mirror_hf(path, repo, path_in_repo, log=print):
    """Best-effort HF mirror of a checkpoint (never fatal -- upload failures are logged, the run continues)."""
    if not repo:
        return
    try:
        from huggingface_hub import HfApi
        t0 = time.time()
        HfApi().upload_file(path_or_fileobj=path, path_in_repo=path_in_repo, repo_id=repo)
        log(f"    ckpt mirrored -> hf://{repo}/{path_in_repo} ({time.time() - t0:.0f}s)")
    except Exception as e:
        log(f"    ckpt HF mirror FAILED (non-fatal): {type(e).__name__}: {str(e)[:160]}")
