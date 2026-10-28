"""CARRIED CONTEXT, ONE LAYER AT A TIME -- the block-wise form of sbr_examples' walker.

Cody, 2026-09-22 (hessian_carry.py): *"we need to make sure we carry context across the fused gated deltanet layers
so my reasoning can be repaired -- thats why its broken, because it never trained it with the context going across
the deltanet."* Cody, 2026-09-24: every training loop carries it -- including the block-wise ones (ternary_gptq's
Hessian pass, block_recon), which never run the whole model: they push a STREAM of rows through ONE decoder layer
at a time (`block_recon.run_layer`), and every row started from a blank DeltaNet state.

WHAT THIS FILE DOES (reuse, not re-derive: sbr_examples' mechanism, per layer):
  * `example_rows(examples, chunk)` lays each COMPLETE example (never cut) out as consecutive `chunk`-token rows
    of a fixed [rows, chunk] pool; the LAST row of an example is RIGHT-padded. Right padding is exact for every real
    token: attention and the DeltaNet recurrence are causal, so no real token ever sees a later padding token, and
    the only thing padding reaches -- the state after an example's last chunk -- is thrown away. It also removes
    the 1-token-chunk trap (sbr_examples.chunk_bounds: a 1-token chunk dispatches the no-backward recurrent
    kernel) because no row is ever shorter than `chunk`. `valid` masks the padding out of Hessians and losses.
  * `run_layer_carried(layer, rows, meta, ...)` runs one layer over the pool example by example, carrying ONE cache
    (`sbr_examples.make_train_cache`: the weakref rebind fix) through the example's chunks, reset between
    examples, detached between chunks under grad (truncated BPTT, like the walker).

THE TRAPS (found 2026-09-24 reading transformers 5.16.1 modeling_qwen3_5.py, handled here):
  1. A carried ATTENTION chunk has q_len < kv_len. With attention_mask=None, SDPA's is_causal is TOP-LEFT aligned:
     queries of chunk 2 would see only chunk 1's first keys. -> an explicit OFFSET causal mask (query at absolute
     position p sees keys 0..p) for every attention chunk after the first.
  2. The decoder layer hands attention_mask to the DeltaNet layer too, where it is a 2-D PADDING mask
     (apply_mask_to_padding_states). -> DeltaNet layers get attention_mask=None (the model's own forward does the
     same: it builds a separate linear-attention mask that is None without padding).
  3. RoPE positions must be ABSOLUTE (s0..s0+chunk-1), and qwen3_5's rotary is mrope from transformers 5.17: a 2-D
     position_ids crashes, the 3 text sections share one position (same fix as block_recon :3142).
  4. The DeltaNet layer reads/writes ONLY cache_params.layers[layer_idx] (conv tail + recurrent state) and needs
     nothing else; the attention layer needs past_key_values + the offset position embeddings + the mask.
The falsifier (carry_equiv_layers in kaggle_carry_cell.py): layer-by-layer carried outputs must equal the
WHOLE-MODEL carried walk's per-layer outputs (hooks) on the same examples; chunk 0 must equal the old reset path.
"""
from __future__ import annotations

import torch

import sbr_examples as SBX


def example_rows(examples, chunk: int, pad_id: int = 0):
    """[{"ids": LongTensor[n]} ...] -> (rows LongTensor[R, chunk], meta list of dicts, valid BoolTensor[R, chunk]).
    meta[r] = {"ex": example index, "c": chunk index, "s0": absolute start, "n": real tokens in this row,
               "first": c == 0, "last": last row of its example}. Every token of every example is laid out."""
    rows, meta, valid = [], [], []
    for ei, e in enumerate(examples):
        ids = e["ids"] if torch.is_tensor(e["ids"]) else torch.tensor(e["ids"])
        ids = ids.reshape(-1)
        n = int(ids.numel())
        if n == 0:
            continue
        starts = list(range(0, n, chunk))
        for c, s0 in enumerate(starts):
            piece = ids[s0:s0 + chunk]
            k = int(piece.numel())
            row = torch.full((chunk,), int(pad_id), dtype=torch.long)
            row[:k] = piece
            v = torch.zeros(chunk, dtype=torch.bool)
            v[:k] = True
            rows.append(row); valid.append(v)
            meta.append({"ex": ei, "c": c, "s0": s0, "n": k, "first": c == 0, "last": c == len(starts) - 1})
    if not rows:
        raise ValueError("example_rows: no tokens in any example")
    laid = sum(m["n"] for m in meta)
    want = sum(int((e["ids"] if torch.is_tensor(e["ids"]) else torch.tensor(e["ids"])).numel()) for e in examples)
    assert laid == want, f"example_rows laid out {laid} of {want} tokens -- never allowed"
    return torch.stack(rows), meta, torch.stack(valid)


def budget_examples(examples, budget_tokens: int, seed: int = 0, log=print):
    """Pick COMPLETE examples for a token budget -- hessian_carry.py's proven policy (the 27B carried Hessian:
    2,184 examples / 512,179 tokens): a SEEDED shuffle (random.Random(0)) so the budget covers every lane, then walk
    until the budget is met; an example is never cut, so the total overshoots by at most the last one. A file-order
    prefix is the lane-blocked-ruler trap: calib.txt is written lane by lane, and on the 0.8B carry test the first
    long trace alone filled the budget (1 example, 39,049 tokens, ONE lane)."""
    import collections
    import random
    ex = list(examples)
    random.Random(seed).shuffle(ex)
    picked, used = [], 0
    for e in ex:
        if used >= budget_tokens:
            break
        picked.append(e)
        used += int(e["n_tokens"])
    lanes = collections.Counter(e.get("lane_idx", e.get("topic")) for e in picked)
    log(f"  carry pool (seeded shuffle, hessian_carry's policy): {len(picked)} complete examples from {len(lanes)} "
        f"lanes, {used:,} tokens for a {budget_tokens:,}-token budget")
    return picked


def example_spans(meta):
    """meta -> [(example index, [row indices in chunk order])] in pool order."""
    spans, cur, rows = [], None, []
    for r, m in enumerate(meta):
        if m["ex"] != cur:
            if rows:
                spans.append((cur, rows))
            cur, rows = m["ex"], []
        assert m["c"] == len(rows), f"row {r}: chunk {m['c']} out of order (expected {len(rows)})"
        rows.append(r)
    if rows:
        spans.append((cur, rows))
    return spans


def rope_at(rotary_emb, like, s0: int, length: int):
    """(position_ids [1, L], position_embeddings) for ABSOLUTE positions s0..s0+L-1 -- mrope-safe (trap 3)."""
    pid = torch.arange(s0, s0 + length, device=like.device).unsqueeze(0)
    # --offload keeps the frozen model (and its rotary_emb) on the CPU while one block runs on the GPU: compute the rope where its
    # buffers live and hand the result back on the data's device (2026-09-27: offload + carry crashed "mat2 on cuda, others on cpu").
    # Same device (every non-offload caller) = the old path exactly.
    rdev = next((b.device for b in rotary_emb.buffers()), like.device)
    lk, pr = (like[:1], pid) if rdev == like.device else (like[:1].to(rdev), pid.to(rdev))
    try:
        pe = rotary_emb(lk, pr)
    except (IndexError, RuntimeError):
        pe = rotary_emb(lk, pr.unsqueeze(0).expand(3, 1, length))
    if rdev != like.device:
        pe = tuple(t.to(like.device) for t in pe)
    return pid, pe


def offset_causal_mask(q_len: int, past: int, like: torch.Tensor):
    """[1, 1, q_len, past + q_len] additive mask: the query at absolute position past+i sees keys 0..past+i (trap 1)."""
    kv = past + q_len
    j = torch.arange(kv, device=like.device)
    i = torch.arange(q_len, device=like.device).unsqueeze(1) + past
    m = torch.zeros(q_len, kv, device=like.device, dtype=like.dtype)
    return m.masked_fill(j > i, torch.finfo(like.dtype).min).view(1, 1, q_len, kv)


def is_linear_attention(layer) -> bool:
    return getattr(layer, "block_type", None) == "linear_attention" or hasattr(layer, "linear_attn")


def layer_chunk(layer, hidden, cache, s0: int, rotary_emb):
    """ONE decoder layer on ONE chunk of ONE example (hidden [1, L, D]) with that example's cache carried in."""
    L = int(hidden.shape[1])
    if getattr(layer, "gradient_checkpointing", False) and layer.training:
        raise SystemExit(CKPT_TRAP)
    pid, pe = rope_at(rotary_emb, hidden, s0, L)
    mask = None
    if not is_linear_attention(layer) and s0 > 0:
        mask = offset_causal_mask(L, s0, hidden)              # trap 1 (chunk 0: None == the plain causal path)
    out = layer(hidden, position_embeddings=pe, attention_mask=mask, position_ids=pid, past_key_values=cache)
    return out[0] if isinstance(out, tuple) else out


CKPT_TRAP = ("carried context + HF gradient checkpointing: transformers' GradientCheckpointingLayer sets "
             "past_key_values=None on every decoder layer in train mode (one warning line, then every chunk silently "
             "RESETS -- measured 2026-09-24 on the tiny Qwen3.5: logits off by 0.2188 vs 0.00000015 with it off). "
             "Turn gradient checkpointing off for a carried run. REFUSING.")


def assert_cache_live(cache, n_seen: int, where: str = ""):
    """Refuse if a carried cache holds NO state after n_seen tokens of an example: an attention layer's keys must
    span n_seen positions and every DeltaNet layer's recurrent state must be initialised. This is what a cache
    dropped by gradient checkpointing (or a forward that ignored past_key_values) looks like."""
    for i, L in enumerate(getattr(cache, "layers", [])):
        k = getattr(L, "keys", None)
        if hasattr(L, "is_recurrent_states_initialized"):
            _ini = L.is_recurrent_states_initialized
            if not all(bool(x) for x in (_ini.values() if isinstance(_ini, dict) else _ini)):
                raise SystemExit(f"{where}: layer {i} DeltaNet state was never written. " + CKPT_TRAP)
        elif torch.is_tensor(k) and k.dim() >= 2:
            if int(k.shape[-2]) != int(n_seen):
                raise SystemExit(f"{where}: layer {i} attention cache spans {int(k.shape[-2])} of {n_seen} tokens. "
                                 + CKPT_TRAP)
        elif k is None or not torch.is_tensor(k):
            raise SystemExit(f"{where}: layer {i} attention cache is empty after {n_seen} tokens. " + CKPT_TRAP)


# The row being processed right now: forward hooks (e.g. GPTQ's Hessian x^T x) read CURRENT["valid"] ([L] bool) to
# drop an example's right-padding -- padded positions still produce NONZERO linear inputs (a padded query attends
# to real keys; the DeltaNet conv reaches 3 real tokens back), so zeroing the stream is not enough.
CURRENT = {"row": None, "valid": None}


def real_tokens(x, valid=None):
    """[1, L, D] (or [L, D]) linear input -> [n_real, D]: the positions of the current row that are real tokens."""
    v = CURRENT["valid"] if valid is None else valid
    x2 = x.reshape(-1, x.shape[-1])
    return x2 if v is None else x2[v.to(x2.device).reshape(-1)]


def trimmed_len(m, full: int, trim: int) -> int:
    """How many positions of row `m` (a meta dict) a trimmed walk runs: the row's real tokens rounded UP to a multiple
    of `trim`, never below 2 (a 1-token chunk with a carried cache takes the decode kernel -- the trap example_rows'
    padding avoids) and never above the row. trim <= 0, or a row that is not its example's padded last row -> full."""
    if trim <= 0 or not m.get("last") or int(m["n"]) >= full:
        return full
    return min(full, max(2, -(-int(m["n"]) // trim) * trim))


def run_layer_carried(layer, stream, meta, config, rotary_emb, *, grad: bool = False, loss_fn=None, on_chunk=None,
                      device=None, out_dtype=None, valid=None, inplace: bool = False,
                      reset_every_chunk: bool = False, trim: int = 0):
    """Run `layer` over the whole pool `stream` [R, chunk, D], example by example, carrying one cache per example.
    Returns the output pool (same shape, on stream's device, `out_dtype` or stream's dtype) unless loss_fn is given.
      grad    : backward per chunk (loss_fn(out_chunk, row) must return a scalar); the cache is detached between
                chunks (truncated BPTT). Returns the summed loss float instead of an output pool.
      on_chunk: callback(out_chunk [1, L, D], row index) for hooks-free consumers (e.g. teacher targets).
      valid   : the [R, chunk] mask from example_rows -- published per row in CURRENT for forward hooks.
      reset_every_chunk: the FALSIFIER CONTROL (hessian_carry.py's --reset-every-chunk): the SAME rows, but a fresh
                cache and positions from 0 on every row -- the old reset math on identical tokens, so any difference
                from the carried run is the carried context and nothing else.
      trim    : 0 (default) = every row at full length, the proven path. > 0 (10-03, Cody "oh wow, yeah run a smoke": on
                the 27B recon 900 of 1334 rows are an example's padded last row, 43.6% of the forward math) = an
                example's LAST row runs only up to trimmed_len (its real tokens rounded up to a multiple of `trim`); the
                rest of that output row is zero-filled. Forward-only: loss_fn / on_chunk / grad callers index full rows.
    Every row of every example is processed in chunk order; padding rows are processed and simply never read."""
    if trim and (grad or loss_fn is not None or on_chunk is not None or reset_every_chunk):
        raise SystemExit("run_layer_carried(trim>0) is forward-only: loss_fn / on_chunk / grad / reset_every_chunk callers "
                         "read full-length rows -- REFUSING")
    dev = device or stream.device
    # inplace: write each row's output over its own input AFTER reading it -- safe under carry because the context
    # from earlier chunks lives in the cache, not in the stream (ternary_gptq's memory-saving advance).
    out = None if loss_fn is not None else (stream if inplace else
                                            torch.empty_like(stream, dtype=out_dtype or stream.dtype))
    total = 0.0
    for _ex, rows in example_spans(meta):
        cache = SBX.make_train_cache(config)
        for r in rows:
            if reset_every_chunk:
                cache = SBX.make_train_cache(config)
            CURRENT["row"] = r
            use = trimmed_len(meta[r], int(stream.shape[1]), trim)
            CURRENT["valid"] = None if valid is None else valid[r][:use]
            x = stream[r:r + 1, :use].to(dev)
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                y = layer_chunk(layer, x, cache, 0 if reset_every_chunk else meta[r]["s0"], rotary_emb)
                if loss_fn is not None:
                    loss = loss_fn(y, r)
                    if grad:
                        loss.backward()
                    total += float(loss.detach())
            if on_chunk is not None:
                on_chunk(y.detach(), r)
            if out is not None:
                out[r:r + 1, :use] = y.detach().to(out.device, out.dtype)
                if use < out.shape[1]:
                    out[r:r + 1, use:] = 0
            if grad:
                SBX.detach_cache_(cache)
        del cache
    CURRENT["row"] = CURRENT["valid"] = None
    return total if loss_fn is not None else out


# ------------------------------------------------------------------------------------ whole-model trainers
# The (a) trainers (CARRY_AUDIT.md: a whole model forward on independent rows) change the same way anchored_qat did:
# the training unit becomes a COMPLETE example walked in `chunk` pieces with one carried cache per model.


def text_examples(corpus, tokenizer, examples_jsonl=None, min_tokens: int = 2):
    """COMPLETE examples of a build_corpus corpus WITHOUT a ruler manifest (the 0.5B/0.8B test cells): every span of
    `<corpus>.examples.jsonl` tokenised on its own -> [{"i", "lane", "topic", "ids", "n_tokens"}]. With a lanes
    manifest + union ruler, use block_recon.load_examples instead (its per-lane tokenisation lines up with the
    ruler's reserved rows and applies the moat). An example shorter than `min_tokens` has no next token to train on
    and is left out (a lone token through a carried cache hits the no-backward recurrent kernel)."""
    import json
    from pathlib import Path
    corpus = Path(corpus)
    exj = Path(examples_jsonl) if examples_jsonl else corpus.with_name(corpus.name + ".examples.jsonl")
    if not exj.exists():
        raise SystemExit(f"carried context needs {exj} (calibration_corpus.build_corpus writes it beside the corpus; "
                         f"rebuilding a cached corpus emits it, text byte-identical)")
    text = corpus.read_text(encoding="utf-8", errors="replace")
    out = []
    for line in exj.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        ids = tokenizer(text[s["start"]:s["end"]], return_tensors="pt", add_special_tokens=False).input_ids[0]
        if int(ids.numel()) < min_tokens:
            continue
        out.append({"i": s["i"], "lane": s.get("lane"), "topic": s.get("topic"), "ids": ids,
                    "n_tokens": int(ids.numel())})
    if not out:
        raise SystemExit(f"{exj}: no example with >= {min_tokens} tokens")
    return out


def split_held(examples, held_tokens: int, budget_tokens: int = 0, seed: int = 0, log=print):
    """(held, train) from one seeded shuffle (budget_examples' policy): the first examples up to `held_tokens` are
    held out (never trained), the rest -- up to `budget_tokens` when given -- train. Disjoint by construction."""
    pool = budget_examples(examples, (held_tokens + budget_tokens) if budget_tokens else 1 << 62, seed=seed, log=log)
    held, train, h = [], [], 0
    for e in pool:
        if h < held_tokens:
            held.append(e)
            h += int(e["n_tokens"])
        else:
            train.append(e)
    if not held or not train:
        raise SystemExit(f"carried context: {len(train)} training / {len(held)} held-out examples -- corpus too small")
    log(f"  carried split: {len(held)} held-out examples ({h:,} tokens) | {len(train)} training examples "
        f"({sum(int(e['n_tokens']) for e in train):,} tokens)")
    return held, train


def sample_examples(pool, budget_tokens: int, gen=None):
    """Random COMPLETE examples until `budget_tokens` (the row step's batch x seq_len) -- the carried batch."""
    picked, used = [], 0
    while used < max(1, int(budget_tokens)):
        e = pool[int(torch.randint(0, len(pool), (1,), generator=gen))]
        picked.append(e)
        used += int(e["n_tokens"])
    return picked, used


def carried_chunks(ids, chunk: int, models, train=None, contexts=None, **fwd):
    """Walk ONE complete example (1-D ids) through each model in `chunk`-token pieces with ONE carried cache per
    model (sbr_examples.make_train_cache: DeltaNet state + conv tail + attention KV + positions), reset per example,
    detached between chunks (truncated BPTT). train[i] False -> model i runs under no_grad (a teacher).
    contexts[i]: an optional zero-argument context-manager factory entered around model i's forward (the welded
    one-model self-distill passes the SAME model twice, the teacher slot under its teacher_mode()).
    Yields (s0, s1, x [1, L], tgt, outs): tgt = the example's NEXT tokens across the chunk end (the last chunk has
    one fewer), so logits[0, :tgt.numel()] are exactly the positions that predict a real token. The caller computes
    its loss on outs and backward()s it before asking for the next chunk. Every token is walked; asserted."""
    from contextlib import nullcontext
    ids = ids.reshape(-1)
    n = int(ids.numel())
    train = [True] * len(models) if train is None else list(train)
    contexts = [None] * len(models) if contexts is None else list(contexts)
    caches = [SBX.make_train_cache(m.config) for m in models]
    done = 0
    for s0, s1 in SBX.chunk_bounds(n, chunk):
        outs = []
        for i, m in enumerate(models):
            dev = next(m.parameters()).device
            with (nullcontext() if train[i] else torch.no_grad()), (contexts[i]() if contexts[i] else nullcontext()):
                o = m(input_ids=ids[s0:s1].unsqueeze(0).to(dev), past_key_values=caches[i], use_cache=True, **fwd)
            caches[i] = o.past_key_values
            if s0 == 0:
                assert_cache_live(caches[i], s1, where=f"carried_chunks model {i}")
            outs.append(o)
        yield s0, s1, ids[s0:s1].unsqueeze(0), ids[s0 + 1:min(s1 + 1, n)], outs
        del outs
        done += s1 - s0
        for i, c in enumerate(caches):
            if train[i]:
                SBX.detach_cache_(c)
    assert done == n, f"carried_chunks walked {done} of {n} tokens -- never allowed"
    del caches


def next_token_ce(logits, tgt, reduction: str = "sum"):
    """CE of logits[1, L, V] against the example's next tokens (carried_chunks' tgt)."""
    k = int(tgt.numel())
    if k == 0:
        return logits.float().sum() * 0.0
    return torch.nn.functional.cross_entropy(logits[0, :k].float(), tgt.to(logits.device), reduction=reduction)


def next_token_kl(t_logits, s_logits, k: int, reduction: str = "sum"):
    """KL(teacher || student) over the first k positions (those that predict a real next token)."""
    if k == 0:
        return s_logits.float().sum() * 0.0
    lp_t = torch.log_softmax(t_logits[0, :k].float().to(s_logits.device), -1)
    lp_s = torch.log_softmax(s_logits[0, :k].float(), -1)
    kl = (lp_t.exp() * (lp_t - lp_s)).sum(-1)
    return kl.sum() if reduction == "sum" else kl.mean()


@torch.no_grad()
def held_ce(model, held, chunk: int):
    """Token-mean next-token CE over held-out COMPLETE examples, carried (the eval the carried trainer is judged by)."""
    tot, cnt = 0.0, 0
    for e in held:
        for _s0, _s1, _x, tgt, (o,) in carried_chunks(e["ids"], chunk, [model], train=[False]):
            tot += float(next_token_ce(o.logits, tgt))
            cnt += int(tgt.numel())
    return tot / max(1, cnt)


# ---- the 0.5B shadow-body family (shadow_body_*_05b, sparse_*_05b, self_distill_rot_05b): teacher -> student KL


@torch.no_grad()
def held_kl(teacher, student, held, chunk: int, contexts=None):
    """Token-mean KL(teacher || student) over held-out COMPLETE examples, both caches carried."""
    tot, cnt = 0.0, 0
    for e in held:
        for *_, tg, (to, so) in carried_chunks(e["ids"], chunk, [teacher, student], train=[False, False],
                                               contexts=contexts):
            tot += float(next_token_kl(to.logits, so.logits, int(tg.numel())))
            cnt += int(tg.numel())
    return tot / max(1, cnt)


def kl_step(teacher, student, pool, budget_tokens: int, pos: int, opt, lr: float, chunk: int, contexts=None):
    """ONE optimizer step of the family's KL objective, carried: complete examples up to budget_tokens (the row
    step's rows x seq; never cut), `pos` positions drawn uniformly over the WHOLE batch (the row step's estimator:
    it sampled POS of all B x seq positions), KL mean over them, backward per chunk, then opt.step(). Returns the
    float loss. opt=None: gradients only (the caller owns lr / zero_grad / extra terms / step)."""
    if opt is not None:
        for g in opt.param_groups:
            g["lr"] = lr
    picked, _ = sample_examples(pool, budget_tokens)
    T = sum(int(e["n_tokens"]) - 1 for e in picked)
    npos = max(1, min(int(pos), T))
    chosen = torch.zeros(T, dtype=torch.bool)
    chosen[torch.randperm(T)[:npos]] = True
    if opt is not None:
        opt.zero_grad()
    tot, off = 0.0, 0
    for e in picked:
        for *_, tg, (to, so) in carried_chunks(e["ids"], chunk, [teacher, student], train=[False, True],
                                               contexts=contexts):
            k = int(tg.numel())
            m = chosen[off:off + k].to(so.logits.device)
            off += k
            if not bool(m.any()):
                continue
            lp_t = torch.log_softmax(to.logits[0, :k][m].float().to(so.logits.device), -1)
            lp_s = torch.log_softmax(so.logits[0, :k][m].float(), -1)
            kl = (lp_t.exp() * (lp_t - lp_s)).sum(-1).sum() / npos
            kl.backward()
            tot += float(kl.detach())
    assert off == T, f"kl_step walked {off} of {T} predicted positions"
    if opt is not None:
        opt.step()
    return tot


def family_examples(paths, tokenizer, held_tokens: int, log=print):
    """(held, train) for the 0.5B family: held out of the FIRST corpus (the row code held its tail rows), the rest
    of it + every extra corpus trains -- complete examples from each corpus's examples sidecar."""
    held, train = split_held(text_examples(paths[0], tokenizer), held_tokens, 0, log=log)
    for p in paths[1:]:
        train = train + text_examples(p, tokenizer)
    return held, train


def kl_sum_step(teacher, student, pool, budget_tokens: int, chunk: int, row_len: int):
    """The row code's kl_div(s_lp, t_lp, log_target=True, reduction="batchmean") over [B, row_len] rows = KL SUMMED
    over positions / B. Carried equivalent at the same scale: complete examples up to budget_tokens (never cut), KL
    summed over every predicted position / (T / row_len); backward per chunk. The caller owns zero_grad / step.
    Returns the float loss."""
    picked, _ = sample_examples(pool, budget_tokens)
    T = sum(int(e["n_tokens"]) - 1 for e in picked)
    rows_eq = max(1e-9, T / float(row_len))
    tot = 0.0
    for e in picked:
        for *_, tg, (to, so) in carried_chunks(e["ids"], chunk, [teacher, student], train=[False, True]):
            kl = next_token_kl(to.logits, so.logits, int(tg.numel())) / rows_eq
            kl.backward()
            tot += float(kl.detach())
    return tot


def ce_step(model, pool, budget_tokens: int, chunk: int, scale: float = 1.0):
    """ONE carried micro-batch of plain next-token CE: complete examples up to budget_tokens (the row step's
    batch x seq_len; never cut), each walked with its cache carried; CE token-mean over the batch x `scale`
    (e.g. 1 / grad_accum), backward per chunk. The caller owns clip / step / zero_grad. Returns the float CE."""
    picked, _ = sample_examples(pool, budget_tokens)
    T = sum(int(e["n_tokens"]) - 1 for e in picked)
    tot = 0.0
    for e in picked:
        for *_, tg, (o,) in carried_chunks(e["ids"], chunk, [model]):
            ce = next_token_ce(o.logits, tg) / max(1, T)
            (ce * scale).backward()
            tot += float(ce.detach())
    return tot
