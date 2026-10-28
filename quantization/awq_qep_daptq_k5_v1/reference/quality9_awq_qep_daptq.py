"""Isolated, resumable AWQ + QEP + DAPTQ K5 conversion on the assigned L4.

Only HumanEval/127 is answered. All calibration is separate, complete gold
training text. Pilot converts block zero and checkpoints; full resumes at one.
No original model, conversion, kernel, answer or paper is overwritten.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path('/content/tk-quality9-1027')
PARENT = ROOT / 'code-v6'
CODE = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'awq-qep-daptq-127-r2-evidence'
ART = ROOT / 'awq-qep-daptq-127-r2-artifacts'
OUT = ROOT / 'awq-qep-daptq-127-r2'
DRIVE = Path('/content/drive/MyDrive/taardis/tk-quality9-1027')
sys.path[:0] = [str(CODE), str(PARENT / 'benchmarks'), str(PARENT)]
os.environ['TK_K5_REFERENCE'] = str(PARENT / 'benchmarks/gemv_reference.py')

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
import carry_stream as carry
import k5_prefill_cuda_l4 as kernel
from k5_manifest_runtime import K5Linear, COUNTERS
from tk_codec import pack_k5_hiscale
from quality9_runtime import save_json, sha, thash, source_name, set_value, load_model, MODEL, REVISION
from quality9_eval import require_remote, generate
from quality9_tasks import score, sandbox_check
from quality9_k5_daptq import statistics, qep_target, refine, algebra_check


def stopped():
    if (ROOT / 'STOP').exists():
        raise RuntimeError('Human STOP present; no automatic resume')


class Saves:
    """Existing Drive daemon and HF API, asynchronous I/O only, no new daemon."""
    def __init__(self, repo):
        from huggingface_hub import HfApi
        self.api = HfApi(token=os.environ['HF_TOKEN'])
        self.api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
        if not self.api.repo_info(repo, repo_type='dataset').private:
            raise RuntimeError('Hessian backup repository must be private')
        self.repo = repo
        self.drive_pool = ThreadPoolExecutor(max_workers=1)
        self.hf_pool = ThreadPoolExecutor(max_workers=1)
        self.drive_jobs, self.hf_jobs = [], []

    def put(self, path, hf=False, repo_path=None):
        path = Path(path)
        target = DRIVE / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        future = self.drive_pool.submit(shutil.copy2, path, target)
        self.drive_jobs.append((path, target, future))
        if hf:
            dest = 'attempt-r2/' + (repo_path or str(path.relative_to(EVIDENCE)))
            def upload():
                future.result()
                return str(self.api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                           repo_id=self.repo, repo_type='dataset', commit_message='Preserve exact combined-method evidence: ' + dest).oid)
            self.hf_jobs.append((path, dest, self.hf_pool.submit(upload)))

    def finish(self):
        for source, target, job in self.drive_jobs:
            job.result()
            if source.stat().st_size != target.stat().st_size:
                raise RuntimeError('Drive copy size mismatch: ' + str(source))
        commits = [{'file': dest, 'bytes': path.stat().st_size, 'commit': job.result()}
                   for path, dest, job in self.hf_jobs]
        self.drive_pool.shutdown(); self.hf_pool.shutdown()
        sys.path.insert(0, '/content/forge')
        from drive_sync import wait_uploaded
        paths = sorted({str(target) for _, target, _ in self.drive_jobs})
        if not wait_uploaded(paths, exact_paths=True):
            raise RuntimeError('Exact-path Drive daemon confirmation failed')
        return {'drive_confirmed': paths, 'private_hessian_repo': self.repo, 'hf_commits': commits}


def capture(block, stream, meta, valid, cfg, rotary, names):
    stores = {name: [] for name in names}
    def hook(name):
        def take(_module, args, _result):
            value = carry.real_tokens(args[0].detach()).cpu().contiguous()
            stores[name].append(value)
        return take
    hooks = [block.get_submodule(n).register_forward_hook(hook(n)) for n in names]
    try:
        output = carry.run_layer_carried(block, stream, meta, cfg, rotary,
                                        device='cuda', valid=valid)
    finally:
        for handle in hooks:
            handle.remove()
    result = {name: torch.cat(values) for name, values in stores.items()}
    count = sum(m['n'] for m in meta)
    if any(x.shape[0] != count for x in result.values()):
        raise RuntimeError('Captured padding or dropped calibration tokens')
    return output, result


def packed(payload):
    c, s, m = payload
    return K5Linear((c.cuda(), s.cuda(), m.cpu()), kernel)


def awq_loss(weights, names, x, scaling):
    """AWQ linear-output search, measured using the actual shipping K5 kernel.

    Shared-norm consumers share one scale and their squared output errors sum.
    This is the upstream linear-output objective, not a nonlinear block search
    or the upstream clipping stage. Every one of the 16 examples contributes.
    """
    loss = torch.zeros((), device='cuda', dtype=torch.float64)
    for name in names:
        w = weights[name]
        module = packed(pack_k5_hiscale(w * scaling[None, :]))
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


def fold_awq(block, weights, original_inputs):
    is_delta = block.block_type == 'linear_attention'
    first = ['linear_attn.' + n for n in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a']] if is_delta else ['self_attn.' + n for n in ['q_proj', 'k_proj', 'v_proj']]
    groups = [('input_layernorm', first),
              ('post_attention_layernorm', ['mlp.gate_proj', 'mlp.up_proj']),
              ('mlp.up_proj', ['mlp.down_proj'])]
    input_scalings = {n: torch.ones(w.shape[1], device='cuda') for n, w in weights.items()}
    report = []
    for previous, names in groups:
        stopped()
        x = original_inputs[names[0]]
        act = x.float().abs().mean(0).cuda()
        prev = block.get_submodule(previous)
        norm = previous.endswith('layernorm')
        old_norm = prev.weight.detach().clone() if norm else None
        best = None
        for i in range(20):
            stopped()
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
            value = awq_loss(weights, names, x, scaling)
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


def read_block(source, index, cfg, number):
    with torch.device('meta'):
        block = Qwen3_5DecoderLayer(cfg, number)
    for name, parameter in list(block.named_parameters()):
        src = source_name(f'model.layers.{number}.{name}')
        with safe_open(str(source / index[src]), framework='pt', device='cpu') as handle:
            tensor = handle.get_tensor(src).to(torch.float16).contiguous()
        if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
            raise RuntimeError('Invalid source block parameter: ' + src)
        set_value(block, name, torch.nn.Parameter(tensor.cuda(), requires_grad=False))
    return block.eval().requires_grad_(False)


def save_matrix(name, payload, layout, original, saves):
    old = ROOT / 'artifacts' / original['files'][name]['tk']
    with safe_open(str(old), framework='pt', device='cpu') as handle:
        shapes = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
        dtypes = {key: handle.get_slice(key).get_dtype() for key in handle.keys()}
    if set(shapes) != {'codes', 'scales', 'meta'}:
        raise RuntimeError('Original K5 storage schema changed: ' + name)
    raw = (payload[0].to(torch.int16) + 121).to(torch.uint8).reshape(shapes['codes']).cpu().contiguous()
    # Match the converter's flat scale storage, as in quality9_scale_first.py.
    # A two-dimensional scale shape changes the padded safetensors header.
    scales = payload[1].reshape(shapes['scales']).cpu().contiguous()
    metadata = payload[2].reshape(shapes['meta']).cpu().contiguous()
    path = ART / original['files'][name]['tk']
    if path.exists():
        raise RuntimeError('Refusing to overwrite a converted matrix: ' + name)
    save_file({'codes': raw, 'scales': scales, 'meta': metadata}, str(path))
    if path.stat().st_size != old.stat().st_size:
        raise RuntimeError('K5 serialized matrix size changed: ' + name)
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if ({key: handle.get_slice(key).get_shape() for key in handle.keys()} != shapes or
                {key: handle.get_slice(key).get_dtype() for key in handle.keys()} != dtypes):
            raise RuntimeError('K5 serialized shape or dtype changed: ' + name)
    check = load_file(str(path), device='cuda')
    if not torch.equal(check['codes'].reshape_as(payload[0]).to(torch.int16) - 121, payload[0].to(torch.int16)):
        raise RuntimeError('Saved K5 codes changed')
    if (not torch.equal(check['scales'].reshape_as(payload[1]), payload[1]) or
            not torch.equal(check['meta'].cpu().reshape_as(payload[2]), payload[2].cpu())):
        raise RuntimeError('Saved integer scales changed')
    entry = copy.deepcopy(original['files'][name])
    entry['fp8'] = '../artifacts/' + entry['fp8']
    entry['sha256']['tk'] = sha(path)
    layout['files'][name] = entry
    saves.put(path)


def convert_block(number, fp_stream, q_stream, meta, valid, cfg, rotary, source, index, original, layout, saves):
    stopped()
    if shutil.disk_usage('/').free < 10 * 1024**3:
        raise RuntimeError('Root-space reserve reached; preserve checkpoints and pending Drive cache')
    tick = time.monotonic()
    fp = read_block(source, index, cfg, number)
    names = [n for n, m in fp.named_modules() if isinstance(m, torch.nn.Linear)]
    fp_out, inputs = capture(fp, fp_stream, meta, valid, cfg, rotary, names)
    if number == 0:
        check = algebra_check(fp.get_submodule(names[0]).weight.float(), inputs[names[0]].cuda())
        path = EVIDENCE / 'algebra-check.json'; save_json(path, check); saves.put(path, hf=True)
    q = copy.deepcopy(fp)
    weights = {n: fp.get_submodule(n).weight.detach().float().clone() for n in names}
    scaling, awq = fold_awq(q, weights, inputs)
    folded, _ = capture(q, fp_stream, meta, valid, cfg, rotary, [])
    rel = float((folded[valid].float() - fp_out[valid].float()).norm() / fp_out[valid].float().norm())
    if rel > 0.005 or not torch.isfinite(folded).all():
        raise RuntimeError(f'AWQ folded-block equivalence exceeds existing numerical limit: {rel:.8f}')
    del folded
    stages = [[n for n in names if n.rsplit('.', 1)[-1] in ['q_proj', 'k_proj', 'v_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b']],
              [n for n in names if n.endswith(('o_proj', 'out_proj'))],
              ['mlp.gate_proj', 'mlp.up_proj'], ['mlp.down_proj']]
    report = {'block': number, 'awq': awq, 'folded_block_relative_l2': rel, 'linears': []}
    for stage, selected in enumerate(stages):
        stopped()
        # Every consumer in a stage really shares the same input and scaling.
        first = selected[0]
        for name in selected[1:]:
            if not torch.equal(inputs[name], inputs[first]) or not torch.equal(scaling[name], scaling[first]):
                raise RuntimeError('Invalid shared-statistics grouping')
        if number == 0 and stage == 1:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                _, current = capture(q, q_stream, meta, valid, cfg, rotary, [first])
                torch.cuda.synchronize()
            trace = EVIDENCE / 'pilot-native-profile.json'
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            kernels = sorted({e.get('name', '') for e in events if e.get('cat', '').lower() == 'kernel'})
            report['native_kernels'] = [n for n in kernels if 'k5_group_cuda' in n or 'gated_delta' in n.lower() or 'causal_conv' in n.lower()]
            if not any('k5_group_cuda' in n for n in kernels) or not any('gated_delta' in n.lower() for n in kernels):
                raise RuntimeError('Actual packed K5/accelerated DeltaNet calibration kernels absent')
            saves.put(trace, hf=True)
            del prof, events
        else:
            _, current = capture(q, q_stream, meta, valid, cfg, rotary, [first])
        xq = current[first].cuda().float()
        xf = inputs[first].cuda().float() / scaling[first]
        h, cross, chol = statistics(xf, xq)
        path = EVIDENCE / 'H' / f'block{number:02d}-stage{stage}.safetensors'
        path.parent.mkdir(exist_ok=True)
        save_file({'hessian': h.cpu(), 'reference_input_before_fold': inputs[first],
                   'quantized_input': current[first], 'input_scaling': scaling[first].cpu()}, str(path))
        saves.put(path, hf=True)
        for name in selected:
            started = time.monotonic()
            target, movement = qep_target(weights[name], h, cross, chol, alpha=0.5)
            payload, refined = refine(target, h, xq, rounds=2, steps=10)
            key = f'model.layers.{number}.{name}.weight'
            save_matrix(key, payload, layout, original, saves)
            set_value(q, name, packed(payload))
            report['linears'].append({'name': key, 'qep_relative_weight_change': movement,
                                     'seconds': time.monotonic() - started, **refined})
            print('__ANCHOR_STEP__ MATRIX', number, name, 'seconds', round(time.monotonic() - started, 2),
                  'energy', round(refined['energy_start'], 8), round(refined['energy_final'], 8), flush=True)
            del target, payload
        del h, cross, chol, xq, xf, current
    q_out, _ = capture(q, q_stream, meta, valid, cfg, rotary, [])
    for local_name, value in q.named_parameters():
        key = f'model.layers.{number}.{local_name}'
        if key not in original['retained']:
            raise RuntimeError('Unexpected remaining dense parameter: ' + key)
        previous = original['retained'][key]
        if thash(value) != previous['tensor_sha256']:
            if local_name not in ['input_layernorm.weight', 'post_attention_layernorm.weight']:
                raise RuntimeError('AWQ changed an unauthorized retained parameter: ' + key)
            path = ART / Path(previous['file']).name
            save_file({'value': value.detach().cpu().contiguous()}, str(path))
            if path.stat().st_size != (ROOT / 'artifacts' / previous['file']).stat().st_size:
                raise RuntimeError('Fold changed retained tensor storage size')
            layout['retained'][key] = {'file': path.name, 'tensor_sha256': thash(value), 'file_sha256': sha(path),
                                       'reason': 'AWQ zero-centered normalization fold; same shape/dtype'}
            saves.put(path)
    report['seconds'] = time.monotonic() - tick
    report['native_counts'] = dict(COUNTERS)
    if not COUNTERS['multirow_gemm'] or COUNTERS['dense_linear_fallback']:
        raise RuntimeError('Actual packed K5 propagation missing')
    path = EVIDENCE / f'block{number:02d}-report.json'; save_json(path, report); saves.put(path, hf=True)
    checkpoint = EVIDENCE / f'after-block{number:02d}.safetensors'
    save_file({'fp': fp_out, 'quantized': q_out}, str(checkpoint)); saves.put(checkpoint)
    save_json(ART / f'layout-after-block{number:02d}.json', layout)
    saves.put(ART / f'layout-after-block{number:02d}.json')
    del fp, q, weights, inputs, scaling
    gc.collect(); torch.cuda.empty_cache()
    print('__ANCHOR_STEP__ BLOCK_DONE', number, round(report['seconds'], 2), flush=True)
    return fp_out, q_out, report


def main(phase):
    require_remote(); stopped()
    if not os.environ.get('HF_TOKEN'):
        raise RuntimeError('Set HF_TOKEN from the existing staged token before launching; no unauthenticated dataset/model run')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract = json.loads((CODE / 'contract.json').read_text())
    for relative, digest in contract['parent_source_lock'].items():
        if sha(PARENT / relative) != digest:
            raise RuntimeError('Original source changed: ' + relative)
    for relative, digest in contract['new_source_lock'].items():
        if sha(CODE / relative) != digest:
            raise RuntimeError('Candidate source changed: ' + relative)
    if sha(EVIDENCE / 'calibration.jsonl') != contract['calibration_sha256']:
        raise RuntimeError('Calibration changed')
    if sha(ROOT / 'tk-k5/answers.jsonl') != contract['original_answers_sha256']:
        raise RuntimeError('Original K5 answers changed')
    if sha(ROOT / 'tasks-15.jsonl') != contract['tasks_sha256']:
        raise RuntimeError('Original questions changed')
    if 'L4' not in torch.cuda.get_device_name(0) or torch.cuda.mem_get_info()[0] < 18 * 1024**3:
        raise RuntimeError('Assigned L4 is not available')
    if shutil.disk_usage('/').free < 80 * 1024**3:
        raise RuntimeError('Insufficient room for candidate, exact Hessians and Drive cache')
    original = json.loads((ROOT / 'artifacts/layout.json').read_text())
    source = Path(json.loads((ROOT / 'source.json').read_text())['source'])
    for name, digest in original['source_shards'].items():
        if sha(source / name) != digest:
            raise RuntimeError('Pinned source shard changed: ' + name)
    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    cfg = AutoConfig.from_pretrained(source).text_config
    cfg._attn_implementation = 'sdpa'
    with torch.device('cuda'):
        rotary = Qwen3_5TextRotaryEmbedding(cfg)
    rows = [json.loads(s) for s in (EVIDENCE / 'calibration.jsonl').read_text().splitlines()]
    ids, meta, valid = carry.example_rows(rows, chunk=256)
    saves = Saves(contract['private_hessian_repo'])
    for path in CODE.iterdir():
        if path.is_file():
            saves.put(path, hf=True, repo_path='source/' + path.name)
    for path in [EVIDENCE / 'calibration.jsonl', EVIDENCE / 'calibration-manifest.json']:
        saves.put(path, hf=True, repo_path='data/' + path.name)
    if phase == 'pilot':
        pool = [{**r, 'ids': torch.tensor(r['ids']), 'n_tokens': r['tokens'], 'has_think': True} for r in rows]
        torch.save(pool, EVIDENCE / 'pool.pt')
        saves.put(EVIDENCE / 'pool.pt', hf=True, repo_path='data/pool.pt')
    card = {'run_id': contract['run_id'], 'phase': phase, 'status': 'running', 'started': datetime.now().astimezone().isoformat(),
            'model': MODEL, 'revision': REVISION, 'format': original['format'], 'contract': contract,
            'uptime_seconds': float(Path('/proc/uptime').read_text().split()[0]),
            'gpu': torch.cuda.get_device_name(0), 'calibration_tokens': sum(r['tokens'] for r in rows),
            'calibration_context': 'Complete examples, 256-token carried chunks, padding excluded from every statistic',
            'scope': 'Single combined candidate; same original interval question only', 'blocks': []}
    if phase == 'pilot':
        ART.mkdir(exist_ok=False)
        layout = copy.deepcopy(original)
        layout['forge_lineage'] = {'tag': contract['run_id'], 'parent_layout_sha256': sha(ROOT / 'artifacts/layout.json'),
                                  'base_id': MODEL, 'base_revision': REVISION, 'stage': 'PLACE', 'training': False}
        for row in layout['files'].values():
            row['tk'] = '../artifacts/' + row['tk']; row['fp8'] = '../artifacts/' + row['fp8']
        for row in layout['retained'].values():
            row['file'] = '../artifacts/' + row['file']
        embed_row = original['retained']['model.embed_tokens.weight']
        embed_path = ROOT / 'artifacts' / embed_row['file']
        if sha(embed_path) != embed_row['file_sha256']:
            raise RuntimeError('Source embedding changed')
        embedding = load_file(str(embed_path))['value']
        fp_stream = torch.nn.functional.embedding(ids, embedding)
        q_stream = fp_stream.clone()
        del embedding
        start, end = 0, 1
    else:
        prior = json.loads((EVIDENCE / 'pilot-run-card.json').read_text())
        if not (EVIDENCE / 'pilot-backup-receipt.json').is_file():
            raise RuntimeError('Pilot has no completed durable-save receipt')
        if prior['status'] != 'pilot_complete' or prior['contract'] != contract:
            raise RuntimeError('Cannot resume a different or incomplete pilot')
        state = load_file(str(EVIDENCE / 'after-block00.safetensors'))
        fp_stream, q_stream = state['fp'], state['quantized']
        layout = json.loads((ART / 'layout-after-block00.json').read_text())
        for row in layout['files'].values():
            if sha(ART / row['tk']) != row['sha256']['tk']:
                raise RuntimeError('Pilot checkpoint weights changed')
        start, end = 1, cfg.num_hidden_layers
    with torch.no_grad():
        for number in range(start, end):
            fp_stream, q_stream, report = convert_block(number, fp_stream, q_stream, meta, valid, cfg, rotary,
                                                       source, index, original, layout, saves)
            card['blocks'].append(report)
            save_json(EVIDENCE / f'{phase}-run-card.json', card)
    if phase == 'pilot':
        card['projected_conversion_seconds'] = card['blocks'][0]['seconds'] * cfg.num_hidden_layers
        card['projection_caveat'] = 'First DeltaNet block only; attention blocks and later propagated error may differ. Save/upload/load/answer time is additional.'
        card['status'] = 'pilot_complete'
        print('FIRST_BLOCK_PROJECTION_SECONDS', round(card['projected_conversion_seconds'], 2), flush=True)
    else:
        effective = sum((ART / row['tk']).stat().st_size for row in layout['files'].values()) + sum((ART / row['file']).stat().st_size for row in layout['retained'].values())
        if effective != original['weight_file_bytes']['tk']:
            raise RuntimeError('Saved candidate changes total weight-file size')
        save_json(ART / 'layout.json', layout); saves.put(ART / 'layout.json')
        del fp_stream, q_stream, rotary
        gc.collect(); torch.cuda.empty_cache()
        OUT.mkdir(exist_ok=False)
        model, _ = load_model(source, ART, 'tk-k5')
        tok = AutoTokenizer.from_pretrained(source)
        task = next(r for r in map(json.loads, (ROOT / 'tasks-15.jsonl').read_text().splitlines()) if r['id'] == 'HumanEval/127')
        baseline = next(r for r in map(json.loads, (ROOT / 'tk-k5/answers.jsonl').read_text().splitlines()) if r['id'] == task['id'])
        sandbox_check(); stopped()
        record = generate(model, tok, task, OUT, 'tk-k5')
        for key in ['prompt_ids', 'seed', 'sampling']:
            if record[key] != baseline[key]:
                raise RuntimeError('Original interval inputs changed: ' + key)
        rid = hashlib.sha256(task['id'].encode()).hexdigest()[:16]
        saves.put(OUT / (rid + '.json'), hf=True, repo_path='answer/raw.json')
        result = {**record, 'variant': 'AWQ-QEP-DAPTQ', 'task': task['task'], 'auto_grade': score(task, record), 'author_read_grade': None}
        (OUT / 'answers.jsonl').write_text(json.dumps(result) + '\n')
        for path in OUT.iterdir():
            if path.is_file(): saves.put(path, hf=True, repo_path='answer/' + path.name)
        card.update(status='done' if record['finish_reason'] == 'eos' else 'incomplete',
                    answer={'tokens': record['tokens'], 'seconds': record['seconds'], 'finish_reason': record['finish_reason'], 'auto_grade': result['auto_grade']},
                    weight_file_bytes=effective, matched_prompt_seed_sampling=True)
        del model
    card['finished'] = datetime.now().astimezone().isoformat()
    path = EVIDENCE / f'{phase}-run-card.json'; save_json(path, card); saves.put(path, hf=True)
    # Reuse the established Hessian bundle builder, preserving its schema and checks.
    bundle = EVIDENCE / ('bundle-' + phase)
    command = [sys.executable, str(CODE / 'hess_bundle.py'), '--hessian', str(EVIDENCE / 'H'),
               '--data', str(EVIDENCE / 'calibration.jsonl'), str(EVIDENCE / 'calibration-manifest.json'),
               '--pool', str(EVIDENCE / 'pool.pt'), '--repo', contract['private_hessian_repo'],
               '--model', MODEL + '@' + REVISION, '--teacher', 'Pinned 9B common FP16 reference; text is GSM8K gold reasoning',
               '--dataset', 'openai/gsm8k main train (MIT), revision 740312add88f781978c0658806c59bc2815b9866',
               '--run-card', str(path), '--out', str(bundle), '--no-push',
               '--note', 'AWQ-folded basis. Each H file includes exact pre-fold reference inputs, input scaling and deployed K5 inputs, so C is reconstructible. Complete gold solutions, no generated teacher answers. Scope=' + phase + '. Exact code-vault/source identities are in source/contract.json.']
    for script in ['quality9_awq_qep_daptq.py', 'quality9_k5_daptq.py', 'carry_stream.py', 'sbr_examples.py']:
        command += ['--script', str(CODE / script)]
    subprocess.run(command, check=True)
    for name in ['manifest.json', 'README.md']:
        saves.put(bundle / name, hf=True, repo_path=name)
    receipt = saves.finish()
    save_json(EVIDENCE / f'{phase}-backup-receipt.json', receipt)
    shutil.copy2(EVIDENCE / f'{phase}-backup-receipt.json', DRIVE / EVIDENCE.name / f'{phase}-backup-receipt.json')
    print('QUALITY9_COMBINED_' + phase.upper() + '_DONE', card['status'], flush=True)
    print('========== DONE ==========', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'full'], required=True)
    args = parser.parse_args()
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(60):
            print('COMBINED_HEALTH', time.time(), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        main(args.phase)
    finally:
        stop.set()
