// Native UI adapter. Architecture, validation and storage math stay upstream.
import fs from 'node:fs';
import {presets} from './vendor/model_atlas/data/presets.js';
import {info} from './vendor/model_atlas/knowledge.js';
import {normalize, blockGraph, precisionInventory, protectionPlan,
  validatePlanImport, tkIntegerFormat, estimateWeightStorage} from './vendor/model_atlas/architecture.js';

const request = JSON.parse(fs.readFileSync(0, 'utf8'));
function inventory(m) {
  const rows = precisionInventory(m).map(e => ({...e, kind:'weight'}));
  for (const b of m.blocks) {
    if (b.type === 'linear_attention' && !m.linear) continue;
    for (const n of blockGraph(m, b.index).nodes) {
      if (!n.path || n.kind === 'state' || ['residual','heads','unknown'].includes(n.key))
        rows.push({path:`model.layers.${b.index}.@flow.${n.id}`, key:n.key, block:b.index,
          kind:n.kind === 'state' ? 'runtime_state' : 'runtime_operation'});
    }
  }
  for (const [key, exists] of [['vision',m.vision],['mtp',m.mtp]])
    if (exists) rows.push({path:`@flow.${key}`,key,block:null,kind:'unexpanded_subsystem'});
  return rows;
}
function describe() {
  const input = request.plan;
  const named = presets.find(p => p.id === (input?.model ?? request.model));
  const raw = input?.config ?? named?.config;
  if (!raw) throw Error('Choose a bundled model, or import a plan containing its config.');
  const m = normalize(raw, input?.model ?? named.id);
  if (input?.family && input.family !== m.family) throw Error('Plan family conflicts with its config.');
  const inv = inventory(m);
  const restored = input ? validatePlanImport(input, m, inv,
    ['original','BF16','FP16','FP32','INT8','INT4','ternary']) : {};
  const options = restored.options ?? {vocabulary:true,delta_gates:true};
  const automatic = protectionPlan(m, {vocabulary:options.vocabulary !== false, gates:options.delta_gates !== false});
  const custom = restored.custom ?? [];
  const optOut = restored.optOut ?? {};
  const selected = restored.precisions ?? {};
  const groups = restored.groups ?? {};
  const size = restored.size ?? {original_bytes:2,scale_group_size:128,scale_bytes_per_group:-1,extras_MB:0};
  const matches = (parent,path) => parent === path || path.startsWith(parent + '.');
  const precision = inv.map(e => {
    const aliases = m.tied && ['lm_head','model.embed_tokens'].includes(e.path)
      ? ['lm_head','model.embed_tokens'] : [e.path];
    const protection = [];
    if (aliases.some(p => !optOut[p] && automatic.entries.some(a => matches(a.path,p)))) protection.push('Automatic');
    if (aliases.some(p => custom.some(a => matches(a.path,p)))) protection.push('Custom');
    const target = protection.length ? 'original' : selected[e.path] ?? 'original';
    const format = tkIntegerFormat(target);
    return {...e, precision:target, protection, format,
      ...(e.kind === 'weight' ? {group_size:format?.group_size ?? groups[e.path] ?? size.scale_group_size} : {})};
  });
  const plan = {schema_version:5,model:m.name,family:m.family,config:raw,tied_embeddings:m.tied,
    basis:'Model Atlas format plan; no model execution',precision,size_assumptions:size,
    options,automatic_protection:automatic,custom_protection:custom,automatic_protection_opt_out:optOut,
    ...(input?.quant_job ? {quant_job:input.quant_job} : {})};
  const estimate = estimateWeightStorage(m, Object.fromEntries(precision.filter(e=>e.kind==='weight').map(e=>[e.path,e])),
    {originalBytes:size.original_bytes,groupSize:size.scale_group_size,scaleBytes:size.scale_bytes_per_group,extraBytes:size.extras_MB*1_000_000});
  return {plan, estimate, model:{name:m.name,depth:m.depth,family:m.family,tied:m.tied,known:m.known,
    hidden:m.hidden,vocab:m.vocab,blocks:m.blocks,warnings:m.warnings},
    graph:blockGraph(m, request.block ?? 0),info};
}
try {
  console.log(JSON.stringify(request.action === 'presets' ? presets.map(({id,revision,url})=>({id,revision,url})) : describe()));
} catch (error) {
  console.error(error.message);
  process.exitCode=1;
}
