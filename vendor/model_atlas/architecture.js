export const SUPPORTED = new Set(['qwen2','qwen2_moe','qwen2_vl','qwen2_5_vl','qwen2_5_vl_text','qwen3','qwen3_moe','qwen3_next','qwen3_5','qwen3_5_text','qwen3_5_moe','qwen3_5_moe_text','qwen3_vl','qwen3_vl_text','qwen3_vl_moe','qwen3_vl_moe_text']);
const integer=(v,name,min=1,max=1e7)=>{if(!Number.isSafeInteger(v)||v<min||v>max)throw Error(`${name} must be an integer from ${min} to ${max}.`);return v;};
export function normalize(raw,name='Imported Qwen config') {
  if(!raw||typeof raw!=='object'||Array.isArray(raw))throw Error('Import a JSON config object.');
  const t=raw.text_config??raw.thinker_config?.text_config??raw;
  const family=t.model_type??raw.model_type??'unknown';
  const depth=integer(t.num_hidden_layers??t.n_layer,'num_hidden_layers',1,512);
  const hidden=integer(t.hidden_size??t.n_embd,'hidden_size');
  const heads=integer(t.num_attention_heads??t.n_head,'num_attention_heads');
  const kv=integer(t.num_key_value_heads??heads,'num_key_value_heads');
  const headDim=integer(t.head_dim??hidden/heads,'head_dim');
  const hybrid=family.includes('qwen3_5')||family==='qwen3_next';
  const known=SUPPORTED.has(family);
  const warnings=[];
  if(!known)warnings.push(`No verified operator adapter for ${family}. Config overview is available; internal flow is intentionally withheld.`);
  let types=t.layer_types;
  if(types!==undefined){if(!Array.isArray(types)||types.length!==depth)throw Error('layer_types length must match num_hidden_layers.');}
  else if(hybrid){const interval=integer(t.full_attention_interval??4,'full_attention_interval');types=Array.from({length:depth},(_,i)=>(i+1)%interval?'linear_attention':'full_attention');warnings.push('Layer types derived from full_attention_interval (default 4 for the verified hybrid families).');}
  else types=Array(depth).fill(t.use_sliding_window?'sliding_attention':'full_attention');
  if(t.use_sliding_window&&!t.layer_types){const limit=t.max_window_layers??depth;types=types.map((v,i)=>i<limit?v:'full_attention');}
  if(types.some(x=>!['linear_attention','full_attention','sliding_attention'].includes(x)))warnings.push('Unrecognized mixer types appear as unsupported blocks.');
  const experts=t.num_experts??t.num_local_experts??0;
  const moe=experts>0;const sparse=t.decoder_sparse_step??1;const denseLayers=t.mlp_only_layers??[];
  const blocks=types.map((type,index)=>({index,type,moe:moe&&!denseLayers.includes(index)&&(index+1)%sparse===0}));
  const tied=t.tie_word_embeddings??raw.tie_word_embeddings??false;
  if(t.tie_word_embeddings!==undefined&&raw.tie_word_embeddings!==undefined&&t.tie_word_embeddings!==raw.tie_word_embeddings)warnings.push('Conflicting tie_word_embeddings: showing text_config value. Verify instantiated model.');
  const vocab=integer(t.vocab_size,'vocab_size');
  const linear=hybrid?{kh:integer(t.linear_num_key_heads??16,'linear_num_key_heads'),vh:integer(t.linear_num_value_heads??32,'linear_num_value_heads'),kd:integer(t.linear_key_head_dim??128,'linear_key_head_dim'),vd:integer(t.linear_value_head_dim??128,'linear_value_head_dim'),kernel:integer(t.linear_conv_kernel_dim??4,'linear_conv_kernel_dim')}:null;
  const qkNorm=family.startsWith('qwen3');
  return {raw,t,name,family,known,hybrid,depth,hidden,heads,kv,headDim,blocks,tied,vocab,linear,moe,experts,activeExperts:t.num_experts_per_tok??0,shared:t.shared_expert_intermediate_size??0,intermediate:t.intermediate_size??t.moe_intermediate_size,context:t.max_position_embeddings??null,vision:raw.vision_config??raw.thinker_config?.vision_config??null,audio:!!(raw.audio_config||raw.thinker_config?.audio_config),mtp:t.mtp_num_hidden_layers??0,qkNorm,gatedAttention:hybrid&&(t.attn_output_gate??true),warnings};
}
export function memoryEstimate(m,tokens=32768,kvBytes=2,stateBytes=4){
  const full=m.blocks.filter(x=>x.type!=='linear_attention').length;const linear=m.depth-full;
  const kv=2*full*tokens*m.kv*m.headDim*kvBytes;
  const state=m.linear?linear*m.linear.vh*m.linear.kd*m.linear.vd*stateBytes:0;
  const conv=m.linear?linear*(2*m.linear.kh*m.linear.kd+m.linear.vh*m.linear.vd)*m.linear.kernel*kvBytes:0;
  return {kv,state,conv,total:kv+state+conv};
}
export const fmt=n=>Number(n).toLocaleString('en-US');
export function blockGraph(m,index){
  const b=m.blocks[index];if(!b)throw Error('Block index out of range');
  const nodes=[],edges=[];const W=190,H=64;
  function add(id,title,sub,col,row,kind='compute',key=id,path='',shape='') {nodes.push({id,title,sub,x:30+col*230,y:30+row*110,w:W,h:H,kind,key,path,shape});return id;}
  const edge=(a,z,label='',kind='flow')=>edges.push({from:a,to:z,label,kind});
  const chain=(...ids)=>ids.slice(1).forEach((id,i)=>edge(ids[i],id));
  const base=`model.layers.${index}`;const h=m.hidden;const dims=(o,i)=>`${fmt(o)} × ${fmt(i)}`;
  add('input','Residual stream',`token vectors · width ${fmt(h)}`,1,0,'stream','residual',base);
  if(!m.known||!['linear_attention','full_attention','sliding_attention'].includes(b.type)){
    add('unknown','Unverified operator',m.family,1,1,'precision','unknown');edge('input','unknown');return {nodes,edges,width:960,height:270};
  }
  add('prenorm','Input RMSNorm','normalize before token mixing',1,1,'precision','norm',base+'.input_layernorm',fmt(h));edge('input','prenorm');
  let end;
  if(b.type==='linear_attention'){
    const {kh,vh,kd,vd,kernel}=m.linear;const k=kh*kd,v=vh*vd;const next=m.family==='qwen3_next';
    add('qkv','Q · K · V projection',next?'slice of fused in_proj_qkvz':'in_proj_qkv',0,2,'delta','qkv',base+'.linear_attn.'+(next?'in_proj_qkvz':'in_proj_qkv'),dims(next?2*k+2*v:2*k+v,h));
    add('a','Decay projection a',next?'slice of fused in_proj_ba':'in_proj_a',1,2,'precision','a',base+'.linear_attn.'+(next?'in_proj_ba':'in_proj_a'),dims(next?2*vh:vh,h));
    add('b','Write projection b',next?'same fused in_proj_ba':'in_proj_b',2,2,'precision','b',base+'.linear_attn.'+(next?'in_proj_ba':'in_proj_b'),dims(next?2*vh:vh,h));
    add('z','Output gate z',next?'same fused in_proj_qkvz':'in_proj_z',3,2,'delta','z',base+'.linear_attn.'+(next?'in_proj_qkvz':'in_proj_z'),dims(next?2*k+2*v:v,h));
    ['qkv','a','b','z'].forEach(id=>edge('prenorm',id));
    add('conv','Causal depthwise Conv1D',`kernel ${kernel} · then SiLU`,0,3,'precision','conv',base+'.linear_attn.conv1d',`${fmt(2*k+v)} × 1 × ${kernel}`);
    add('decay','Decay α = exp(g)','A_log · dt_bias · softplus',1,3,'precision','decay',base+'.linear_attn.{A_log,dt_bias}',`${vh} per tensor`);
    add('beta','Write strength β','sigmoid(b)',2,3,'delta','beta');
    chain('qkv','conv');chain('a','decay');chain('b','beta');
    add('l2','Split Q / K / V','L2-normalize Q & K; share heads',0,4,'delta','l2');edge('conv','l2');
    add('state','Previous memory S','persistent matrix · no token list',3,4,'state','state','runtime recurrent state',`${vh} × ${kd} × ${vd}`);
    add('delta','Gated delta update','decay → predict → correct → read',1,5,'delta','delta');
    edge('l2','delta','q, k, v');edge('decay','delta','α');edge('beta','delta','β');edge('state','delta','Sₜ₋₁','state');
    add('gatednorm','Gated RMSNorm','normalize output × SiLU(z)',1,6,'precision','gatednorm',base+'.linear_attn.norm',fmt(vd));edge('delta','gatednorm');edge('z','gatednorm','z');
    add('stateout','Next memory S','carry to the next token / chunk',3,6,'state','state');edge('delta','stateout','Sₜ','state');
    add('mixout','Output projection','out_proj',1,7,'delta','out',base+'.linear_attn.out_proj',dims(h,v));edge('gatednorm','mixout');end=8;
  }else{
    const q=m.heads*m.headDim,kv=m.kv*m.headDim;
    add('q','Query projection',m.gatedAttention?'q_proj → Q + output gate':'q_proj',0,2,'attention','q',base+'.self_attn.q_proj',dims(q*(m.gatedAttention?2:1),h));
    add('k','Key projection','k_proj',1,2,'attention','k',base+'.self_attn.k_proj',dims(kv,h));
    add('v','Value projection','v_proj',2,2,'attention','v',base+'.self_attn.v_proj',dims(kv,h));
    ['q','k','v'].forEach(id=>edge('prenorm',id));
    add('qknorm',m.qkNorm?'Q / K RMSNorm':'Q / K head reshape',m.qkNorm?'per head · before RoPE':'no learned Q/K norm in this family',0,3,m.qkNorm?'precision':'attention',m.qkNorm?'qknorm':'heads',base+'.self_attn',m.qkNorm?`${m.headDim} each`:'');edge('q','qknorm');edge('k','qknorm');
    add('rope','Rotary position encoding','RoPE on Q and K',0,4,'attention','rope');edge('qknorm','rope');
    add('kv','KV cache','append K / V across tokens',2,4,'state','kv','runtime attention cache');edge('rope','kv','K');edge('v','kv','V');
    add('softmax','Causal grouped attention',`${m.heads} Q heads · ${m.kv} KV heads`,1,5,'attention',b.type==='sliding_attention'?'sliding':'attention');edge('rope','softmax','Q');edge('kv','softmax','K, V');
    if(m.gatedAttention){add('attngate','Attention output gate','multiply by sigmoid(gate)',1,6,'attention','attngate');edge('q','attngate','gate');edge('softmax','attngate');}
    add('mixout','Output projection','o_proj',1,7,'attention','out',base+'.self_attn.o_proj',dims(h,q));edge(m.gatedAttention?'attngate':'softmax','mixout');end=8;
  }
  add('add1','Residual addition','original x + mixer output',1,end,'stream','add');edge('mixout','add1');edge('input','add1','skip / identity','residual');
  add('postnorm','Post-attention RMSNorm','prepare for the feed-forward path',1,end+1,'precision','norm',base+'.post_attention_layernorm',fmt(h));edge('add1','postnorm');
  if(b.moe){
    add('router','Expert router',`${m.activeExperts} selected of ${m.experts}`,0,end+2,'precision','router',base+'.mlp.gate',dims(m.experts,h));edge('postnorm','router');
    add('experts','Routed expert MLPs','each: gate / up → SiLU × → down',0,end+3,'mlp','experts',base+'.mlp.experts',`${m.experts} experts · width ${fmt(m.t.moe_intermediate_size??m.intermediate??0)}`);edge('router','experts','top-k + weights');edge('postnorm','experts','token vectors');
    add('combine','Weighted expert sum','combine selected outputs',1,end+4,'mlp','combine');edge('experts','combine');
    if(m.shared){add('shared','Shared expert + gate','always evaluated · sigmoid gate',2,end+3,'mlp','shared',base+'.mlp.shared_expert',`width ${fmt(m.shared)}`);edge('postnorm','shared');edge('shared','combine');}
  }else{
    add('mlpgate','MLP gate projection','gate_proj → SiLU',0,end+2,'mlp','mlpgate',base+'.mlp.gate_proj',dims(m.intermediate,h));
    add('up','MLP up projection','up_proj',2,end+2,'mlp','up',base+'.mlp.up_proj',dims(m.intermediate,h));edge('postnorm','mlpgate');edge('postnorm','up');
    add('product','Elementwise product','SiLU(gate) ⊙ up',1,end+3,'mlp','product');edge('mlpgate','product');edge('up','product');
    add('combine','MLP down projection','down_proj → residual width',1,end+4,'mlp','down',base+'.mlp.down_proj',dims(h,m.intermediate));edge('product','combine');
  }
  add('add2','Residual addition','feed-forward output + skip',1,end+5,'stream','add');edge('combine','add2');edge('add1','add2','skip / identity','residual');
  add('output','To the next block',`same token count · width ${fmt(h)}`,1,end+6,'stream','residual');edge('add2','output');
  return {nodes,edges,width:960,height:30+(end+7)*110};
}
// A 2×2 teaching recurrence, using the same decay-before-correction ordering as the reference implementation.
export function deltaStep(S,k,v,q,alpha,beta){
  const old=S.map(row=>row.map(x=>x*alpha));
  const prediction=v.map((_,j)=>k.reduce((sum,x,i)=>sum+x*old[i][j],0));
  const correction=v.map((x,j)=>beta*(x-prediction[j]));
  const next=old.map((row,i)=>row.map((x,j)=>x+k[i]*correction[j]));
  const output=q.map((_,j)=>q.reduce((sum,x,i)=>sum+x*next[i][j],0)/Math.sqrt(q.length));
  return {next,prediction,correction,output};
}

// Config-derived protection policy, not a checkpoint inventory or sensitivity measurement.
export function protectionPlan(m,{vocabulary=true,gates=true}={}) {
 const entries=[];
 const add=(path,key,block=null,node=null,note='')=>{if(!entries.some(e=>e.path===path))entries.push({path,key,block,node,note,action:'Keep original checkpoint precision; exclude from low-bit weight quantization'});};
 const unsupported=m.blocks.filter(b=>!['linear_attention','full_attention','sliding_attention'].includes(b.type)||(b.type==='linear_attention'&&!m.linear));
 if(!m.known||unsupported.length)return {entries,complete:false,notes:['No complete protection template for this architecture. Inspect the instantiated model before quantizing.']};
 if(vocabulary){add('model.embed_tokens','embeddings',null,null,m.tied?'Shares weights with lm_head; count storage once.':'');add('lm_head','head',null,null,m.tied?'Alias of token embeddings; protect both module names.':'');}
 add('model.norm','finalnorm');
 for(const b of m.blocks){
  const base=`model.layers.${b.index}`;
  for(const n of blockGraph(m,b.index).nodes){
   if(n.kind!=='precision'||!n.path||(!gates&&['a','b'].includes(n.key)))continue;
   if(n.key==='decay'){add(base+'.linear_attn.A_log',n.key,b.index,n.id);add(base+'.linear_attn.dt_bias',n.key,b.index,n.id);}
   else if(n.key==='qknorm'){add(base+'.self_attn.q_norm',n.key,b.index,n.id);add(base+'.self_attn.k_norm',n.key,b.index,n.id);}
   else add(n.path,n.key,b.index,n.id,m.family==='qwen3_next'&&['a','b'].includes(n.key)?'Fused a/b projection: protect the entire in_proj_ba module.':'');
  }
  if(b.moe&&m.shared)add(base+'.mlp.shared_expert_gate','router',b.index,'shared','Protect the shared-expert routing gate separately from its MLP.');
 }
 const notes=['Paths are relative to the text-model wrapper. Module entries include all their parameters; A_log and dt_bias are direct parameters. Resolve names against named_modules/named_parameters before applying to a quantizer.','This policy does not set runtime activation, cache, accumulation, or recurrent-state precision. Keep the reference runtime defaults until separately validated.'];
 if(m.vision||m.audio||m.mtp)notes.push('Vision, audio, and MTP internals are not enumerated. Leave these unmodeled components at their original precision until separately reviewed.');
 return {entries,complete:!(m.vision||m.audio||m.mtp),notes};
}

// Module groups, not a safetensors inventory; fused tensors appear once.
export function precisionInventory(m){
 if(!m.known)return [];
 const rows=new Map();const add=(path,key,block=null)=>{if(!rows.has(path))rows.set(path,{path,key,block});};
 add('model.embed_tokens','embeddings');add('model.norm','finalnorm');add('lm_head','head');
 for(const b of m.blocks){if(b.type==='linear_attention'&&!m.linear)continue;
 for(const n of blockGraph(m,b.index).nodes){
 if(!n.path||n.kind==='state'||n.key==='residual'||n.key==='unknown')continue;
 if(n.key==='decay'){for(const name of ['A_log','dt_bias'])add(`model.layers.${b.index}.linear_attn.${name}`,n.key,b.index);}
 else if(n.key==='qknorm'){for(const name of ['q_norm','k_norm'])add(`model.layers.${b.index}.self_attn.${name}`,n.key,b.index);}
 else if(n.key!=='heads')add(n.path,n.key,b.index);
 }
 if(b.moe&&m.shared)add(`model.layers.${b.index}.mlp.shared_expert_gate`,'router',b.index);
 }
 return [...rows.values()];
}

export function matchingBlockPaths(m,path,allTypes=false){
 const match=/^model\.layers\.(\d+)(.*)$/.exec(path);if(!match)return [path];
 const source=m.blocks[Number(match[1])];if(!source)return [path];
 const suffix=match[2],inventory=new Set(precisionInventory(m).map(e=>e.path));
 return m.blocks.filter(b=>(allTypes||b.type===source.type)).map(b=>`model.layers.${b.index}${suffix}`).filter(p=>!suffix||inventory.has(p));
}

export function tkIntegerFormat(value){
 const factor=/^NQ-(binary|ternary)-r([1-9][0-9]{0,5})$/.exec(value);
 if(factor)return {name:factor[1]==='binary'?'NanoQuant binary':'NanoQuant-style ternary',method:'factorization',kind:factor[1],rank:Number(factor[2]),scale_bytes:2,codec_status:factor[1]==='binary'?'Official binary ADMM initialization; adapted reconstruction':'Experimental ternary extension',packing:factor[1]==='binary'?'one bit per factor entry; each factor byte-aligned':'five trits per byte; each factor row byte-aligned'};
 const intermediate=/^TKinteger-k(1[0-6]|[1-9])-g([1-9][0-9]*)-s12$/.exec(value);
 if(intermediate){const k=Number(intermediate[1]),g=Number(intermediate[2]);if(!Number.isSafeInteger(g)||g>65536)return null;return {name:k===1&&g===20?'TK 1 2.2':'TKinteger S12',base:3,k,levels:3**k,group_size:g,scale_bits:12,integrated_scales:true,payload_bits_per_value:Math.ceil(g*k/5)*8/g,total_bits_per_value:(Math.ceil(g*k/5)*8+12)/g,balanced_integer_min:-(3**k-1)/2,balanced_integer_max:(3**k-1)/2,packing:'row groups padded to g; five trits per byte; shared 12-bit scale stream',codec_status:'Experimental s12-log2-v1; uniform k, no mixed k weights',scale_mapping:'0=zero; codes 1..4095 map to exp2(-32+(code-1)*48/4094)'};}

 if(value==='TKinteger-20_44-k1')return {name:'TKinteger 20/44',base:3,k:1,trits_per_value:1,values_per_block:20,bits_per_block:44,levels:3,balanced_integer_min:-1,balanced_integer_max:1,payload_bits_per_value:2.2,packing:'20 ternary values per 44-bit block; blocks bit-packed consecutively within each tensor',payload_bytes_formula:'ceil(ceil(value_count / 20) * 44 / 8) per tensor',excluded_overhead:['scales','anchors','headers','alignment','partial-block and final-byte padding'],codec_status:'user-defined packing target only; no encoder or quantization executed',runtime_compute_dtype:'unspecified'};

 const scaled=/^TKinteger-k([1-9]|1[0-6])-g([1-9][0-9]*)-s(16|32)$/.exec(value);
 if(scaled){const k=Number(scaled[1]),g=Number(scaled[2]),bits=Number(scaled[3]);if(!Number.isSafeInteger(g)||g>65536)return null;return {...tkIntegerFormat(`TKinteger-k${k}`),group_size:g,scale_bytes:bits/8,scale_bits:bits,total_bits_per_value:8*k/5+bits/g};}
 const match=/^TKinteger-k([1-9]|1[0-6])$/.exec(value);if(!match)return null;
 const k=Number(match[1]),levels=3**k;
 return {name:'TKinteger',base:3,k,trits_per_value:k,trits_per_byte:5,levels,balanced_integer_min:-(levels-1)/2,balanced_integer_max:(levels-1)/2,payload_bits_per_value:8*k/5,scale_bytes:k<=5?2:4,packing:'independent matrix rows; five base-3 digits per byte',payload_bytes_formula:'rows * ceil(columns * k / 5)',excluded_overhead:['scales','zero points or anchors','headers','alignment','final partial-byte padding'],codec_status:'base3_pack.py storage accounting; Atlas does not execute quantization',runtime_compute_dtype:'unspecified; packed storage does not determine arithmetic dtype'};
}

// Counts of represented text weight tensors, excluding unspecified biases and frontends.
export function weightTensorInventory(m){
 const graphCache=new Map();
 return precisionInventory(m).map(e=>{
 let perTensor=0,copies=1,matrices=[];
 if(['embeddings','head'].includes(e.key))perTensor=m.vocab*m.hidden;
 else if(['norm','finalnorm'].includes(e.key))perTensor=m.hidden;
 else if(e.key==='qknorm')perTensor=m.headDim;
 else if(e.key==='decay')perTensor=m.linear.vh;
 else if(e.key==='gatednorm')perTensor=m.linear.vd;
 else if(e.key==='experts'){perTensor=m.hidden*(m.t.moe_intermediate_size??m.intermediate);copies=3*m.experts;}
 else if(e.key==='shared'){perTensor=m.hidden*m.shared;copies=3;}
 else if(e.path.endsWith('shared_expert_gate'))perTensor=m.hidden;
 else{if(!graphCache.has(e.block))graphCache.set(e.block,blockGraph(m,e.block));const n=graphCache.get(e.block).nodes.find(n=>n.path===e.path);const shape=(n?.shape??'').replaceAll(',','');if(/^\d+( × \d+)*$/.test(shape))perTensor=shape.split(' × ').reduce((a,v)=>a*Number(v),1);}
 if(['embeddings','head'].includes(e.key))matrices=[{rows:m.vocab,columns:m.hidden,copies:1}];
 else if(['experts','shared'].includes(e.key)){const width=e.key==='experts'?(m.t.moe_intermediate_size??m.intermediate):m.shared,n=e.key==='experts'?m.experts:1;matrices=[{rows:width,columns:m.hidden,copies:2*n},{rows:m.hidden,columns:width,copies:n}];}
 else {const node=graphCache.get(e.block)?.nodes.find(n=>n.path===e.path),dims=(node?.shape??'').replaceAll(',','').split(' × ').map(Number);if(dims.length===2&&dims.every(n=>Number.isSafeInteger(n)&&n>0))matrices=[{rows:dims[0],columns:dims[1],copies}];}
 return {...e,perTensor,copies,matrices,parameters:perTensor*copies,alias:m.tied&&e.key==='head'};
 });
}
export function estimateWeightStorage(m,choices,{originalBytes=2,groupSize=128,scaleBytes=-1,extraBytes=0}={}){
 if(![2,4].includes(originalBytes)||!Number.isSafeInteger(groupSize)||groupSize<1||!Number.isFinite(scaleBytes)||(scaleBytes<0&&scaleBytes!==-1)||!Number.isFinite(extraBytes)||extraBytes<0)throw Error('Invalid size assumptions.');
 let parameters=0,payload=0,scales=0,protectedBytes=0;const unknown=[],components=[],warnings=[];
 if(m.tied&&choices.lm_head?.precision&&choices['model.embed_tokens']?.precision&&choices.lm_head.precision!==choices['model.embed_tokens'].precision)throw Error('Tied embeddings and head must use the same precision.');
 for(const row of weightTensorInventory(m)){
 if(row.alias)continue;if(!Number.isSafeInteger(row.parameters)||row.parameters<=0){unknown.push(row.path);continue;}parameters+=row.parameters;
 const choice=choices[row.path]??(m.tied&&row.key==='embeddings'?choices.lm_head:undefined)??{},precision=choice.precision??'original',format=tkIntegerFormat(precision);
 const rowGroup=choice.group_size??format?.group_size??groupSize;
 if(!Number.isSafeInteger(rowGroup)||rowGroup<1||rowGroup>65536)throw Error('Invalid layer group size: '+row.path);
 if(format?.group_size&&choice.group_size!==undefined&&choice.group_size!==format.group_size)throw Error('Conflicting format and layer group sizes: '+row.path);
 let bytes,quantized=false,exactTK=format&&!format.bits_per_block&&!format.integrated_scales&&format.method!=='factorization';
 if(format?.method==='factorization'){
 const r=format.rank;bytes=0;
 if(!row.matrices.length){bytes=row.parameters*originalBytes;warnings.push(row.path+': factorization requires a matrix; counted at original precision until an adapter exists.');}
 for(const sh of row.matrices){if(r>Math.min(sh.rows,sh.columns))warnings.push(row.path+': rank exceeds matrix dimensions; runner will reject.');bytes+=(format.kind==='binary'?Math.ceil(sh.rows*r/8)+Math.ceil(r*sh.columns/8):sh.rows*Math.ceil(r/5)+r*Math.ceil(sh.columns/5))*sh.copies;}
 }else if(format?.integrated_scales){
 const shapes=row.matrices.length?row.matrices:[{rows:1,columns:row.perTensor,copies:row.copies}];
 bytes=shapes.reduce((n,sh)=>n+sh.rows*Math.ceil(sh.columns/format.group_size)*Math.ceil(format.group_size*format.k/5)*sh.copies,0);
 if(!row.matrices.length)warnings.push(row.path+': S12 quantization requires a matrix adapter.');
 }else if(exactTK){const shapes=row.matrices.length?row.matrices:[{rows:1,columns:row.perTensor,copies:row.copies}];bytes=shapes.reduce((n,s)=>n+s.rows*Math.ceil(s.columns*format.k/5)*s.copies,0);quantized=true;
 if(!row.matrices.length)warnings.push(row.path+': non-matrix target; hypothetical flattened packing, requires a codec adapter.');
 if(shapes.some(s=>s.columns%rowGroup!==0))warnings.push(row.path+': input width is not divisible by scale group; uploaded quantizer rejects this. Scale estimate allows a partial group.');
 }else if(format){bytes=tkPayloadBytes(row.perTensor,format)*row.copies;quantized=true;}
 else if(precision==='ternary'){bytes=Math.ceil(row.perTensor/5)*row.copies;quantized=true;}
 else if(['INT4','INT8'].includes(precision)){bytes=Math.ceil(row.perTensor*(precision==='INT4'?.5:1))*row.copies;quantized=true;}
 else bytes=row.parameters*(precision==='FP32'?4:precision==='original'?originalBytes:2);
 const bytesPerScale=scaleBytes===-1?(exactTK?format.scale_bytes:2):scaleBytes;
 const groups=exactTK&&row.matrices.length?row.matrices.reduce((n,s)=>n+s.rows*Math.ceil(s.columns/rowGroup)*s.copies,0):Math.ceil(row.perTensor/rowGroup)*row.copies;
 const scaleCost=format?.method==='factorization'?row.matrices.reduce((n,sh)=>n+(sh.rows+sh.columns)*2*sh.copies,0):format?.integrated_scales?(row.matrices.length?row.matrices:[{rows:1,columns:row.perTensor,copies:row.copies}]).reduce((n,sh)=>n+Math.ceil(sh.rows*Math.ceil(sh.columns/format.group_size)*12/8)*sh.copies,0):quantized?groups*bytesPerScale:0;
 payload+=bytes;scales+=scaleCost;if(choice.protection?.length)protectedBytes+=bytes;
 components.push({path:row.path,key:row.key,parameters:row.parameters,precision,group_size:rowGroup,payload:bytes,scales:scaleCost,total:bytes+scaleCost});
 }
 const sum=items=>items.reduce((a,e)=>({parameters:a.parameters+e.parameters,payload:a.payload+e.payload,scales:a.scales+e.scales,total:a.total+e.total}),{parameters:0,payload:0,scales:0,total:0});
 const breakdown={body:sum(components.filter(e=>!['embeddings','head'].includes(e.key))),embeddings:sum(components.filter(e=>e.key==='embeddings')),head:sum(components.filter(e=>e.key==='head')),tied:m.tied};
 return {parameters,payload,scales,protectedBytes,extraBytes,total:payload+scales+extraBytes,components,breakdown,unknown,warnings,scope:'Represented text weights plus user-supplied extras. Excludes unspecified biases, vision/audio/MTP tensors, alignment, and container headers unless supplied as extras. Runtime state is not stored model weight size.'};
}

// Validate the entire import before the UI changes any model or saved settings.
export function validatePlanImport(data,m,inventory,allowedPrecisions){
 const obj=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 if(!obj(data)||!Number.isInteger(data.schema_version)||data.schema_version<1||data.schema_version>5)throw Error('Expected a Qwen Atlas plan export (schema 1–5).');
 const byPath=new Map(inventory.map(e=>[e.path,e]));
 const pathCheck=p=>{if(typeof p!=='string'||!byPath.has(p))throw Error(`Unknown layer in this model: ${String(p).slice(0,150)}`);};
 const checkMap=(map,check)=>{if(!obj(map))throw Error('Invalid saved settings map.');for(const [p,v] of Object.entries(map)){pathCheck(p);check(v);}return {...map};};
 const checkPrecision=v=>{if(!allowedPrecisions.includes(v)&&!tkIntegerFormat(v))throw Error(`Unsupported precision: ${String(v).slice(0,80)}`);};
 const checkBool=v=>{if(typeof v!=='boolean')throw Error('Protection options must be true or false.');};
 const checkGroup=v=>{if(!Number.isSafeInteger(v)||v<1||v>65536)throw Error('Group size must be an integer from 1 to 65536.');};
 const result={};
 if(data.quant_job!==undefined){const j=data.quant_job;if(!obj(j)||typeof j.model!=='string'||!j.model.trim()||typeof j.revision!=='string'||!['ultramega','multicorpus_xl','reasoning_xl','ultramega_reasoning_88'].includes(j.corpus))throw Error('Invalid quantization job.');for(const key of ['calibration_tokens','max_document_tokens','chunk_size','admm_steps','block_steps','kl_steps'])if(!Number.isSafeInteger(j[key])||j[key]<(['block_steps','kl_steps'].includes(key)?0:1)||j[key]>100000000)throw Error('Invalid quantization job '+key);if(j.chunk_size>=j.max_document_tokens)throw Error('Document length must exceed chunk size.');if(j.compensation_steps!==undefined&&(!Number.isSafeInteger(j.compensation_steps)||j.compensation_steps<0||j.compensation_steps>10000))throw Error('Invalid compensation steps');if(j.hessian_dir!==undefined&&typeof j.hessian_dir!=='string')throw Error('Invalid Hessian path');if(j.reuse_hessian!==undefined&&typeof j.reuse_hessian!=='boolean')throw Error('Invalid Hessian reuse flag');result.job={...j};}

 if(data.precision!==undefined){
  if(!Array.isArray(data.precision))throw Error('Precision entries must be a list.');
  result.precisions={};result.optOut={};result.groups={};const inferred=[];
  for(const e of data.precision){if(!obj(e))throw Error('Invalid precision entry.');pathCheck(e.path);checkPrecision(e.precision);if(Object.hasOwn(result.precisions,e.path))throw Error('Duplicate precision entry: '+e.path);result.precisions[e.path]=e.precision;if(e.group_size!==undefined){checkGroup(e.group_size);if(e.path.includes('@flow.'))throw Error('Runtime operations do not have weight group scales.');const f=tkIntegerFormat(e.precision);if(f?.group_size&&f.group_size!==e.group_size)throw Error('Conflicting format and layer group sizes.');result.groups[e.path]=e.group_size;}
   if(e.protection!==undefined&&(!Array.isArray(e.protection)||e.protection.some(x=>!['Automatic','Custom'].includes(x))))throw Error('Invalid protection flags.');
   result.optOut[e.path]=!e.protection?.includes('Automatic');if(e.protection?.includes('Custom'))inferred.push({path:e.path,scope:e.path.includes('@flow.')?'runtime':'component',label:byPath.get(e.path).key});
  }result.custom=inferred;
 }
 if(data.custom_protection!==undefined){
  if(!Array.isArray(data.custom_protection))throw Error('Custom protection must be a list.');const expanded=[];
  for(const e of data.custom_protection){if(!obj(e)||typeof e.path!=='string'||!['component','runtime','block'].includes(e.scope))throw Error('Invalid custom protection entry.');
   if(e.scope==='block'){const match=/^model\.layers\.(\d+)$/.exec(e.path);if(!match||!m.blocks[Number(match[1])])throw Error('Unknown protected block.');for(const row of inventory.filter(row=>row.kind==='weight'&&row.path.startsWith(e.path+'.')))expanded.push({path:row.path,scope:'component',label:row.key});}
   else{pathCheck(e.path);expanded.push({path:e.path,scope:e.path.includes('@flow.')?'runtime':'component',label:typeof e.label==='string'?e.label.slice(0,500):byPath.get(e.path).key});}
  }result.custom=[...new Map(expanded.map(e=>[e.path,e])).values()];
 }
 if(data.automatic_protection_opt_out!==undefined)result.optOut={...(result.optOut??{}),...checkMap(data.automatic_protection_opt_out,checkBool)};
 let autoEntries=data.automatic_protection?.entries??data.entries;
 if(autoEntries!==undefined){if(!Array.isArray(autoEntries))throw Error('Invalid automatic protection entries.');for(const e of autoEntries)pathCheck(e.path);
  result.options={vocabulary:autoEntries.some(e=>['model.embed_tokens','lm_head'].includes(e.path)),delta_gates:autoEntries.some(e=>/\.in_proj_(a|b|ba)$/.test(e.path))};
  const included=new Set(autoEntries.map(e=>e.path));for(const e of protectionPlan(m).entries){if(!result.optOut)result.optOut={};if(!(e.path in result.optOut))result.optOut[e.path]=!included.has(e.path);}
 }
 if(data.options!==undefined){if(!obj(data.options))throw Error('Invalid policy options.');for(const key of ['vocabulary','delta_gates'])if(data.options[key]!==undefined){checkBool(data.options[key]);result.options={...result.options,[key]:data.options[key]};}}
 if(data.size_assumptions!==undefined){const s=data.size_assumptions;if(!obj(s))throw Error('Invalid size assumptions.');estimateWeightStorage(m,{}, {originalBytes:s.original_bytes,groupSize:s.scale_group_size,scaleBytes:s.scale_bytes_per_group,extraBytes:s.extras_MB*1e6});if(![s.original_bytes,s.scale_group_size,s.scale_bytes_per_group,s.extras_MB].every(Number.isFinite))throw Error('Missing size assumptions.');result.size=s;}
 if(data.editor_state!==undefined){const e=data.editor_state;if(!obj(e))throw Error('Invalid editor state.');if(e.group_sizes!==undefined)result.groups=checkMap(e.group_sizes,checkGroup);if(e.precision_choices!==undefined)result.precisions=checkMap(e.precision_choices,checkPrecision);if(e.automatic_opt_out!==undefined)result.optOut=checkMap(e.automatic_opt_out,checkBool);if(e.repeat_type!==undefined){checkBool(e.repeat_type);result.repeat=e.repeat_type;}}
 if(!['precision','custom_protection','entries','automatic_protection'].some(k=>Object.hasOwn(data,k)))throw Error('This JSON does not contain an exported plan. Use Import config.json for architecture configs.');
 if(m.tied&&result.precisions){const a=result.precisions['lm_head'],b=result.precisions['model.embed_tokens'];if(a!==undefined&&b!==undefined&&a!==b)throw Error('Tied embeddings and head have conflicting precisions.');if(a!==undefined||b!==undefined){result.precisions.lm_head=a??b;result.precisions['model.embed_tokens']=a??b;}}
 if(m.tied&&result.groups){const a=result.groups.lm_head,b=result.groups['model.embed_tokens'];if(a!==undefined&&b!==undefined&&a!==b)throw Error('Tied embeddings and head have conflicting group sizes.');if(a!==undefined||b!==undefined){result.groups.lm_head=a??b;result.groups['model.embed_tokens']=a??b;}}
 return result;
}

export function tkPayloadBytes(valueCount,format){
 if(!Number.isSafeInteger(valueCount)||valueCount<0)throw Error('Value count must be a nonnegative integer.');
 return format.bits_per_block?Math.ceil(Math.ceil(valueCount/format.values_per_block)*format.bits_per_block/8):Math.ceil(valueCount*format.k/5);
}
