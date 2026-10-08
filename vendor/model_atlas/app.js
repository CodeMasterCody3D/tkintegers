import {launchQuantization} from './github-actions.js';
import {presets} from './data/presets.js';
import {normalize,blockGraph,memoryEstimate,deltaStep,fmt,protectionPlan,precisionInventory,matchingBlockPaths,tkIntegerFormat,estimateWeightStorage,validatePlanImport} from './architecture.js';
import {info,sources,lessons} from './knowledge.js';
const $=s=>document.querySelector(s);const $$=s=>[...document.querySelectorAll(s)];
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let activePreset=presets[0],model,index=0,filter='all',scale=1,graph,pinned=null,traceTimer=null,view='explore';
let precisionPage=0;
let customSelections=[],precisionChoices={},groupChoices={},automaticOptOut={},planningKey='';
const params=new URLSearchParams(location.hash.slice(1));
if(params.get('model'))activePreset=presets.find(p=>p.id===params.get('model'))??activePreset;
$('#modelSelect').innerHTML=presets.map(p=>`<option value="${escape(p.id)}">${escape(p.id.replace('Qwen/',''))}</option>`).join('');
$('#modelSelect').value=activePreset.id;
function showNotice(text){$('#notice').textContent=text;$('#notice').hidden=!text;}
function sourceHTML(refs){return refs.map(key=>{const s=sources[key];return `<a href="${s.url}" target="_blank" rel="noopener noreferrer">${escape(s.title)} ↗</a>`;}).join('');}
function modelLoad(raw,name){model=normalize(raw,name);loadPlanning(raw);index=Math.min(Math.max(0,Number(params.get('block'))||0),model.depth-1);pinned=null;filter='all';$$('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter==='all'));renderModel();renderBlock();renderMemoryEstimate();renderProtection();renderPlanning();}
function renderModel(){
 const delta=model.blocks.filter(b=>b.type==='linear_attention').length;
 const metrics=[[model.depth,'DECODER BLOCKS'],[`${delta} / ${model.depth-delta}`,'DELTANET / ATTENTION'],[fmt(model.hidden),'HIDDEN WIDTH'],[`${model.heads} / ${model.kv}`,'QUERY / KV HEADS'],[model.moe?`${model.activeExperts} / ${model.experts}`:'Dense',model.moe?'ACTIVE / TOTAL EXPERTS':'FEED-FORWARD'],[model.tied?'Tied':'Separate','EMBEDDING & HEAD']];
 $('#metrics').innerHTML=metrics.map(([value,label])=>`<div class="metric"><strong>${escape(value)}</strong><span>${label}</span></div>`).join('');
 $('#stackCount').textContent=`${model.depth} blocks`;
 $('#blockGrid').innerHTML=model.blocks.map(b=>`<button class="block-button ${b.type==='linear_attention'?'delta':'attention'}" data-block="${b.index}" aria-label="Block ${b.index+1}: ${b.type.replaceAll('_',' ')}${b.moe?', mixture of experts':''}" title="Block ${b.index+1} · ${b.type.replaceAll('_',' ')}${b.moe?' · MoE':''}">${String(b.index+1).padStart(2,'0')}${b.moe?'<small>E</small>':''}</button>`).join('');
 $$('[data-block]').forEach(button=>{button.onclick=()=>{index=Number(button.dataset.block);renderBlock();};const b=model.blocks[Number(button.dataset.block)];const message=`${b.type==='linear_attention'?'Gated DeltaNet compresses token history into recurrent matrix memory.':'Causal attention reads the token KV cache.'} Followed by ${b.moe?'routed experts':'a dense SwiGLU MLP'} and two residual additions.`;button.onpointerenter=e=>showTip(e,`Block ${b.index+1}`,message);button.onpointerleave=hideTip;button.onfocus=()=>showTipAt(button,`Block ${b.index+1}`,message);button.onblur=hideTip;});
 $('#visionButton').hidden=!model.vision;$('#mtpButton').hidden=!model.mtp;
 showNotice(model.warnings.join(' '));
}
function renderBlock(){
 stopTrace();hideTip();pinned=null;const b=model.blocks[index];
 $$('[data-block]').forEach(el=>{const i=Number(el.dataset.block);el.classList.toggle('selected',i===index);el.setAttribute('aria-pressed',String(i===index));el.classList.toggle('filtered',filter!=='all'&&model.blocks[i].type!==filter);});
 $('#blockTitle').textContent=`Block ${String(index+1).padStart(2,'0')} / ${b.type==='linear_attention'?'Gated DeltaNet':b.type==='sliding_attention'?'Sliding attention':'Full attention'}`;
 $('#blockSubtitle').textContent=`${b.moe?'Routed expert':'Dense SwiGLU'} feed-forward · index ${index} in code · ${fmt(model.hidden)}-wide residual stream`;
 $('#prevBlock').disabled=index===0;$('#nextBlock').disabled=index===model.depth-1;
 graph=blockGraph(model,index);drawGraph();$('#viewport').scrollTop=0;fit();inspect(b.type==='linear_attention'?'delta':'attention');
 updateBlockProtection();
 if(activePreset)history.replaceState(null,'',`#model=${encodeURIComponent(activePreset.id)}&block=${index}`);
}
function drawGraph(){
 const svg=$('#flowSvg');svg.setAttribute('viewBox',`0 0 ${graph.width} ${graph.height}`);
 const byId=Object.fromEntries(graph.nodes.map(n=>[n.id,n]));
 let edges=graph.edges.map((e,i)=>{
  const a=byId[e.from],b=byId[e.to];const x1=a.x+a.w/2,y1=a.y+a.h,x2=b.x+b.w/2,y2=b.y;
  let d,labelX=(x1+x2)/2,labelY=(y1+y2)/2-6;
  if(e.kind==='residual'){const rail=e.from==='input'?12:graph.width-12;d=`M ${a.x+(rail<20?0:a.w)} ${a.y+a.h/2} H ${rail} V ${b.y+b.h/2} H ${b.x+(rail<20?0:b.w)}`;labelX=rail+(rail<20?8:-75);labelY=(a.y+b.y)/2;}
  else{const mid=y1+Math.max(20,(y2-y1)/2);d=`M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;}
  return `<path class="graph-edge ${e.kind}" d="${d}" marker-end="url(#arrow)"/>${e.label?`<text class="edge-label" x="${labelX}" y="${labelY}">${escape(e.label)}</text>`:''}`;
 }).join('');
 svg.innerHTML=`<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#92ad9e"/></marker></defs>${edges}`+graph.nodes.map(n=>`<g class="graph-node ${n.kind}" data-node="${n.id}" role="button" tabindex="0" aria-label="${escape(n.title+': '+(info[n.key]?.body??''))}" transform="translate(${n.x},${n.y})"><rect width="${n.w}" height="${n.h}" rx="9"/><text class="node-title" style="font-size:${n.title.length>24?10.5:n.title.length>21?11.5:13}px" x="13" y="27">${escape(n.title)}</text><text class="node-sub" x="13" y="46">${escape(n.sub)}</text></g>`).join('');
 svg.classList.toggle('hide-precision',!$('#sensitivityToggle').checked);
 $$('[data-node]').forEach(el=>{const n=byId[el.dataset.node],entry=info[n.key]??info.unknown;
  el.onpointerenter=e=>{showTip(e,entry.title,entry.body);};el.onpointermove=moveTip;el.onpointerleave=hideTip;
  el.onfocus=()=>showTipAt(el,entry.title,entry.body);el.onblur=hideTip;
  const select=()=>{pinned=n.id;$$('[data-node]').forEach(g=>g.classList.toggle('pinned',g.dataset.node===pinned));inspect(n.key,n);hideTip();openLayerChoice(n.key,n);};
  el.onclick=select;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
 });
}
function inspect(key,node){
 const entry=info[key]??info.unknown;let extra='';
 if(key==='embeddings'||key==='head')extra=`<h4>VOCABULARY × HIDDEN WIDTH</h4><div class="shape">${fmt(model.vocab)} × ${fmt(model.hidden)}</div><p>${model.tied?'The config ties these weights: count their shared storage once.':'The config keeps these as separate learned matrices.'}</p>`;
 if(key==='vision'&&model.vision){const v=model.vision;extra=`<h4>VISION CONFIG</h4><p>${v.depth??'?'} vision blocks · width ${v.hidden_size??v.embed_dim??'?'} · ${v.num_heads??'?'} heads · patch ${v.patch_size??'?'} · temporal patch ${v.temporal_patch_size??'?'}. Spatial merge ${v.spatial_merge_size??'?'} × ${v.spatial_merge_size??'?'}.</p><div class="mini-flow"><span>Pixels / frames</span><span>Patch embedding</span><span>Vision transformer blocks</span><span>Patch merger → text width</span><span>Insert visual embeddings</span><span>Text decoder stack</span></div><p>This is a frontend overview. Vision-block internals and audio/talker graphs are outside the verified decoder adapters.</p>`;}
 if(key==='mtp')extra=`<p>Config declares ${model.mtp} auxiliary layer(s). They are not counted among the ${model.depth} decoder blocks.</p>`;
 if(node?.shape)extra+=`<h4>${node.kind==='state'?'STATE SHAPE (EXCLUDING BATCH)':'WEIGHT SHAPE / CONFIG DIMENSION'}</h4><div class="shape">${escape(node.shape)}</div><p class="muted">Linear weights use [output, input]. Fused projection slices share a physical tensor.</p>`;
 $('#inspector').innerHTML=`<div class="eyebrow">${node?'PINNED COMPONENT':'COMPONENT NOTES'}</div><h2>${escape(entry.title)}</h2><p>${escape(entry.body)}</p>${entry.math?`<pre class="formula">${escape(entry.math)}</pre>`:''}${extra}${node?.path?`<details><summary>Technical module path</summary><pre class="codepath">${escape(node.path)}</pre><p class="muted">Composite checkpoints may prefix paths with language_model.</p></details>`:''}<h4>PRECISION & SENSITIVITY</h4><span class="pill">${escape(entry.risk)}</span><p class="note">${escape(entry.why)}</p><h4>READ THE SOURCE</h4><div class="source-links">${sourceHTML(entry.refs)}${activePreset?`<a href="${escape(activePreset.url)}" target="_blank" rel="noopener noreferrer">Loaded config · revision ${escape(activePreset.revision.slice(0,8))} ↗</a>`:''}</div><p class="config-note">Sensitivity is recipe-dependent. Amber is a precaution, not an empirical ranking. Config: ${escape(model.family)}.</p>`;
 const paths=componentPaths(key,node);
 if(paths.length){const button=document.createElement('button');button.id='protectComponent';const selected=paths.flatMap(editPaths).every(path=>customSelections.some(e=>e.path===path));button.textContent=selected?'Remove component protection'+($('#repeatType').checked?' across this type':''):'☆ Protect this component'+($('#repeatType').checked?' across this type':'');button.onclick=()=>{for(const path of paths)for(const target of editPaths(path))toggleCustom(target,target.includes('@flow.')?'runtime':'component',info[key].title,!selected);savePlanning();renderPlanning();inspect(key,node);};$('#inspector').append(button);}

}
function componentPaths(key,node){
 if(['embeddings','head','finalnorm'].includes(key))return [{embeddings:'model.embed_tokens',head:'lm_head',finalnorm:'model.norm'}[key]];
 if(!node)return [`@flow.${key}`];
 if(!node.path||node.kind==='state'||['residual','heads','unknown'].includes(key))return [`model.layers.${index}.@flow.${node.id}`];
 if(key==='decay')return ['A_log','dt_bias'].map(n=>`model.layers.${index}.linear_attn.${n}`);
 if(key==='qknorm')return ['q_norm','k_norm'].map(n=>`model.layers.${index}.self_attn.${n}`);
 return [node.path];
}
function fit(){const width=$('#viewport').clientWidth;scale=Math.max(window.innerWidth<650?.7:.45,Math.min(1,width/graph.width));applyScale();}
function applyScale(){$('#flowSvg').style.width=`${graph.width*scale}px`;$('#flowSvg').style.height=`${graph.height*scale}px`;}
function showTip(e,title,text){$('#tooltip').innerHTML=`<b>${escape(title)}</b>${escape(text)}<small>Click to pin details and sources</small>`;$('#tooltip').hidden=false;moveTip(e);}
function moveTip(e){const t=$('#tooltip'),w=t.offsetWidth,h=t.offsetHeight;t.style.left=`${Math.max(8,Math.min(window.innerWidth-w-8,e.clientX+14))}px`;t.style.top=`${Math.max(8,Math.min(window.innerHeight-h-8,e.clientY+16))}px`;}
function showTipAt(el,title,text){const r=el.getBoundingClientRect();showTip({clientX:r.right,clientY:r.top},title,text);}
function hideTip(){$('#tooltip').hidden=true;}
function stopTrace(){clearInterval(traceTimer);traceTimer=null;$('#traceButton').textContent='▶ Trace flow';$$('.traced').forEach(el=>el.classList.remove('traced'));$('#flowStatus').textContent='Arrows show dependencies';}
$('#traceButton').onclick=()=>{if(traceTimer){stopTrace();return;}let step=0;const order=[...graph.nodes].sort((a,b)=>a.y-b.y||a.x-b.x);$('#traceButton').textContent='■ Stop trace';const tick=()=>{if(step>=order.length){stopTrace();return;}const n=order[step++];$$('.traced').forEach(el=>el.classList.remove('traced'));const el=$(`[data-node="${n.id}"]`);el.classList.add('traced');inspect(n.key,n);el.scrollIntoView({block:'nearest',inline:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});$('#flowStatus').textContent=`${step} / ${order.length} · explanatory order; branches may run in parallel`;};tick();traceTimer=setInterval(tick,1500);};
$('#prevBlock').onclick=()=>{if(index>0){index--;renderBlock();}};$('#nextBlock').onclick=()=>{if(index<model.depth-1){index++;renderBlock();}};
$('#fitButton').onclick=fit;$('#zoomIn').onclick=()=>{scale=Math.min(1.6,scale+.15);applyScale();};$('#zoomOut').onclick=()=>{scale=Math.max(.3,scale-.15);applyScale();};$('#sensitivityToggle').onchange=()=>$('#flowSvg').classList.toggle('hide-precision',!$('#sensitivityToggle').checked);
$$('[data-filter]').forEach(el=>el.onclick=()=>{filter=el.dataset.filter;$$('[data-filter]').forEach(x=>x.classList.toggle('active',x===el));$$('[data-block]').forEach(b=>b.classList.toggle('filtered',filter!=='all'&&model.blocks[Number(b.dataset.block)].type!==filter));});
$$('[data-info]').forEach(el=>{el.onclick=()=>{inspect(el.dataset.info);openLayerChoice(el.dataset.info);};el.onpointerenter=e=>showTip(e,info[el.dataset.info].title,info[el.dataset.info].body);el.onpointerleave=hideTip;el.onfocus=()=>showTipAt(el,info[el.dataset.info].title,info[el.dataset.info].body);el.onblur=hideTip;});
function switchView(next){stopTrace();hideTip();view=next;['explore','protect','custom','precision','quant','lab','learn'].forEach(v=>$(`#${v}View`).hidden=v!==next);$$('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===next));if(next==='explore')fit();if(next==='lab')renderMemoryEstimate();}
$$('[data-view]').forEach(b=>b.onclick=()=>switchView(b.dataset.view));
$('#openRunControls').onclick=()=>{switchView('quant');$('#githubActionsHeading').scrollIntoView({block:'start'});$('#githubRunToken').focus({preventScroll:true});};
$('#sourcesButton').onclick=()=>{switchView('learn');$('#sourceList').scrollIntoView({block:'start'});};
let importedConfig=null;
$('#modelSelect').onchange=e=>{if(e.target.value==='imported'&&importedConfig){activePreset=null;modelLoad(importedConfig.raw,importedConfig.name);return;}activePreset=presets.find(p=>p.id===e.target.value);modelLoad(activePreset.config,activePreset.id);};
$('#linkButton').onclick=async()=>{if(!activePreset){showNotice('Imported configs are local. Share the config file separately; it is not embedded in a URL.');return;}try{await navigator.clipboard.writeText(location.href);showNotice('View link copied.');}catch{showNotice('Copy this URL: '+location.href);}};
$('#importButton').onclick=()=>$('#importDialog').showModal();$('#chooseFile').onclick=()=>$('#configFile').click();
function importConfig(text){try{if(text.length>2_000_000)throw Error('Config exceeds the 2 MB limit.');const raw=JSON.parse(text);normalize(raw);activePreset=null;const name=$('#importName').value.trim()||'Imported Qwen';importedConfig={raw,name};modelLoad(raw,name);let option=$('#modelSelect option[value="imported"]');if(!option){option=document.createElement('option');option.value='imported';$('#modelSelect').append(option);}option.textContent=name;$('#modelSelect').value='imported';history.replaceState(null,'',location.pathname);$('#importDialog').close();$('#importError').textContent='';}catch(e){$('#importError').textContent=e.message;}}
$('#configFile').onchange=async e=>{const f=e.target.files[0];if(f){if(f.size>2_000_000){$('#importError').textContent='Config exceeds the 2 MB limit.';return;}importConfig(await f.text());}e.target.value='';};$('#loadPaste').onclick=()=>importConfig($('#configPaste').value);
let S=[[0,0],[0,0]],steps=0;
function renderMemory(output=[0,0]){$('#memoryGrid').innerHTML=S.flat().map(v=>`<div style="background:hsl(157 35% ${96-Math.min(1,Math.abs(v))*35}%)">${v.toFixed(3)}</div>`).join('');$('#stepCount').textContent=`${steps} tokens`;$('#memoryOutput').textContent=`Read output: [${output.map(x=>x.toFixed(3)).join(', ')}]`;}
$('#alpha').oninput=()=>$('#alphaValue').textContent=Number($('#alpha').value).toFixed(2);$('#beta').oninput=()=>$('#betaValue').textContent=Number($('#beta').value).toFixed(2);
$('#stepMemory').onclick=()=>{const choice=Number($('#association').value),k=choice===1?[0,1]:[1,0],v=choice===0?[1,0]:[0,1];const r=deltaStep(S,k,v,k,Number($('#alpha').value),Number($('#beta').value));S=r.next;steps++;renderMemory(r.output);$('#labNarration').textContent=`Prediction before writing: [${r.prediction.map(x=>x.toFixed(3))}]. Correction: [${r.correction.map(x=>x.toFixed(3))}]. The update writes only along the current key direction.`;};
$('#resetMemory').onclick=()=>{S=[[0,0],[0,0]];steps=0;renderMemory();$('#labNarration').textContent='State reset. In real inference, reset between independent sequences—not arbitrary chunks of one sequence.';};
function renderMemoryEstimate(){const n=Number($('#contextTokens').value);if(!Number.isSafeInteger(n)||n<1||n>2e6){$('#memoryEstimate').textContent='Enter 1–2,000,000 context tokens.';return;}const e=memoryEstimate(model,n);const mb=x=>x>=2**30?`${(x/2**30).toFixed(2)} GiB`:`${(x/2**20).toFixed(2)} MiB`;$('#memoryEstimate').innerHTML=[[e.kv,'Attention KV · grows with context'],[e.state,'DeltaNet matrix · fixed per sequence'],[e.conv,'Short convolution history']].map(([v,l])=>`<div><strong>${mb(v)}</strong><small>${l}</small></div>`).join('');}
$('#contextTokens').oninput=renderMemoryEstimate;
function renderLessons(){const term=$('#guideSearch').value.toLowerCase();const shown=lessons.filter(x=>(x.title+x.text+x.tag).toLowerCase().includes(term));$('#lessons').innerHTML=shown.length?shown.map(x=>`<article class="lesson"><div class="eyebrow">${escape(x.tag)}</div><h3>${escape(x.title)}</h3><p>${escape(x.text)}</p>${sourceHTML(x.refs)}</article>`).join(''):'<p>No matching lessons. Try “state” or “precision”.</p>';}
$('#guideSearch').oninput=renderLessons;
$('#sourceList').innerHTML=`<h2>Research & source trail</h2><p class="muted">Architecture templates reviewed against official Transformers implementations. Bundled configs are pinned to the revisions recorded in data/manifest.json. Review date: 2 October 2026. Runtime implementations and custom checkpoints may differ.</p><ul>${Object.values(sources).map(s=>`<li><a href="${s.url}" target="_blank" rel="noopener noreferrer">${escape(s.title)} ↗</a></li>`).join('')}</ul><p class="muted">Coverage: verified text-decoder adapters, vision overview, and MTP explanation. Original Qwen-1, audio/talker internals, and unknown future architectures are not silently treated as a known model. Imported files stay in your browser.</p>`;
window.addEventListener('resize',()=>{if(view==='explore')fit();hideTip();});document.addEventListener('keydown',e=>{if(e.key==='Escape')hideTip();});


function selectedProtection(){return protectionPlan(model,{vocabulary:$('#protectVocabulary').checked,gates:$('#protectGates').checked});}
function renderProtection(){
 const plan=selectedProtection(),term=$('#protectionSearch').value.toLowerCase();
 const rows=plan.entries.filter(e=>(e.path+' '+info[e.key].title+' '+info[e.key].why+' '+e.note).toLowerCase().includes(term));
 $('#protectionSummary').textContent=`${model.name}: ${plan.entries.length} module / parameter entries (${rows.length} shown). Config-derived recommendations, not measured sensitivity scores.`;
 $('#protectionNotes').innerHTML=plan.notes.map(n=>`<li>${escape(n)}</li>`).join('');
 $('#exportProtection').disabled=!plan.entries.length;
 $('#protectionRows').innerHTML=rows.length?rows.map(e=>`<tr><td>${e.block===null?'Global':`<button data-protect-block="${e.block}" data-protect-node="${escape(e.node)}" data-protect-key="${escape(e.key)}">${e.block+1}</button>`}</td><td><code>${escape(e.path)}</code><small>${escape(info[e.key].title)}</small></td><td>${escape(info[e.key].why)}${e.note?`<p>${escape(e.note)}</p>`:''}</td></tr>`).join(''):'<tr><td colspan="3">No matching protection entries.</td></tr>';
 $$('[data-protect-block]').forEach(el=>el.onclick=()=>{index=Number(el.dataset.protectBlock);switchView('explore');renderBlock();const node=graph.nodes.find(n=>n.id===el.dataset.protectNode);inspect(el.dataset.protectKey,node);if(node){const target=$(`[data-node="${node.id}"]`);target.classList.add('pinned');target.scrollIntoView({block:'center'});}});
}
$('#protectVocabulary').onchange=()=>{savePlanning();renderProtection();renderPlanning();};$('#protectGates').onchange=()=>{savePlanning();renderProtection();renderPlanning();};$('#protectionSearch').oninput=renderProtection;
$('#exportProtection').onclick=()=>{
 const plan=selectedProtection();const data={schema_version:1,model:model.name,family:model.family,source:activePreset?{url:activePreset.url,revision:activePreset.revision}:null,policy:'conservative-original-precision',basis:'config-derived; not empirically measured',scope:'text decoder, optional vocabulary; not a complete checkpoint inventory',complete_checkpoint_coverage:plan.complete,tied_embeddings:model.tied,options:{vocabulary:$('#protectVocabulary').checked,delta_gates:$('#protectGates').checked},notes:plan.notes,entries:plan.entries.map(e=>({...e,reason:info[e.key].why}))};
 const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='qwen-protection-list.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};

function loadPlanning(raw){
 planningKey='qwen-atlas-plan-v1:'+JSON.stringify(raw);customSelections=[];precisionChoices={};groupChoices={};automaticOptOut={};for(const [id,value] of Object.entries({sizeOriginal:2,sizeGroup:128,sizeScale:-1,sizeExtras:0}))$('#'+id).value=value;
 $('#protectVocabulary').checked=true;$('#protectGates').checked=true;$('#repeatType').checked=true;
 try{const saved=JSON.parse(localStorage.getItem(planningKey)||'null');if(saved){$('#repeatType').checked=saved.repeat_type!==false;if(saved.sizeInputs)for(const id of ['sizeOriginal','sizeGroup','sizeScale','sizeExtras'])if(saved.sizeInputs[id]!==undefined)$('#'+id).value=saved.sizeInputs[id];groupChoices=saved.groups&&typeof saved.groups==='object'?saved.groups:{};customSelections=Array.isArray(saved.custom)?saved.custom.filter(e=>typeof e.path==='string'&&['block','component','runtime'].includes(e.scope)):[];precisionChoices=saved.precisions&&typeof saved.precisions==='object'?saved.precisions:{};automaticOptOut=saved.automaticOptOut&&typeof saved.automaticOptOut==='object'?saved.automaticOptOut:{};$('#protectVocabulary').checked=saved.vocabulary!==false;$('#protectGates').checked=saved.gates!==false;}}catch{}
 const inventory=precisionInventory(model),expanded=[];
 for(const e of customSelections){if(e.scope==='block'){for(const row of inventory.filter(row=>row.path.startsWith(e.path+'.')))expanded.push({path:row.path,scope:'component',label:info[row.key].title});}else expanded.push(e);}
 customSelections=[...new Map(expanded.map(e=>[e.path,e])).values()];
}

function savePlanning(){try{localStorage.setItem(planningKey,JSON.stringify({custom:customSelections,precisions:precisionChoices,groups:groupChoices,automaticOptOut,repeat_type:$('#repeatType').checked,sizeInputs:Object.fromEntries(['sizeOriginal','sizeGroup','sizeScale','sizeExtras'].map(id=>[id,$('#'+id).value])),vocabulary:$('#protectVocabulary').checked,gates:$('#protectGates').checked}));}catch{showNotice('Browser storage is unavailable or full. Download your plan to keep these selections.');}}
function toggleCustom(path,scope,label,keep){customSelections=customSelections.filter(e=>e.path!==path);if(keep)customSelections.push({path,scope,label});}
function updateBlockProtection(){renderInnerLayers();}
function protectionSources(path,auto){const paths=model.tied&&['lm_head','model.embed_tokens'].includes(path)?['lm_head','model.embed_tokens']:[path];const matches=(parent,p)=>p===parent||p.startsWith(parent+'.');const sources=[];if(paths.some(p=>!automaticOptOut[p]&&auto.some(e=>matches(e.path,p))))sources.push('Automatic');if(paths.some(p=>customSelections.some(e=>matches(e.path,p))))sources.push('Custom');return sources;}
function precisionRows(){const auto=selectedProtection().entries;return planningInventory().map(e=>{const sources=protectionSources(e.path,auto);return {...e,protection:sources,precision:sources.length?'original':precisionChoices[e.path]||'original',...(e.kind==='weight'?{group_size:tkIntegerFormat(precisionChoices[e.path])?.group_size??groupChoices[e.path]??Number($('#sizeGroup').value)}:{})};});}
const precisionOptions=['original','BF16','FP16','FP32','INT8','INT4','ternary','TKinteger-20_44-k1',...Array.from({length:16},(_,i)=>`TKinteger-k${i+1}`),...[120,60,40,30,20,15,10,5,4,3,2,1].map(g=>`TKinteger-k1-g${g}-s12`),...[16,32].flatMap(bits=>[1,2,4,8,16,32,64].map(g=>`TKinteger-k1-g${g}-s${bits}`)),...['binary','ternary'].flatMap(kind=>[64,128,256].map(r=>`NQ-${kind}-r${r}`))];
function precisionLabel(value){const f=tkIntegerFormat(value);return f?.method==='factorization'?`${f.name} · rank ${f.rank}`:f?`${f.name} k${f.k}${f.group_size?' G'+f.group_size:''} · ${(f.total_bits_per_value??f.payload_bits_per_value).toFixed(3)} ${f.total_bits_per_value!==undefined?'bits/weight incl. scales':'code bits/value'}`:value==='original'?'Original precision':value;}
function renderPlanning(){
 for(const value of Object.values(precisionChoices))if(tkIntegerFormat(value)&&!precisionOptions.includes(value))precisionOptions.push(value);
 updateBlockProtection();renderSizeEstimate();
 $('#customSummary').textContent=`${customSelections.length} custom selections for ${model.name}. Custom protection adds to automatic protection.`;
 $('#customRows').innerHTML=customSelections.length?customSelections.map((e,i)=>`<tr><td><code>${escape(e.path)}</code><small>${escape(e.label)}</small></td><td>${e.scope==='runtime'?'Runtime operation / state':e.scope==='block'?'Entire decoder block and all descendants':'Module / parameter group'}</td><td><button data-remove-custom="${i}">Remove</button></td></tr>`).join(''):'<tr><td colspan="3">No custom selections yet. Select a block in Explore to get started.</td></tr>';
 $$('[data-remove-custom]').forEach(el=>el.onclick=()=>{customSelections.splice(Number(el.dataset.removeCustom),1);savePlanning();renderPlanning();});
 const rows=precisionRows(),term=$('#precisionSearch').value.toLowerCase(),matches=rows.filter(e=>(e.path+' '+info[e.key].title).toLowerCase().includes(term));
 precisionPage=Math.max(0,Math.min(precisionPage,Math.ceil(matches.length/50)-1));const shown=matches.slice(precisionPage*50,(precisionPage+1)*50);
 $('#precisionPrev').disabled=precisionPage===0;$('#precisionNext').disabled=(precisionPage+1)*50>=matches.length;
 $('#precisionSummary').textContent=`${rows.length} weight / runtime targets; ${rows.filter(e=>e.protection.length).length} protected. ${matches.length} match; page ${precisionPage+1} of ${Math.max(1,Math.ceil(matches.length/50))}.`;
 $('#precisionRows').innerHTML=shown.map(e=>`<tr><td><code>${escape(e.path)}</code><small>${escape(info[e.key].title)} · ${escape(e.kind)}</small></td><td>${e.protection.join(' + ')||'No protection selected'}</td><td><select data-precision-path="${escape(e.path)}" aria-label="Precision for ${escape(e.path)}" >${precisionOptions.map(x=>`<option value="${x}" ${x===e.precision?'selected':''}>${escape(precisionLabel(x))}</option>`).join('')}</select>${e.kind==='weight'?`<small>G${e.group_size} · independent scales per tensor</small>`:''}</td></tr>`).join('')||'<tr><td colspan="3">No modeled modules match.</td></tr>';
 $$('[data-precision-path]').forEach(el=>el.onchange=()=>{const path=el.dataset.precisionPath;setPlannedPrecision(editPaths(path),el.value);savePlanning();renderPlanning();});
}
$('#precisionSearch').oninput=()=>{precisionPage=0;renderPlanning();};
$('#precisionPrev').onclick=()=>{precisionPage--;renderPlanning();};$('#precisionNext').onclick=()=>{precisionPage++;renderPlanning();};
function buildPlan(kind='combined'){const data={schema_version:5,options:{vocabulary:$('#protectVocabulary').checked,delta_gates:$('#protectGates').checked},automatic_protection_opt_out:automaticOptOut,model:model.name,family:model.family,config:model.raw,basis:'User plan and config-derived policy; no weights have been changed',path_semantics:'Text-model-relative module groups; resolve wrapper prefixes against the actual checkpoint. Module and block selections cover descendants.',scope:'Weight module groups plus graph-level runtime operation/state targets. @flow identifiers are planning IDs, not checkpoint tensor names. Unmodeled frontend internals remain outside scope.',tied_embeddings:model.tied};if(kind!=='precision')data.custom_protection=customSelections;if(kind==='combined'){data.automatic_protection=selectedProtection();data.editor_state={precision_choices:precisionChoices,group_sizes:groupChoices,automatic_opt_out:automaticOptOut,repeat_type:$('#repeatType').checked};}if(kind!=='custom'){try{data.size_estimate=currentSizeEstimate();data.size_assumptions={original_bytes:Number($('#sizeOriginal').value),scale_group_size:Number($('#sizeGroup').value),scale_bytes_per_group:Number($('#sizeScale').value),extras_MB:Number($('#sizeExtras').value)};}catch{data.size_estimate=null;}}if(kind!=='custom')data.quant_job=quantJobSettings();if(kind!=='custom')data.precision=precisionRows().map(row=>({...row,format:tkIntegerFormat(row.precision)}));return data;}
function downloadPlan(kind){const data=buildPlan(kind);const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`qwen-${kind}-plan.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('#exportCustom').onclick=()=>downloadPlan('custom');$('#exportPrecision').onclick=()=>downloadPlan('precision');$('#exportPlan').onclick=()=>downloadPlan('combined');

function editPaths(path){
 if(!$('#repeatType').checked)return [path];
 const match=/^model\.layers\.(\d+)\.@flow\.(.+)$/.exec(path);
 if(match){const b=model.blocks[Number(match[1])];return model.blocks.filter(x=>x.type===b.type&&blockGraph(model,x.index).nodes.some(n=>n.id===match[2])).map(x=>`model.layers.${x.index}.@flow.${match[2]}`);}
 return matchingBlockPaths(model,path);
}
function planningInventory(m=model){
 const rows=precisionInventory(m).map(e=>({...e,kind:'weight'}));
 for(const b of m.blocks){if(b.type==='linear_attention'&&!m.linear)continue;for(const n of blockGraph(m,b.index).nodes){if(!n.path||n.kind==='state'||['residual','heads','unknown'].includes(n.key))rows.push({path:`model.layers.${b.index}.@flow.${n.id}`,key:n.key,block:b.index,kind:n.kind==='state'?'runtime_state':'runtime_operation'});}}
 for(const [key,exists] of [['vision',m.vision],['mtp',m.mtp]])if(exists)rows.push({path:`@flow.${key}`,key,block:null,kind:'unexpanded_subsystem'});
 return rows;
}
function aliasesFor(path){return model.tied&&['lm_head','model.embed_tokens'].includes(path)?['lm_head','model.embed_tokens']:[path];}
function setPlannedPrecision(paths,value){for(const path of [...new Set(paths.flatMap(aliasesFor))]){precisionChoices[path]=value;if(tkIntegerFormat(value)?.group_size)delete groupChoices[path];automaticOptOut[path]=true;customSelections=customSelections.filter(e=>e.path!==path);}}

function setEditScope(repeat){$('#repeatType').checked=repeat;savePlanning();renderPlanning();inspect(model.blocks[index].type==='linear_attention'?'delta':'attention');}
$('#repeatType').onchange=()=>setEditScope($('#repeatType').checked);
$('#editThisBlock').onclick=()=>setEditScope(false);
$('#editMatchingBlocks').onclick=()=>setEditScope(true);

for(const [id,type] of [['openDeltaTemplate','linear_attention'],['openAttentionTemplate','full_attention']]){
 $('#'+id).onclick=()=>{const b=model.blocks.find(b=>b.type===type);if(b){index=b.index;switchView('explore');renderBlock();$('.flow-panel').scrollIntoView({block:'start'});}};
}
$('#clearTypeCustom').onclick=()=>{const indices=new Set(model.blocks.filter(b=>$('#repeatType').checked?b.type===model.blocks[index].type:b.index===index).map(b=>b.index));customSelections=customSelections.filter(e=>{const m=/^model\.layers\.(\d+)\./.exec(e.path);return !m||!indices.has(Number(m[1]));});savePlanning();renderPlanning();};

function renderInnerLayers(){
 $('#openDeltaTemplate').disabled=!model.blocks.some(b=>b.type==='linear_attention');$('#openAttentionTemplate').disabled=!model.blocks.some(b=>b.type==='full_attention');
 const count=model.blocks.filter(b=>b.type===model.blocks[index].type).length;
 $('#innerLayerScope').textContent=$('#repeatType').checked?`Layer choices apply across ${count} matching blocks.`:`Editing only Block ${index+1}.`;
 const repeat=$('#repeatType').checked;
 $('#editThisBlock').textContent=`This block only · ${index+1}`;
 $('#editThisBlock').setAttribute('aria-pressed',String(!repeat));
 $('#editMatchingBlocks').textContent=`All matching blocks · ${count}`;
 $('#editMatchingBlocks').setAttribute('aria-pressed',String(repeat));
 $('#blockScopeHelp').textContent=repeat?'Click a layer in the flow chart to change that layer across matching blocks.':`Click a layer in the flow chart to change it only in Block ${index+1}. This mode stays on when you move to another block.`;
 $('#clearTypeCustom').textContent=repeat?'Clear custom protection for this type':`Clear custom protection for Block ${index+1}`;
 const rows=precisionRows();
 $$('[data-node]').forEach(el=>{const node=graph.nodes.find(n=>n.id===el.dataset.node),paths=componentPaths(node.key,node),custom=paths.length>0&&paths.every(path=>customSelections.some(e=>e.path===path));el.classList.toggle('custom-layer-selected',custom);const statuses=rows.filter(e=>paths.includes(e.path));const auto=statuses.some(e=>e.protection.includes('Automatic'));const label=el.querySelector('.node-sub');if(paths.length){const values=[...new Set(statuses.map(e=>e.precision))];label.textContent=custom?'★ Custom protected':auto?'Auto protected':values.length===1&&values[0]!=='original'?`Precision: ${values[0]==='TKinteger-20_44-k1'?'20/44 k1 · 2.2b':tkIntegerFormat(values[0])?precisionLabel(values[0]):values[0]}`:'Click to choose';}el.setAttribute('aria-label',node.title+(custom?' — custom protected':auto?' — automatically protected':' — click for settings'));});
}
function openLayerChoice(key,node){
 const entry=info[key]??info.unknown,paths=componentPaths(key,node),targets=[...new Set(paths.flatMap(editPaths).flatMap(aliasesFor))];
 $('#layerChoiceTitle').textContent=entry.title;$('#layerChoiceDescription').textContent=entry.body;
 const runtime=paths.some(p=>p.includes('@flow.')),selected=targets.filter(path=>customSelections.some(e=>e.path===path)).length;
 const rows=precisionRows().filter(e=>targets.includes(e.path)),values=[...new Set(rows.map(e=>e.precision))];
 const blocks=new Set(targets.map(p=>/^model\.layers\.(\d+)/.exec(p)?.[1]).filter(x=>x!==undefined));
 const autoEntries=selectedProtection().entries,autoEligible=targets.some(p=>autoEntries.some(e=>p===e.path||p.startsWith(e.path+'.'))),autoActive=rows.some(e=>e.protection.includes('Automatic'));
 $('#layerChoiceControls').innerHTML=`<p class="choice-scope">${blocks.size?`Applies to this ${runtime?'operation / state':'layer'} in ${blocks.size} matching block${blocks.size===1?'':'s'}.`:'Applies to this global component.'}</p><label class="choice-protect"><input id="graphProtect" type="checkbox" ${selected===targets.length?'checked':''}> Keep original precision (custom protection)</label>${autoEligible?`<label class="choice-protect"><input id="graphAuto" type="checkbox" ${autoActive?'checked':''}> Use automatic protection recommendation</label>`:''}<label for="graphPrecision">${runtime?'Runtime / operation precision target':'Weight precision target'}</label><select id="graphPrecision">${values.length>1?'<option value="mixed" selected disabled>Mixed settings</option>':''}${precisionOptions.map(x=>`<option value="${x}" ${values.length===1&&values[0]===x?'selected':''}>${escape(precisionLabel(x))}</option>`).join('')}</select><div id="tkFormatNote"></div><p class="muted">Choosing a precision overrides protection for this selection. All options remain editable.</p><p class="note">${runtime?'This box represents an operation, state, or an unexpanded subsystem. This target is planning metadata, not a stored weight dtype. Low-bit execution requires runtime support and separate validation.':'This is a weight precision plan. No model weights are changed by this app.'}</p>${model.family==='qwen3_next'&&['a','b','qkv','z'].includes(key)?'<p class="note">This box shares a fused tensor with another box. Settings apply to that shared tensor.</p>':''}`;

 if(!runtime){
 const ownRows=precisionRows().filter(e=>paths.includes(e.path)),groupValues=[...new Set(ownRows.map(e=>e.group_size))];
 const groupPanel=document.createElement('section');groupPanel.className='layer-group-controls';
 groupPanel.innerHTML=`<h3>Group scales for this layer type</h3><label>Weights per scale <input id="graphGroupSize" type="number" min="1" max="65536" step="1" value="${groupValues.length===1?groupValues[0]:128}"></label><label>Apply group size to <select id="graphGroupScope"><option value="this">This block only</option><option value="matching" ${$('#repeatType').checked?'selected':''}>Matching block type</option><option value="all">This layer type across all blocks</option></select></label><button id="applyGraphGroup" type="button">Apply group size</button><p class="muted">${groupValues.length>1?'Selected layers currently have different group sizes. ':''}Each tensor gets its own fitted scales. This shares the group-size setting, never scale values. Protected weights stay protected. Factorization uses its own row/column scales.</p><p id="graphGroupMessage" role="status"></p>`;
 $('#layerChoiceControls').append(groupPanel);
 $('#applyGraphGroup').onclick=()=>{try{
 const g=Number($('#graphGroupSize').value);if(!Number.isSafeInteger(g)||g<1||g>65536)throw Error('Choose an integer group size from 1 to 65536.');
 const scope=$('#graphGroupScope').value,ps=[...new Set(paths.flatMap(p=>scope==='this'?[p]:matchingBlockPaths(model,p,scope==='all')).flatMap(aliasesFor))];
 for(const p of ps){const f=tkIntegerFormat(precisionChoices[p]);if(f?.method==='factorization')continue;if(f?.group_size){const v=`TKinteger-k${f.k}-g${g}-s${f.scale_bits}`;precisionChoices[p]=v;if(!precisionOptions.includes(v))precisionOptions.push(v);delete groupChoices[p];}else groupChoices[p]=g;}
 savePlanning();renderPlanning();openLayerChoice(key,node);$('#graphGroupMessage').textContent=`Applied G${g} to matching weight settings. Scale values remain independent per tensor.`;
 }catch(e){$('#graphGroupMessage').textContent=e.message;}};
 }
 const selectedFormat=values.length===1?tkIntegerFormat(values[0]):null;
 $('#tkFormatNote').textContent=selectedFormat?precisionLabel(values[0])+'. '+selectedFormat.packing+'. '+selectedFormat.codec_status+'. Size includes scales in the calculator; headers remain extras.':'';
 $('#graphProtect').indeterminate=selected>0&&selected<targets.length;
 $('#graphProtect').onchange=e=>{for(const path of targets)toggleCustom(path,runtime?'runtime':'component',entry.title,e.target.checked);savePlanning();renderPlanning();inspect(key,node);openLayerChoice(key,node);};
 if($('#graphAuto'))$('#graphAuto').onchange=e=>{for(const path of targets)automaticOptOut[path]=!e.target.checked;savePlanning();renderPlanning();openLayerChoice(key,node);};
 $('#graphPrecision').onchange=e=>{setPlannedPrecision(targets,e.target.value);savePlanning();renderPlanning();inspect(key,node);openLayerChoice(key,node);};
 if(!$('#layerChoice').open)$('#layerChoice').showModal();
}
$('#layerChoiceDone').onclick=()=>$('#layerChoice').close();

function currentSizeEstimate(){return estimateWeightStorage(model,Object.fromEntries(precisionRows().filter(e=>e.kind==='weight').map(e=>[e.path,e])),{originalBytes:Number($('#sizeOriginal').value),groupSize:Number($('#sizeGroup').value),scaleBytes:Number($('#sizeScale').value),extraBytes:Number($('#sizeExtras').value)*1e6});}
function renderSizeEstimate(){
 try{const r=currentSizeEstimate();const gb=n=>(n/1e9).toFixed(3)+' GB';
 $('#sizeTotal').textContent=gb(r.total);$('#sizeGib').textContent=(r.total/2**30).toFixed(3)+' GiB';
 $('#sizeBreakdown').textContent=`${(r.parameters/1e9).toFixed(3)}B represented parameters · codes / weights ${gb(r.payload)} · scales ${gb(r.scales)} · extras ${gb(r.extraBytes)} · protected weights ${gb(r.protectedBytes)} (included in weights).`;
 const choices=precisionRows();
 for(const [id,path] of [['sizeEmbeddingPrecision','model.embed_tokens'],['sizeHeadPrecision','lm_head']]){const row=choices.find(e=>e.path===path);$('#'+id).innerHTML=precisionOptions.map(value=>`<option value="${value}" ${value===(row?.precision??'original')?'selected':''}>${escape(precisionLabel(value))}</option>`).join('');$('#'+id).disabled=!row;}
 $('#sizeVocabularyNote').textContent=model.tied?'Shared embedding/head weights: changing either selector updates both. Their payload and scales are counted once under embeddings.':`Separate matrices: embeddings and head each contain ${fmt(model.vocab)} × ${fmt(model.hidden)} weights. Both are included, each at its selected precision.`;
 $('#sizeComponentRows').innerHTML=[['Decoder body + final norm',r.breakdown.body],['Embeddings'+(model.tied?' + shared head':''),r.breakdown.embeddings],['Output head'+(model.tied?' (shared; no extra storage)':''),r.breakdown.head],['Other tensors + headers',{payload:r.extraBytes,scales:0,total:r.extraBytes}]].map(([label,c])=>`<tr><td>${escape(label)}</td><td>${gb(c.payload)}</td><td>${gb(c.scales)}</td><td>${gb(c.total)}</td></tr>`).join('');
 $('#sizeReference').textContent=`Compared with your 5.9 GB result: ${r.total>=5.9e9?'+':'−'}${(Math.abs(r.total-5.9e9)/1e9).toFixed(3)} GB. This compares bytes, not model quality.`;
 $('#sizeScope').textContent=r.scope+(r.warnings.length?` ${r.warnings.length} codec compatibility warnings; see below and export for all paths.`:'')+(r.unknown.length?` ${r.unknown.length} module groups could not be counted.`:'')+(!model.known?' Unknown architecture: no usable weight estimate.':'');
 $('#sizeCodecWarnings').textContent=r.warnings.slice(0,12).join('\n')+(r.warnings.length>12?'\nAdditional warnings are included in the exported size estimate.':'');
 }catch(e){$('#sizeTotal').textContent='Check inputs';$('#sizeBreakdown').textContent=e.message;$('#sizeGib').textContent='';$('#sizeReference').textContent='';}
}
for(const [id,path] of [['sizeEmbeddingPrecision','model.embed_tokens'],['sizeHeadPrecision','lm_head']])$('#'+id).onchange=event=>{setPlannedPrecision([path],event.target.value);savePlanning();renderPlanning();};
for(const id of ['sizeOriginal','sizeGroup','sizeScale','sizeExtras'])$('#'+id).oninput=()=>{savePlanning();renderSizeEstimate();};
$('#sizeBulkFormat').innerHTML=precisionOptions.filter(x=>tkIntegerFormat(x)).map(x=>`<option value="${x}">${precisionLabel(x)}</option>`).join('');$('#sizeBulkFormat').value='TKinteger-k1';
$('#sizeApply').onclick=()=>{for(const row of precisionRows())if(row.kind==='weight'&&!row.protection.length)precisionChoices[row.path]=$('#sizeBulkFormat').value;savePlanning();renderPlanning();showNotice('Applied the selected format to unprotected weight modules. Review codec compatibility warnings before exporting the run.');};

$('#importPlanButton').onclick=()=>$('#planFile').click();
$('#planFile').onchange=async event=>{
 const file=event.target.files[0];event.target.value='';if(!file)return;
 try{
  if(file.size>20_000_000)throw Error('Plan exceeds the 20 MB limit.');
  const data=JSON.parse(await file.text());
  if(!data||typeof data!=='object'||Array.isArray(data))throw Error('Expected a JSON plan object.');
  const namedPreset=presets.find(p=>p.id===data.model);
  const raw=data.config??namedPreset?.config??(data.model===model.name?model.raw:null);
  if(!raw)throw Error('This older export has no model configuration. Select its original model first, then import again.');
  const name=typeof data.model==='string'?data.model:'Imported plan';const candidate=normalize(raw,name);
  if(data.family&&candidate.family!==data.family)throw Error('Plan family does not match its model configuration.');
  const restored=validatePlanImport(data,candidate,planningInventory(candidate),precisionOptions);
  for(const row of data.precision??[])if(tkIntegerFormat(row.precision)&&!precisionOptions.includes(row.precision))precisionOptions.push(row.precision);
  // No changes occur until the complete file has passed validation.
  const preset=presets.find(p=>JSON.stringify(p.config)===JSON.stringify(raw));
  activePreset=preset??null;modelLoad(raw,preset?.id??name);
  if(preset)$('#modelSelect').value=preset.id;
  else{importedConfig={raw,name};let option=$('#modelSelect option[value="imported"]');if(!option){option=document.createElement('option');option.value='imported';$('#modelSelect').append(option);}option.textContent=name;$('#modelSelect').value='imported';history.replaceState(null,'',location.pathname);}
  if(restored.groups!==undefined)groupChoices=restored.groups;
  if(restored.precisions!==undefined)precisionChoices=restored.precisions;
  if(restored.custom!==undefined)customSelections=restored.custom;
  if(restored.optOut!==undefined)automaticOptOut=restored.optOut;
  if(restored.options?.vocabulary!==undefined)$('#protectVocabulary').checked=restored.options.vocabulary;
  if(restored.options?.delta_gates!==undefined)$('#protectGates').checked=restored.options.delta_gates;
  if(restored.repeat!==undefined)$('#repeatType').checked=restored.repeat;
  if(restored.size){const s=restored.size;for(const [id,value] of Object.entries({sizeOriginal:s.original_bytes,sizeGroup:s.scale_group_size,sizeScale:s.scale_bytes_per_group,sizeExtras:s.extras_MB}))$('#'+id).value=value;}
  if(restored.job){const j=restored.job;for(const [id,key] of Object.entries({jobModel:'model',jobRevision:'revision',jobCorpus:'corpus',jobTokens:'calibration_tokens',jobDocument:'max_document_tokens',jobChunk:'chunk_size',jobADMM:'admm_steps',jobCompensation:'compensation_steps',jobBlock:'block_steps',jobKL:'kl_steps',jobHessian:'hessian_dir'}))$('#'+id).value=j[key]??(key==='compensation_steps'?8:'');}
  savePlanning();renderProtection();renderPlanning();switchView('explore');
  showNotice(`Imported ${file.name} for ${model.name}. Restored ${Object.keys(restored.precisions??{}).length} precision choices and ${restored.custom?.length??0} custom protections. Size estimate recalculated. Sections absent from a partial export retain this model’s saved settings.`);
 }catch(error){showNotice('Plan import failed: '+error.message+' Your existing settings were not replaced.');}
};
modelLoad(activePreset.config,activePreset.id);renderMemory();renderLessons();




function quantJobSettings(){
 const num=(id,min,max)=>{const n=Number($('#'+id).value);if(!Number.isSafeInteger(n)||n<min||n>max)throw Error('Invalid '+id);return n;};
 const job={model:$('#jobModel').value.trim()||model.name,revision:$('#jobRevision').value.trim()||'main',corpus:$('#jobCorpus').value,calibration_tokens:num('jobTokens',128,100000000),max_document_tokens:num('jobDocument',32,1000000),chunk_size:num('jobChunk',1,100000),admm_steps:num('jobADMM',1,10000),compensation_steps:num('jobCompensation',0,10000),block_steps:num('jobBlock',0,10000),kl_steps:num('jobKL',0,10000),reconstruction_documents:4,eval_documents_per_lane:2,hessian_dir:$('#jobHessian').value.trim(),reuse_hessian:!!$('#jobHessian').value.trim(),context_policy:'native-cache-within-document',reset_policy:'document-boundary'};
 if(job.chunk_size>=job.max_document_tokens)throw Error('Maximum document length must exceed chunk size for the carry check.');return job;
}
function calculateFormats(){try{
 const k=Number($('#calcK').value),g=Number($('#calcGroup').value),o=Number($('#calcRows').value),i=Number($('#calcCols').value),r=Number($('#factorRank').value),kind=$('#factorKind').value;
 if(![k,g,o,i,r].every(Number.isSafeInteger)||o<1||i<1||r<1)throw Error('Use positive integer dimensions and rank.');
 const bits=Number($('#calcScale').value),f=tkIntegerFormat(`TKinteger-k${k}-g${g}-s${bits}`);if(!f)throw Error('k must be 1–16; group size an integer from 1 to 65536.');
 const groups=o*Math.ceil(i/g),payload=bits===12?groups*Math.ceil(g*k/5):o*Math.ceil(i*k/5),scales=Math.ceil(groups*bits/8);
 $('#codecCalculation').textContent=`${f.total_bits_per_value.toFixed(4)} bits/weight including scales (full groups; actual padding below)\n${payload.toLocaleString()} code bytes + ${scales.toLocaleString()} scale bytes = ${(payload+scales).toLocaleString()} bytes\n${((payload+scales)*8/(o*i)).toFixed(6)} actual bits/weight with group and byte padding${bits!==12&&i%g?' — runner requires group size to divide matrix columns':''}`;
 const factors=kind==='binary'?Math.ceil(o*r/8)+Math.ceil(r*i/8):o*Math.ceil(r/5)+r*Math.ceil(i/5),fs=2*(o+i);
 $('#factorCalculation').textContent=`U: ${o} × ${r}; Vᵀ: ${r} × ${i}\n${factors.toLocaleString()} factor bytes + ${fs.toLocaleString()} scale bytes\n${((factors+fs)*8/(o*i)).toFixed(6)} effective bits/original weight${r>Math.min(o,i)?' — rank exceeds matrix dimensions':''}`;
 }catch(e){$('#codecCalculation').textContent=e.message;$('#factorCalculation').textContent='Check calculator inputs';}}
function registerFormat(value){if(!tkIntegerFormat(value))throw Error('Invalid format');if(!precisionOptions.includes(value))precisionOptions.push(value);$('#sizeBulkFormat').innerHTML=precisionOptions.filter(x=>tkIntegerFormat(x)).map(x=>`<option value="${x}">${escape(precisionLabel(x))}</option>`).join('');$('#sizeBulkFormat').value=value;renderPlanning();$('#quantNotice').textContent=`${precisionLabel(value)} is available in every layer selector. Choose a layer in the flow graph, or apply the bulk selection above.`;}
$('#addTKFormat').onclick=()=>{try{registerFormat(`TKinteger-k${Number($('#calcK').value)}-g${Number($('#calcGroup').value)}-s${Number($('#calcScale').value)}`);}catch(e){$('#quantNotice').textContent=e.message;}};
$('#addFactorFormat').onclick=()=>{try{registerFormat(`NQ-${$('#factorKind').value}-r${Number($('#factorRank').value)}`);}catch(e){$('#quantNotice').textContent=e.message;}};
for(const id of ['calcK','calcGroup','calcScale','calcRows','calcCols','factorRank','factorKind'])$('#'+id).oninput=calculateFormats;
function actionsSettings(){const repo=$('#jobRepo').value.trim(),branch=$('#jobBranch').value.trim(),path=$('#jobPlanPath').value.trim();if(!/^[\w.-]+\/[\w.-]+$/.test(repo)||!branch||!path||path.includes('..')||path.startsWith('/'))throw Error('Enter owner/repo, a branch, and a repository-relative plan path.');return {repo,branch,path};}
const shellQuote=s=>"'"+s.replaceAll("'","'\\''")+"'";
function renderActions(){try{const s=actionsSettings();$('#actionsCommand').textContent=`gh workflow run atlas-quant.yml --repo ${shellQuote(s.repo)} --ref ${shellQuote(s.branch)} -f plan_path=${shellQuote(s.path)}`;}catch(e){$('#actionsCommand').textContent=e.message;}}
for(const id of ['jobRepo','jobBranch','jobPlanPath'])$('#'+id).oninput=renderActions;
$('#openQuantActions').onclick=()=>{try{window.open(`https://github.com/${actionsSettings().repo}/actions/workflows/atlas-quant.yml`,'_blank','noopener,noreferrer');}catch(e){$('#quantNotice').textContent=e.message;}};
$('#exportRunPlan').onclick=()=>{try{const data=buildPlan(),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='atlas-run.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('#quantNotice').textContent='Plan downloaded. Commit it to the selected repository path before starting the workflow.';}catch(e){$('#quantNotice').textContent=e.message;}};
calculateFormats();renderActions();

$('#startQuantRun').onclick=async()=>{
 const button=$('#startQuantRun');if(button.disabled)return;button.disabled=true;
 try{
  const s=actionsSettings(),plan=buildPlan();
  const r=await launchQuantization({repo:s.repo,branch:s.branch,planPath:s.path,token:$('#githubRunToken').value,plan,onStatus:message=>$('#quantNotice').textContent=message});
  $('#quantNotice').textContent='GitHub accepted the run request. It may be queued until your GPU runner is available. Plan commit: '+r.commit.slice(0,12)+'. ';
  const link=document.createElement('a');link.href=r.url;link.target='_blank';link.rel='noopener noreferrer';link.textContent=r.runId?'View run #'+r.runId:'View workflow runs';$('#quantNotice').append(link);
 }catch(e){$('#quantNotice').textContent=e.message;}
 finally{$('#githubRunToken').value='';button.disabled=false;}
};
