// 数据与交互控制：联读 stage 03 原文和 stage 04 图谱，并提供逐 chunk 的可视化对照。
const state={graphs:[],texts:new Map(),active:0,tab:'schema',sort:'score',highlight:true,filters:{entity:true,event:true},simulation:null};
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=v=>String(v??'—').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const confidence=o=>Number(o?.metadata?.confidence??o?.final_score??o?.edge_score??0);
const truncate=(s,n=72)=>String(s||'').length>n?String(s).slice(0,n)+'…':String(s||'');

async function loadData(){
  $('#loadingScreen').classList.remove('hidden'); $('#errorScreen').classList.add('hidden');
  try{
    const [extraction,summary]=await Promise.all([
      fetch('../stage_04_text_extraction.json',{cache:'no-store'}).then(checkResponse),
      fetch('../stage_03_document_summary.json',{cache:'no-store'}).then(checkResponse)
    ]);
    state.graphs=extraction.graphs||[]; state.texts=new Map(); collectChunks(summary.document,state.texts);
    if(!state.graphs.length) throw new Error('stage_04_text_extraction.json 中没有 graphs 数据。');
    state.active=0; renderAll(extraction.statistics||{}); $('#loadingScreen').classList.add('hidden');
  }catch(error){
    $('#loadingScreen').classList.add('hidden'); $('#errorScreen').classList.remove('hidden');
    $('#errorMessage').textContent=error.message||String(error);
  }
}
async function checkResponse(response){if(!response.ok)throw new Error(`读取 ${response.url.split('/').pop()} 失败（${response.status}）`);return response.json()}
function collectChunks(node,map){
  if(!node)return;if(Array.isArray(node)){node.forEach(v=>collectChunks(v,map));return}
  if(typeof node==='object'){if(node.id&&typeof node.text==='string')map.set(node.id,node.text);Object.values(node).forEach(v=>collectChunks(v,map))}
}

function renderAll(stats){
  $('#topStats').innerHTML=[['CHUNKS',stats.graph_count||state.graphs.length],['ENTITIES',stats.entity_count||sum('entities')],['RELATIONS',stats.relation_count||sum('relations')],['EVENTS',stats.event_count||sum('events')]].map(([k,v])=>`<div class="top-stat"><b>${v}</b><span>${k}</span></div>`).join('');
  $('#chunkCount').textContent=state.graphs.length; renderChunkList(); selectChunk(0);
}
function sum(k){return state.graphs.reduce((n,g)=>n+(g[k]?.length||0),0)}
function getText(g){return state.texts.get(g.metadata?.chunk_id)||[...(g.entities||[]),...(g.relations||[]),...(g.events||[])].map(x=>x.provenance).filter(Boolean).filter((x,i,a)=>a.indexOf(x)===i).join(' ')}
function renderChunkList(query=''){
  const q=query.trim().toLowerCase();
  $('#chunkList').innerHTML=state.graphs.map((g,i)=>({g,i,text:getText(g)})).filter(x=>!q||x.g.metadata?.chunk_id?.toLowerCase().includes(q)||x.text.toLowerCase().includes(q)).map(({g,i,text})=>`<button class="chunk-item ${i===state.active?'active':''}" data-index="${i}" type="button"><div class="chunk-id"><span>CHUNK ${esc(g.metadata?.chunk_id)}</span><span>${String(i+1).padStart(2,'0')}</span></div><div class="chunk-preview">${esc(text)}</div><div class="mini-counts"><span>● ${g.entities?.length||0} 实体</span><span>⌁ ${g.relations?.length||0} 关系</span><span>◆ ${g.events?.length||0} 事件</span></div></button>`).join('')||'<div class="meta-block">没有匹配的文本分段</div>';
  $$('.chunk-item').forEach(b=>b.onclick=()=>selectChunk(Number(b.dataset.index)));
}
function selectChunk(index){
  state.active=index; renderChunkList($('#chunkSearch').value); const g=state.graphs[index],m=g.metadata||{},ss=m.extra?.schema_selection||{};
  $('#documentName').textContent=m.document_id||'未知文档'; $('#chunkTitle').textContent=`Chunk ${m.chunk_id||index}`;
  $('#chunkMetrics').innerHTML=[['实体',g.entities?.length||0],['关系',g.relations?.length||0],['事件',g.events?.length||0],['Schema',ss.concepts?.length||0]].map(([k,v])=>`<div class="metric"><b>${v}</b><span>${k}</span></div>`).join('');
  renderSource(); renderDetails(); requestAnimationFrame(()=>createStableGraph(g)); window.scrollTo?.({top:0,behavior:'smooth'});
}
function renderSource(){
  const g=state.graphs[state.active],text=getText(g); let html=esc(text);
  if(state.highlight){
    const names=(g.entities||[]).map(e=>e.name).filter(Boolean).sort((a,b)=>b.length-a.length);
    names.forEach(name=>{const safe=esc(name),pattern=new RegExp(escapeRegExp(safe),'g');html=html.replace(pattern,`<mark class="entity-mark" data-entity="${esc(name)}">${safe}</mark>`)});
  }
  $('#sourceText').innerHTML=html||'<span class="fallback">未找到对应原文</span>';
  $$('.entity-mark').forEach(mark=>mark.onclick=()=>inspectByName(mark.dataset.entity));
}
function escapeRegExp(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}

function renderDetails(){
  const g=state.graphs[state.active], items=detailItems(g,state.tab), selection=g.metadata?.extra?.schema_selection||{};
  $('#detailSummary').textContent=state.tab==='schema'?`${items.length} 个候选 · 置信度 ${(selection.selection_confidence||0).toFixed(3)}`:`${items.length} 条结果`;
  const sorted=[...items].sort(state.sort==='name'?((a,b)=>String(a.name||a.schema||a.type).localeCompare(String(b.name||b.schema||b.type),'zh')):((a,b)=>confidence(b)-confidence(a)));
  let head='';
  if(state.tab==='schema')head=`<div class="schema-overview"><strong>${(selection.selection_confidence||0).toFixed(3)}</strong><p>Schema 选择置信度 · ${esc(selection.selector_version)}</p><div class="query-terms">${(selection.query_terms||[]).slice(0,16).map(x=>`<span>${esc(x)}</span>`).join('')}</div>${selection.fallback_used?'<p class="fallback">本 chunk 使用了 fallback 补全候选</p>':''}</div>`;
  $('#detailContent').innerHTML=head+sorted.map(x=>renderDetailCard(x,state.tab,g)).join('');
  $$('.detail-card[data-id]').forEach(card=>card.onclick=()=>inspectById(card.dataset.id));
}
function detailItems(g,tab){if(tab==='schema')return g.metadata?.extra?.schema_selection?.concepts||[];return g[tab]||[]}
function renderDetailCard(x,tab,g){
  if(tab==='schema')return `<article class="detail-card"><div class="card-top"><span class="card-name">${esc(x.zhName)}</span><span class="schema-code">${esc(x.schema)}</span></div><div class="card-sub">${esc(x.category)}</div><div class="score-bar"><i style="width:${Math.max(1,confidence(x)*100)}%"></i></div><div class="card-sub">综合 ${confidence(x).toFixed(3)} · 向量 ${(x.vector_score||0).toFixed(3)} · 词法 ${(x.lexical_score||0).toFixed(2)}</div><div class="reason-tags">${(x.selection_reasons||[]).map(r=>`<span>${esc(r)}</span>`).join('')}</div></article>`;
  if(tab==='entities')return `<article class="detail-card" data-id="${esc(x.id)}"><div class="card-top"><span class="card-name">${esc(x.name)}</span><span class="confidence">${Math.round(confidence(x)*100)}%</span></div><div class="card-sub">${esc(x.type_zh)} · ${esc(x.type)}</div>${attrs(x.attributes)}<div class="evidence">${esc(x.provenance)}</div></article>`;
  if(tab==='relations'){const map=new Map((g.entities||[]).map(e=>[e.id,e.name]));return `<article class="detail-card" data-id="${esc(x.id)}"><div class="card-top"><span class="card-name">${esc(x.type_zh||x.type)}</span><span class="confidence">${Math.round(confidence(x)*100)}%</span></div><div class="relation-flow"><span>${esc(map.get(x.source_id)||x.source_id)}</span><i>── ${esc(x.type)} →</i><span>${esc(map.get(x.target_id)||x.target_id)}</span></div>${attrs(x.attributes)}<div class="evidence">${esc(x.provenance)}</div></article>`}
  return `<article class="detail-card" data-id="${esc(x.id)}"><div class="card-top"><span class="card-name">${esc(x.name)}</span><span class="confidence">${Math.round(confidence(x)*100)}%</span></div><div class="card-sub">${esc(x.type)}${x.time?' · '+esc(x.time):''}${x.location?' · '+esc(x.location):''}</div>${attrs(x.attributes)}<div class="evidence">${esc(x.provenance)}</div></article>`;
}
function attrs(obj){if(!obj||!Object.keys(obj).length)return'';return `<div class="meta-block">${Object.entries(obj).map(([k,v])=>`<b>${esc(k)}</b>：${esc(typeof v==='object'?JSON.stringify(v):v)}`).join('<br>')}</div>`}

// Canvas 图谱：实现轻量力导向布局、节点拖拽、画布平移缩放与点击检查。
function createGraph(g){
  state.simulation?.stop?.(); const canvas=$('#graphCanvas'),ctx=canvas.getContext('2d'),wrap=canvas.parentElement,dpr=devicePixelRatio||1;
  canvas.width=wrap.clientWidth*dpr;canvas.height=wrap.clientHeight*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  const w=wrap.clientWidth,h=wrap.clientHeight, entityNodes=(g.entities||[]).map((e,i)=>({...e,kind:'entity',x:w/2+Math.cos(i)*100+Math.random()*40,y:h/2+Math.sin(i)*100+Math.random()*40,r:9}));
  const eventNodes=(g.events||[]).map((e,i)=>({...e,kind:'event',x:w/2+Math.cos(i*2)*150,y:h/2+Math.sin(i*2)*150,r:11}));
  const nodes=[...entityNodes,...eventNodes],map=new Map(nodes.map(n=>[n.id,n])),links=[];
  (g.relations||[]).forEach(r=>{if(map.has(r.source_id)&&map.has(r.target_id))links.push({...r,source:map.get(r.source_id),target:map.get(r.target_id),kind:'relation'})});
  eventNodes.forEach(e=>(e.participants||[]).forEach(id=>{if(map.has(id))links.push({source:e,target:map.get(id),kind:'participation',type:'参与'})}));
  let view={x:0,y:0,scale:1},drag=null,pan=null,running=true,frame=0;
  const visible=n=>state.filters[n.kind];
  function physics(){for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=nodes[i],b=nodes[j],dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy+1,force=850/d2;a.vx=(a.vx||0)-dx*force*.018;a.vy=(a.vy||0)-dy*force*.018;b.vx=(b.vx||0)+dx*force*.018;b.vy=(b.vy||0)+dy*force*.018}links.forEach(l=>{const dx=l.target.x-l.source.x,dy=l.target.y-l.source.y,d=Math.hypot(dx,dy)||1,f=(d-105)*.0015;l.source.vx=(l.source.vx||0)+dx*f;l.source.vy=(l.source.vy||0)+dy*f;l.target.vx=(l.target.vx||0)-dx*f;l.target.vy=(l.target.vy||0)-dy*f});nodes.forEach(n=>{if(n===drag)return;n.vx=((n.vx||0)+(w/2-n.x)*.00035)*.88;n.vy=((n.vy||0)+(h/2-n.y)*.00035)*.88;n.x+=n.vx;n.y+=n.vy})}
  function draw(){ctx.clearRect(0,0,w,h);ctx.save();ctx.translate(view.x,view.y);ctx.scale(view.scale,view.scale);links.forEach(l=>{if(!visible(l.source)||!visible(l.target))return;ctx.beginPath();ctx.moveTo(l.source.x,l.source.y);ctx.lineTo(l.target.x,l.target.y);ctx.strokeStyle=l.kind==='relation'?'rgba(88,214,199,.33)':'rgba(241,164,91,.22)';ctx.lineWidth=1;ctx.stroke();const mx=(l.source.x+l.target.x)/2,my=(l.source.y+l.target.y)/2;ctx.fillStyle='#65747a';ctx.font='9px Microsoft YaHei';ctx.fillText(l.type_zh||l.type||'',mx+3,my-3)});nodes.forEach(n=>{if(!visible(n))return;ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fillStyle=n.kind==='event'?'#f1a45b':'#58d6c7';ctx.shadowColor=ctx.fillStyle;ctx.shadowBlur=10;ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#dce5e5';ctx.font=n.kind==='event'?'11px Microsoft YaHei':'10px Microsoft YaHei';ctx.textAlign='center';ctx.fillText(truncate(n.name,12),n.x,n.y+n.r+15)});ctx.restore();if(running&&frame++<220)physics();requestAnimationFrame(draw)}
  function point(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left-view.x)/view.scale,y:(e.clientY-r.top-view.y)/view.scale}}
  function hit(e){const p=point(e);return [...nodes].reverse().find(n=>visible(n)&&Math.hypot(n.x-p.x,n.y-p.y)<n.r+7)}
  canvas.onpointerdown=e=>{const n=hit(e);if(n){drag=n;canvas.setPointerCapture(e.pointerId)}else pan={x:e.clientX-view.x,y:e.clientY-view.y}};
  canvas.onpointermove=e=>{if(drag){const p=point(e);drag.x=p.x;drag.y=p.y;drag.vx=drag.vy=0;running=true;frame=0}else if(pan){view.x=e.clientX-pan.x;view.y=e.clientY-pan.y}};
  canvas.onpointerup=e=>{const n=drag||hit(e);drag=null;pan=null;if(n)showInspector(n)};
  canvas.onwheel=e=>{e.preventDefault();const old=view.scale,next=Math.max(.45,Math.min(2.5,old*(e.deltaY>0?.9:1.1))),r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;view.x=mx-(mx-view.x)*next/old;view.y=my-(my-view.y)*next/old;view.scale=next};
  state.simulation={stop(){running=false},reset(){view={x:0,y:0,scale:1};nodes.forEach((n,i)=>{n.x=w/2+Math.cos(i)*120;n.y=h/2+Math.sin(i)*120});running=true;frame=0},nodes};draw();
}
function showInspector(item){const el=$('#nodeInspector');el.classList.add('show');el.innerHTML=`<div class="type">${esc(item.kind==='event'?'事件 · '+item.type:(item.type_zh||item.type))}</div><h3>${esc(item.name)}</h3>${item.time?`<p>${esc(item.time)}${item.location?' · '+esc(item.location):''}</p>`:''}${attrs(item.attributes)}<dl><dt>置信度</dt><dd>${Math.round(confidence(item)*100)}%</dd><dt>证据</dt><dd>${esc(item.provenance)}</dd></dl>`}
function inspectByName(name){const node=state.simulation?.nodes.find(n=>n.name===name);if(node)showInspector(node)}
function inspectById(id){const g=state.graphs[state.active],x=[...(g.entities||[]),...(g.events||[]),...(g.relations||[])].find(v=>v.id===id);if(x)showInspector({...x,kind:(g.events||[]).includes(x)?'event':'entity'})}

// 稳定图谱布局：使用确定性的双环布局代替持续力模拟，确保所有浏览器都能立即绘出首帧。
function createStableGraph(g){
  state.simulation?.stop?.();
  const canvas=$('#graphCanvas'),wrap=canvas.parentElement,ctx=canvas.getContext('2d');
  const dpr=Math.max(1,window.devicePixelRatio||1),w=Math.max(320,wrap.clientWidth),h=Math.max(300,wrap.clientHeight);
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);canvas.style.width=`${w}px`;canvas.style.height=`${h}px`;
  const entities=(g.entities||[]).map(e=>({...e,kind:'entity',r:9}));
  const events=(g.events||[]).map(e=>({...e,kind:'event',r:12}));
  const nodes=[...entities,...events],map=new Map(nodes.map(n=>[n.id,n])),links=[];
  (g.relations||[]).forEach(r=>{if(map.has(r.source_id)&&map.has(r.target_id))links.push({...r,source:map.get(r.source_id),target:map.get(r.target_id),kind:'relation'})});
  events.forEach(event=>(event.participants||[]).forEach(id=>{if(map.has(id))links.push({source:event,target:map.get(id),kind:'participation',type_zh:'参与'})}));
  const cx=w/2,cy=h/2,entityRadius=Math.min(w,h)*.29,eventRadius=Math.min(w,h)*.43;
  entities.forEach((n,i)=>{const a=-Math.PI/2+i*Math.PI*2/Math.max(entities.length,1);n.x=cx+Math.cos(a)*entityRadius;n.y=cy+Math.sin(a)*entityRadius});
  events.forEach((n,i)=>{const a=-Math.PI/2+(i+.5)*Math.PI*2/Math.max(events.length,1);n.x=cx+Math.cos(a)*eventRadius;n.y=cy+Math.sin(a)*eventRadius});
  let view={x:0,y:0,scale:1},drag=null,pan=null;
  const visible=n=>state.filters[n.kind]!==false;
  function lineLabel(link){return link.type_zh||link.type||''}
  function draw(){
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.save();ctx.translate(view.x,view.y);ctx.scale(view.scale,view.scale);
    links.forEach(l=>{if(!visible(l.source)||!visible(l.target))return;const dx=l.target.x-l.source.x,dy=l.target.y-l.source.y,angle=Math.atan2(dy,dx);ctx.beginPath();ctx.moveTo(l.source.x,l.source.y);ctx.lineTo(l.target.x,l.target.y);ctx.strokeStyle=l.kind==='relation'?'rgba(88,214,199,.42)':'rgba(241,164,91,.26)';ctx.lineWidth=1;ctx.stroke();ctx.beginPath();ctx.moveTo(l.target.x-Math.cos(angle)*11,l.target.y-Math.sin(angle)*11);ctx.lineTo(l.target.x-Math.cos(angle-.45)*17,l.target.y-Math.sin(angle-.45)*17);ctx.lineTo(l.target.x-Math.cos(angle+.45)*17,l.target.y-Math.sin(angle+.45)*17);ctx.closePath();ctx.fillStyle=l.kind==='relation'?'rgba(88,214,199,.52)':'rgba(241,164,91,.36)';ctx.fill();const mx=(l.source.x+l.target.x)/2,my=(l.source.y+l.target.y)/2;ctx.font='9px Microsoft YaHei';ctx.textAlign='center';ctx.fillStyle='#74858b';ctx.fillText(lineLabel(l),mx,my-5)});
    nodes.forEach(n=>{if(!visible(n))return;ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fillStyle=n.kind==='event'?'#f1a45b':'#58d6c7';ctx.shadowColor=ctx.fillStyle;ctx.shadowBlur=12;ctx.fill();ctx.shadowBlur=0;ctx.beginPath();ctx.arc(n.x,n.y,n.r+4,0,Math.PI*2);ctx.strokeStyle=n.kind==='event'?'rgba(241,164,91,.35)':'rgba(88,214,199,.35)';ctx.stroke();ctx.fillStyle='#dce5e5';ctx.font=n.kind==='event'?'11px Microsoft YaHei':'10px Microsoft YaHei';ctx.textAlign='center';ctx.fillText(truncate(n.name,12),n.x,n.y+n.r+17)});
    ctx.restore();
  }
  function localPoint(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left-view.x)/view.scale,y:(e.clientY-r.top-view.y)/view.scale}}
  function hit(e){const p=localPoint(e);return [...nodes].reverse().find(n=>visible(n)&&Math.hypot(n.x-p.x,n.y-p.y)<n.r+8)}
  canvas.onpointerdown=e=>{const n=hit(e);if(n){drag=n;canvas.setPointerCapture(e.pointerId)}else pan={x:e.clientX-view.x,y:e.clientY-view.y}};
  canvas.onpointermove=e=>{if(drag){const p=localPoint(e);drag.x=p.x;drag.y=p.y;draw()}else if(pan){view.x=e.clientX-pan.x;view.y=e.clientY-pan.y;draw()}};
  canvas.onpointerup=e=>{const n=drag||hit(e);drag=null;pan=null;if(n)showInspector(n)};
  canvas.onpointercancel=()=>{drag=null;pan=null};
  canvas.onwheel=e=>{e.preventDefault();const old=view.scale,next=Math.max(.45,Math.min(2.5,old*(e.deltaY>0?.9:1.1))),r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;view.x=mx-(mx-view.x)*next/old;view.y=my-(my-view.y)*next/old;view.scale=next;draw()};
  state.simulation={nodes,stop(){},redraw:draw,reset(){view={x:0,y:0,scale:1};entities.forEach((n,i)=>{const a=-Math.PI/2+i*Math.PI*2/Math.max(entities.length,1);n.x=cx+Math.cos(a)*entityRadius;n.y=cy+Math.sin(a)*entityRadius});events.forEach((n,i)=>{const a=-Math.PI/2+(i+.5)*Math.PI*2/Math.max(events.length,1);n.x=cx+Math.cos(a)*eventRadius;n.y=cy+Math.sin(a)*eventRadius});draw()}};
  draw();
}

$('#chunkSearch').oninput=e=>renderChunkList(e.target.value);
$('#highlightToggle').onclick=e=>{state.highlight=!state.highlight;e.currentTarget.classList.toggle('active',state.highlight);renderSource()};
$('#copyText').onclick=async e=>{await navigator.clipboard.writeText(getText(state.graphs[state.active]));const old=e.currentTarget.textContent;e.currentTarget.textContent='已复制';setTimeout(()=>e.currentTarget.textContent=old,1200)};
$$('.tab').forEach(t=>t.onclick=()=>{$$('.tab').forEach(x=>x.classList.toggle('active',x===t));state.tab=t.dataset.tab;renderDetails()});
$('#sortSelect').onchange=e=>{state.sort=e.target.value;renderDetails()};
$$('.filter-chip').forEach(b=>b.onclick=()=>{state.filters[b.dataset.kind]=!state.filters[b.dataset.kind];b.classList.toggle('active',state.filters[b.dataset.kind]);state.simulation?.redraw?.()});
$('#resetGraph').onclick=()=>{state.simulation?.reset();$('#nodeInspector').classList.remove('show')};
$('#reloadButton').onclick=loadData;$('#retryButton').onclick=loadData;window.addEventListener('resize',()=>state.graphs.length&&createStableGraph(state.graphs[state.active]));
loadData();
