const $=x=>document.getElementById(x);
let running=false,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,lastDecision=null,activeLearning=null,learningMemory={},learningSeq=0;

const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const clamp01=x=>Math.max(0,Math.min(1,x));
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function normalize(a){
 let z=a.reduce((s,x)=>s+x,0);
 if(!Number.isFinite(z)||z<=0)return Array(a.length).fill(1/a.length);
 return a.map(x=>x/z);
}

// RESEARCH V16 · PPM-C (Prediction by Partial Matching)
// Contextos variables de orden 0..6 con backoff/escape.
// Horizonte exacto T+1.
// Sin HMM, Bayes, safe gate, conteos 1/2/3, bloqueos o penalizaciones heredadas.
const PPM_K=10;
const PPM_MAX_ORDER=6;
const PPM_WINDOW=360;

function contextStats(seq,order){
 const n=seq.length;
 const counts=Array(PPM_K).fill(0);
 if(order===0){
  for(let i=0;i<n;i++)counts[seq[i]]++;
 }else{
  if(n<=order)return{counts,total:0,distinct:0};
  const ctx=seq.slice(n-order);
  for(let t=order;t<n;t++){
   let ok=true;
   for(let k=0;k<order;k++){
    if(seq[t-order+k]!==ctx[k]){ok=false;break}
   }
   if(ok)counts[seq[t]]++;
  }
 }
 const total=counts.reduce((a,b)=>a+b,0);
 const distinct=counts.reduce((s,c)=>s+(c>0?1:0),0);
 return{counts,total,distinct};
}

function ppmCDistribution(seq){
 let p=Array(PPM_K).fill(1/PPM_K);
 let deepest=0,support=0,escape=1,distinct=0;
 const levels=[];

 for(let order=0;order<=PPM_MAX_ORDER;order++){
  const st=contextStats(seq,order);
  if(st.total===0)continue;

  const seen=[],unseen=[];
  for(let d=0;d<PPM_K;d++){
   if(st.counts[d]>0)seen.push(d);
   else unseen.push(d);
  }

  let next=Array(PPM_K).fill(0);
  let esc=0;

  if(unseen.length===0){
   for(let d=0;d<PPM_K;d++)next[d]=st.counts[d]/st.total;
  }else{
   const den=st.total+st.distinct;
   esc=st.distinct/den;
   for(const d of seen)next[d]=st.counts[d]/den;

   let backoffMass=0;
   for(const d of unseen)backoffMass+=p[d];
   if(backoffMass<=0){
    for(const d of unseen)next[d]=esc/unseen.length;
   }else{
    for(const d of unseen)next[d]=esc*(p[d]/backoffMass);
   }
  }

  p=normalize(next);
  levels.push({order,p:p.slice(),support:st.total,distinct:st.distinct,escape:esc});
  deepest=order;
  support=st.total;
  distinct=st.distinct;
  escape=esc;
 }
 return{p,deepest,support,escape,distinct,levels};
}

function contextHash(seq){
 let h=2166136261>>>0;
 const tail=seq.slice(-10);
 for(const d of tail){
  h^=(d+17);
  h=Math.imul(h,16777619)>>>0;
 }
 return h>>>0;
}

const LEARN_KEY='quantum_ppmc_learning_v16';

function loadLearning(){
 try{
  const raw=localStorage.getItem(LEARN_KEY);
  if(!raw)return;
  const obj=JSON.parse(raw);
  if(obj&&typeof obj==='object'){
   learningMemory=obj.memory&&typeof obj.memory==='object'?obj.memory:{};
   learningSeq=Number(obj.seq)||0;
  }
 }catch(e){}
}
function saveLearning(){
 try{
  const entries=Object.entries(learningMemory);
  if(entries.length>700){
   entries.sort((a,b)=>(b[1]?.last||0)-(a[1]?.last||0));
   learningMemory=Object.fromEntries(entries.slice(0,700));
  }
  localStorage.setItem(LEARN_KEY,JSON.stringify({seq:learningSeq,memory:learningMemory}));
 }catch(e){}
}
function decayedNode(key){
 const raw=learningMemory[key]||{m:0,w:0,last:learningSeq};
 const age=Math.max(0,learningSeq-(raw.last||0));
 const decay=Math.pow(.992,age);
 return{
  m:(Number(raw.m)||0)*decay,
  w:(Number(raw.w)||0)*decay,
  last:learningSeq
 };
}
function contextLearningKeys(s){
 const keys=[];
 const d=s.q.d;
 for(let k=1;k<=4;k++){
  if(hist.length>=k){
   keys.push({key:'S'+k+'|D'+d+'|'+hist.slice(-k).join(''),weight:k});
  }
 }
 const ob=Math.min(6,Math.max(0,Number(s.order)||0));
 const ab=Math.min(4,Math.floor((Number(s.agreement)||0)*5));
 const eb=Math.min(4,Math.floor((Number(s.escape)||0)*5));
 keys.push({key:'F|D'+d+'|O'+ob+'|A'+ab+'|E'+eb,weight:3});
 return keys;
}
function learnedContext(s){
 const keys=contextLearningKeys(s);
 let wr=0,ws=0,evidence=0;

 for(const item of keys){
  const n=decayedNode(item.key);
  const samples=n.m+n.w;
  // Beta prior centered at the neutral 10% MATCH rate.
  const posteriorMatch=(1+n.m)/(10+samples);
  const reliability=samples/(samples+1.5);
  const weight=item.weight*(.25+.75*reliability);
  wr+=posteriorMatch*weight;
  ws+=weight;
  evidence+=samples*item.weight;
 }

 const risk=ws>0?wr/ws:.10;
 const ev=keys.length?evidence/keys.reduce((s,x)=>s+x.weight,0):0;
 const reliability=ev/(ev+1.5);
 const excess=Math.max(0,risk-.10);

 // A bad learned context makes the current PPM-C signal work harder.
 // Wins naturally pull the posterior back down.
 const penaltyPoints=100*clamp01(excess/.12)*.30*reliability;
 return{risk,evidence:ev,reliability,penaltyPoints,keys};
}
function buildLearningSnapshot(s){
 const L=learnedContext(s);
 return{
  digit:s.q.d,
  keys:L.keys.map(x=>x.key),
  order:s.order,
  agreement:s.agreement,
  escape:s.escape,
  probability:s.q.pt,
  confidence:s.baseConfidence??s.quality
 };
}
function learnFromSettlement(win){
 if(!activeLearning)return;
 learningSeq++;

 for(const key of activeLearning.keys){
  const n=decayedNode(key);
  if(win)n.w+=1;
  else n.m+=1;
  n.last=learningSeq;
  learningMemory[key]=n;
 }

 const tag=win?'WIN':'MATCH';
 log('LEARN '+tag+' · D'+activeLearning.digit+' · '+activeLearning.keys.length+' huellas actualizadas');
 activeLearning=null;
 saveLearning();
}

function analyse(){
 if(hist.length<80)return null;

 const seq=hist.slice(-PPM_WINDOW);
 const model=ppmCDistribution(seq);
 const hash=contextHash(seq);

 const rows=model.p.map((p,d)=>({
  d,
  pt:p,
  risk:p,
  tie:(d+hash)%PPM_K
 })).sort((a,b)=>{
  const delta=a.pt-b.pt;
  if(Math.abs(delta)>1e-12)return delta;
  return a.tie-b.tie;
 });

 const q=rows[0],second=rows[1];
 const spread=Math.max(0,second.pt-q.pt);

 let H=0;
 for(const x of model.p)if(x>0)H-=x*Math.log2(x);
 const entropyNorm=H/Math.log2(PPM_K);

 const usable=model.levels.filter(x=>x.support>=2);
 let agree=0,nearAgree=0;
 for(const lv of usable){
  const ranked=lv.p.map((p,d)=>({p,d})).sort((a,b)=>a.p-b.p||a.d-b.d);
  if(ranked[0].d===q.d)agree++;
  if(ranked.slice(0,2).some(x=>x.d===q.d))nearAgree++;
 }
 const agreement=usable.length?agree/usable.length:0;
 const nearAgreement=usable.length?nearAgree/usable.length:0;

 const edge=clamp01((.10-q.pt)/.035);
 const separation=clamp01(spread/.012);
 const supportScore=clamp01(model.support/8);
 const escapeCertainty=clamp01(1-model.escape);

 const baseConfidence=100*(
  .34*edge+
  .24*separation+
  .20*agreement+
  .10*nearAgreement+
  .07*supportScore+
  .05*escapeCertainty
 );

 const temp={
  q,second,spread,baseConfidence,quality:baseConfidence,
  entropy:H,entropyNorm,
  order:model.deepest,support:model.support,escape:model.escape,distinct:model.distinct,
  agreement,nearAgreement,levels:usable.length
 };

 const learned=learnedContext(temp);
 const confidence=Math.max(0,baseConfidence-learned.penaltyPoints);

 // Same PPM-C confirmation, now adjusted by what happened in similar past entries.
 // One MATCH raises the bar; repeated MATCHes raise it more.
 // WINs pull the learned context risk back toward/below neutral.
 const learnedCaution=
  learned.evidence>=.8&&learned.risk>=.155;

 const buySignal=
  confidence>=46&&
  q.pt<.097&&
  spread>=.0015&&
  (agreement>=.34||nearAgreement>=.60)&&
  (model.support>=2||q.pt<.078)&&
  (!learnedCaution||confidence>=58);

 return{
  ...temp,
  quality:confidence,
  baseConfidence,
  buySignal,
  learnedRisk:learned.risk,
  learnedEvidence:learned.evidence,
  learnedPenalty:learned.penaltyPoints,
  learnedCaution
 };
}
function resetCycle(){
 lastDecision=null;
 if($('evidenceCandidates'))$('evidenceCandidates').textContent='V16 · esperando señal PPM-C confirmada.';
}

function showSignal(s){
 lastSignal=s;
 const buy=$('buy');

 if(!s){
  $('decision').textContent=running?'V16 · CALIBRANDO PPM-C':'AUTO DETENIDO';
  $('reason').textContent=running?'Construyendo contextos y memoria de resultados.':'Pulsa INICIAR AUTO para comenzar.';
  if($('sepPick'))$('sepPick').textContent='—';
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='V16 · esperando historial mínimo.';
  buy.textContent=running?'AUTO V16 · CALIBRANDO':'INICIAR AUTO';
  buy.disabled=running;
  return;
 }

 $('risk').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+s.q.d;
 $('spread').textContent=(s.spread*100).toFixed(2);
 $('entropy').textContent=s.entropy.toFixed(2);
 $('phase').textContent=s.learnedCaution?'MEMORIA ALERTA':'CTX '+s.order;
 $('meter').style.width=Math.min(100,s.quality)+'%';

 if($('finalScore'))$('finalScore').textContent=s.quality.toFixed(0)+'/100';
 if($('delayState'))$('delayState').textContent='T+1';
 if($('regimeState'))$('regimeState').textContent='MEM '+(s.learnedRisk*100).toFixed(1)+'%';
 if($('techScore'))$('techScore').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('techRsi'))$('techRsi').textContent=(s.second.pt*100).toFixed(1)+'%';
 if($('techMacd'))$('techMacd').textContent='PEN '+s.learnedPenalty.toFixed(1);
 if($('techTrend'))$('techTrend').textContent='EXP '+s.learnedEvidence.toFixed(1);

 if(pending){
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='OPERACIÓN EN CURSO · guardando contexto para aprender del resultado';
  $('decision').textContent='OPERACIÓN EN CURSO · DIFFER D'+pending.d;
  $('reason').textContent='Al terminar WIN/MATCH se actualizará la memoria del contexto.';
  buy.textContent='OPERACIÓN EN CURSO';
  buy.disabled=true;
  return;
 }

 if($('evidenceCandidates')){
  $('evidenceCandidates').textContent=
   'PPM-C · D'+s.q.d+
   ' · CONF BASE '+s.baseConfidence.toFixed(0)+
   ' · CONF APRENDIDA '+s.quality.toFixed(0)+
   ' · MATCH MEM '+(s.learnedRisk*100).toFixed(1)+'%'+
   ' · '+(s.buySignal?'COMPRAR':'ESPERAR');
 }

 if(!running){
  $('decision').textContent='V16 · CANDIDATO D'+s.q.d;
  $('reason').textContent='PPM-C + aprendizaje por resultados listo · AUTO detenido.';
  buy.textContent='INICIAR AUTO';
  buy.disabled=false;
 }else if(s.buySignal){
  $('decision').textContent='V16 · COMPRA D'+s.q.d;
  $('reason').textContent='PPM-C confirma la señal y la memoria de contextos no la invalida.';
  buy.textContent='AUTO · COMPRA D'+s.q.d;
  buy.disabled=true;
 }else{
  $('decision').textContent='V16 · ESPERANDO';
  $('reason').textContent=s.learnedCaution
   ?'Contextos parecidos tuvieron MATCH; esta entrada necesita evidencia mucho más fuerte.'
   :'La señal actual todavía no justifica comprar.';
  buy.textContent='AUTO · ESPERANDO SEÑAL';
  buy.disabled=true;
 }
}
function ui(d){
 if(d!==undefined)$('tick').textContent='D'+d;
 $('pnl').textContent=(pnl>=0?'+':'')+'$'+pnl.toFixed(2);
 $('stake').textContent='$'+stake.toFixed(2);
 $('wins').textContent=wins;
 $('losses').textContent=losses;
 $('ops').textContent=ops;
 $('pick').textContent=lastPick===null?'—':'D'+lastPick;
}

// COMPRA AUTOMÁTICA: función preservada exactamente.
function enter(s){
 if(!running||pending||!s)return;
 const d=s.q.d,mode=$('mode').value;
 if(mode==='DEMO'&&!window.demoReady){
  $('status').textContent='CONECTA DEMO DERIV PRIMERO';
  return;
 }
 lastPick=d;observe=0;pending={d,stake,mode};ops++;
 $('decision').textContent='AUTO '+mode+' · DIFFER D'+d;
 $('reason').textContent='V9 · evidencia bayesiana · $'+stake.toFixed(2)+' · duración 1 tick';
 log('V9 BOCPD · COMPRA D'+d+' · Q'+s.quality.toFixed(0)+' · $'+stake.toFixed(2));
 ui();

 if(mode==='DEMO'){
  $('status').textContent='AUTO · ENVIANDO A DERIV…';
  window.sendDemoTrade(d,stake).catch(e=>tradeError(e));
 }else{
  $('status').textContent='AUTO SIM ABIERTA';
 }
}

function tradeError(e){
 log('ERROR DERIV '+(e?.message||e));
 activeLearning=null;
 pending=null;observe=0;
 resetCycle();
 $('status').textContent='ERROR · V16 SIGUE OBSERVANDO';
 ui();
}

function finish(profit,label){
 profit=Number(profit);
 if(!Number.isFinite(profit)){tradeError(new Error('Resultado inválido'));return}
 pnl+=profit;
 learnFromSettlement(profit>0);
 if(profit>0){
  wins++;
  stake=Math.max(baseStake(),stake+profit);
  log('WIN '+label+' +$'+profit.toFixed(2));
 }else{
  losses++;
  loadLearning();
stake=baseStake();
  log('MATCH '+label+' $'+profit.toFixed(2));
 }

 pending=null;observe=0;
 resetCycle();

 if(pnl>=target()){
  running=false;
  $('status').textContent='META +$'+target().toFixed(2)+' · STOP';
  $('phase').textContent='META';
  $('buy').textContent='INICIAR AUTO';
  $('buy').disabled=false;
 }else if(running){
  $('status').textContent='V16 · BUSCANDO SIGUIENTE CONTEXTO';
 }
 ui();
}

function tick(d){
 if(pending&&pending.mode==='SIM'){
  const p=pending;
  finish(d===p.d?-p.stake:p.stake*.10,'SIM');
 }

 hist.push(d);
 if(hist.length>1000)hist.shift();
 if(running&&!pending)observe++;
 ui(d);

 const s=analyse();
 lastDecision=s;
 showSignal(s);

 if(running&&!pending&&s&&s.buySignal){
  activeLearning=buildLearningSnapshot(s);
  enter(s);
 }
}
// CONEXIÓN PÚBLICA DE TICKS: función preservada exactamente.
function connect(){
 clearTimeout(retry);
 ws=new WebSocket('wss://api.derivws.com/trading/v1/options/ws/public');
 ws.onopen=()=>ws.send(JSON.stringify({ticks_history:'R_75',count:300,end:'latest',style:'ticks'}));
 ws.onmessage=e=>{
  const m=JSON.parse(e.data);
  if(m.history&&m.history.prices){
   const p=Number(m.pip_size||4);
   hist=m.history.prices.map(x=>Number(Number(x).toFixed(p).slice(-1)));
   ws.send(JSON.stringify({ticks:'R_75',subscribe:1}));
   $('status').textContent='LISTO · '+hist.length+' TICKS';
  }
  if(m.tick){
   const ep=+m.tick.epoch;
   if(ep===lastEpoch)return;
   lastEpoch=ep;
   const p=Number(m.tick.pip_size||4);
   const d=Number(Number(m.tick.quote).toFixed(p).slice(-1));
   tick(d);
  }
 };
 ws.onclose=()=>retry=setTimeout(connect,2500);
}

function startAuto(){
 const mode=$('mode').value;
 if(mode==='DEMO'&&!window.demoReady){
  $('status').textContent='CONECTA DEMO DERIV PRIMERO';
  return;
 }

 pnl=0;stake=baseStake();wins=0;losses=0;ops=0;pending=null;observe=0;
 lastPick=null;lastSignal=null;lastDecision=null;activeLearning=null;running=true;

 $('status').textContent='V16 ACTIVA · PPM-C + APRENDIZAJE DE RESULTADOS · T+1';
 $('buy').textContent='AUTO V16 · BUSCANDO SEÑAL';
 $('buy').disabled=true;
 log('V16 PPM-C INICIADA '+mode+' · STAKE $'+stake.toFixed(2)+' · META STOP $'+target().toFixed(2));
 ui();
}

$('start').onclick=startAuto;
$('stop').onclick=()=>{
 running=false;
 $('status').textContent='AUTO DETENIDO';
 $('buy').textContent='INICIAR AUTO';
 $('buy').disabled=false;
};
$('buy').onclick=()=>{
 if(running){
  $('status').textContent='AUTO YA ESTÁ ACTIVO';
  return;
 }
 startAuto();
};

window.demoSettlement=p=>finish(p,'DERIV '+($('mode').value||''));
window.demoTradeError=tradeError;

stake=baseStake();
$('status').textContent='AUTO DETENIDO · CONECTANDO TICKS';
$('buy').textContent='INICIAR AUTO';
$('buy').disabled=false;
ui();
connect();
