const $=x=>document.getElementById(x);
let running=false,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,lastDecision=null;

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

// RESEARCH V14 · PPM-C (Prediction by Partial Matching)
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

 for(let order=0;order<=PPM_MAX_ORDER;order++){
  const st=contextStats(seq,order);
  if(st.total===0)continue;

  const seen=[],unseen=[];
  for(let d=0;d<PPM_K;d++){
   if(st.counts[d]>0)seen.push(d);
   else unseen.push(d);
  }

  let next=Array(PPM_K).fill(0);

  // PPM-C Method C:
  // seen symbol c/(N+U), escape U/(N+U), then back off for unseen symbols.
  if(unseen.length===0){
   for(let d=0;d<PPM_K;d++)next[d]=st.counts[d]/st.total;
   escape=0;
  }else{
   const den=st.total+st.distinct;
   const esc=st.distinct/den;
   for(const d of seen)next[d]=st.counts[d]/den;

   let backoffMass=0;
   for(const d of unseen)backoffMass+=p[d];
   if(backoffMass<=0){
    for(const d of unseen)next[d]=esc/unseen.length;
   }else{
    for(const d of unseen)next[d]=esc*(p[d]/backoffMass);
   }
   escape=esc;
  }

  p=normalize(next);
  deepest=order;
  support=st.total;
  distinct=st.distinct;
 }
 return{p,deepest,support,escape,distinct};
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

 // Solo diagnóstico visual; NO filtra la compra.
 const quality=100*clamp01((.105-q.pt)/.065);

 return{
  q,
  second,
  spread,
  quality,
  entropy:H,
  entropyNorm,
  order:model.deepest,
  support:model.support,
  escape:model.escape,
  distinct:model.distinct
 };
}

function resetCycle(){
 lastDecision=null;
 if($('evidenceCandidates'))$('evidenceCandidates').textContent='V14 · esperando siguiente predicción PPM-C.';
}

function showSignal(s){
 lastSignal=s;
 const buy=$('buy');

 if(!s){
  $('decision').textContent=running?'V14 · CALIBRANDO PPM-C':'AUTO DETENIDO';
  $('reason').textContent=running?'Construyendo contextos parciales.':'Pulsa INICIAR AUTO para comenzar.';
  if($('sepPick'))$('sepPick').textContent='—';
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='V14 · esperando historial mínimo.';
  buy.textContent=running?'AUTO V14 · CALIBRANDO':'INICIAR AUTO';
  buy.disabled=running;
  return;
 }

 $('risk').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+s.q.d;
 $('spread').textContent=(s.spread*100).toFixed(1);
 $('entropy').textContent=s.entropy.toFixed(2);
 $('phase').textContent='ORDEN '+s.order;
 $('meter').style.width=Math.min(100,s.quality)+'%';

 if($('finalScore'))$('finalScore').textContent=s.quality.toFixed(0)+'/100';
 if($('delayState'))$('delayState').textContent='T+1';
 if($('regimeState'))$('regimeState').textContent='CTX '+s.order+' · N'+s.support;
 if($('techScore'))$('techScore').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('techRsi'))$('techRsi').textContent=(s.second.pt*100).toFixed(1)+'%';
 if($('techMacd'))$('techMacd').textContent=(s.escape*100).toFixed(1)+'%';
 if($('techTrend'))$('techTrend').textContent='U '+s.distinct;

 if(pending){
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='OPERACIÓN EN CURSO · PPM-C sigue recalculando';
  $('decision').textContent='OPERACIÓN EN CURSO · DIFFER D'+pending.d;
  $('reason').textContent='La compra automática espera la liquidación antes de abrir otra.';
  buy.textContent='OPERACIÓN EN CURSO';
  buy.disabled=true;
  return;
 }

 if($('evidenceCandidates')){
  $('evidenceCandidates').textContent=
   'PPM-C T+1 · DIFFER D'+s.q.d+
   ' · P '+(s.q.pt*100).toFixed(1)+'%'+
   ' · ORDEN '+s.order+
   ' · SOPORTE '+s.support+
   ' · SIN FILTROS';
 }

 if(!running){
  $('decision').textContent='V14 · CANDIDATO D'+s.q.d;
  $('reason').textContent='PPM-C variable-order listo · AUTO detenido.';
  buy.textContent='INICIAR AUTO';
  buy.disabled=false;
 }else{
  $('decision').textContent='V14 · COMPRA D'+s.q.d;
  $('reason').textContent='PPM-C eligió el dígito de menor probabilidad T+1 usando contexto actual.';
  buy.textContent='AUTO · COMPRA D'+s.q.d;
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
 pending=null;observe=0;
 resetCycle();
 $('status').textContent='ERROR · V14 SIGUE OBSERVANDO';
 ui();
}

function finish(profit,label){
 profit=Number(profit);
 if(!Number.isFinite(profit)){tradeError(new Error('Resultado inválido'));return}
 pnl+=profit;
 if(profit>0){
  wins++;
  stake=Math.max(baseStake(),stake+profit);
  log('WIN '+label+' +$'+profit.toFixed(2));
 }else{
  losses++;
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
  $('status').textContent='V14 · BUSCANDO SIGUIENTE CONTEXTO';
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

 // Sin filtros: cada nueva evaluación PPM-C se compra cuando no hay operación abierta.
 if(running&&!pending&&s){
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
 lastPick=null;lastSignal=null;lastDecision=null;running=true;

 $('status').textContent='V14 ACTIVA · PPM-C VARIABLE-ORDER · AUTO T+1';
 $('buy').textContent='AUTO V14 · ACTIVO';
 $('buy').disabled=true;
 log('V14 PPM-C INICIADA '+mode+' · STAKE $'+stake.toFixed(2)+' · META STOP $'+target().toFixed(2));
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
