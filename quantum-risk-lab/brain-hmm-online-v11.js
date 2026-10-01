const $=x=>document.getElementById(x);
let running=false,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,candidateHistory=[],lastDecision=null,hmm=null;

const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const clamp01=x=>Math.max(0,Math.min(1,x));
function log(s){$('log').textContent=s+'\n'+$('log').textContent}

// RESEARCH V11 · HMM ADAPTATIVO
// Tres estados ocultos y diez emisiones categóricas.
// Forward filtering + actualización online de transiciones y emisiones.
// Predicción directa T+1. Sin safe gate ni filtros heredados.

const HMM_S=3,HMM_K=10;
function normalizeRow(a){
 let z=a.reduce((s,x)=>s+x,0);
 if(!Number.isFinite(z)||z<=0)z=1;
 return a.map(x=>x/z);
}
function stateEntropy(a){
 let h=0;
 for(const p of a)if(p>0)h-=p*Math.log(p);
 return h/Math.log(HMM_S);
}
function initHMM(){
 const seq=hist.slice(-300);
 const T=seq.length;
 let A=Array.from({length:HMM_S},(_,i)=>
  Array.from({length:HMM_S},(_,j)=>(i===j ? .84 : .08))
 );
 let B=Array.from({length:HMM_S},()=>Array(HMM_K).fill(.5));

 // Inicialización basada en tres tramos temporales distintos del historial.
 for(let s=0;s<HMM_S;s++){
  const lo=Math.floor(T*s/HMM_S);
  const hi=Math.floor(T*(s+1)/HMM_S);
  for(let t=lo;t<hi;t++)B[s][seq[t]]++;
  B[s]=normalizeRow(B[s]);
 }

 let alpha=Array(HMM_S).fill(1/HMM_S);

 // Forward pass sobre el historial para estimar el estado oculto actual.
 for(const obs of seq){
  const pred=Array(HMM_S).fill(0);
  for(let j=0;j<HMM_S;j++)for(let i=0;i<HMM_S;i++)pred[j]+=alpha[i]*A[i][j];
  const next=Array(HMM_S).fill(0);
  for(let j=0;j<HMM_S;j++)next[j]=pred[j]*B[j][obs];
  alpha=normalizeRow(next);
 }

 hmm={
  A,
  B,
  alpha,
  transCounts:A.map(r=>r.map(x=>x*35)),
  emitCounts:B.map(r=>r.map(x=>x*35)),
  surprise:0,
  updates:0
 };
}
function updateHMM(obs){
 if(!hmm)return;

 const prev=hmm.alpha.slice();
 const pred=Array(HMM_S).fill(0);
 for(let j=0;j<HMM_S;j++)for(let i=0;i<HMM_S;i++)pred[j]+=prev[i]*hmm.A[i][j];

 const gamma=Array(HMM_S).fill(0);
 let like=0;
 for(let j=0;j<HMM_S;j++){
  gamma[j]=pred[j]*hmm.B[j][obs];
  like+=gamma[j];
 }
 like=Math.max(like,1e-12);
 for(let j=0;j<HMM_S;j++)gamma[j]/=like;

 const xi=Array.from({length:HMM_S},()=>Array(HMM_S).fill(0));
 let z=0;
 for(let i=0;i<HMM_S;i++)for(let j=0;j<HMM_S;j++){
  xi[i][j]=prev[i]*hmm.A[i][j]*hmm.B[j][obs];
  z+=xi[i][j];
 }
 z=Math.max(z,1e-12);
 for(let i=0;i<HMM_S;i++)for(let j=0;j<HMM_S;j++)xi[i][j]/=z;

 // Memoria adaptativa: conserva historia pero permite cambiar de régimen.
 const forget=.985;
 for(let i=0;i<HMM_S;i++){
  for(let j=0;j<HMM_S;j++)hmm.transCounts[i][j]*=forget;
  for(let d=0;d<HMM_K;d++)hmm.emitCounts[i][d]*=forget;
 }
 for(let i=0;i<HMM_S;i++)for(let j=0;j<HMM_S;j++)hmm.transCounts[i][j]+=xi[i][j];
 for(let j=0;j<HMM_S;j++)hmm.emitCounts[j][obs]+=gamma[j];

 for(let i=0;i<HMM_S;i++){
  hmm.A[i]=normalizeRow(hmm.transCounts[i].map(x=>x+.05));
  hmm.B[i]=normalizeRow(hmm.emitCounts[i].map(x=>x+.08));
 }

 hmm.alpha=gamma;
 hmm.surprise=-Math.log(like);
 hmm.updates++;
}
function analyse(){
 if(hist.length<240)return null;
 if(!hmm)initHMM();

 const nextState=Array(HMM_S).fill(0);
 for(let j=0;j<HMM_S;j++)for(let i=0;i<HMM_S;i++)nextState[j]+=hmm.alpha[i]*hmm.A[i][j];

 const probs=Array(HMM_K).fill(0);
 for(let d=0;d<HMM_K;d++)for(let j=0;j<HMM_S;j++)probs[d]+=nextState[j]*hmm.B[j][d];

 const rows=probs.map((p,d)=>({d,pt:p,risk:p})).sort((a,b)=>a.pt-b.pt||a.d-b.d);
 const q=rows[0],second=rows[1];
 const regimeProb=Math.max(...hmm.alpha);
 const regime=hmm.alpha.indexOf(regimeProb)+1;
 const entropy=stateEntropy(hmm.alpha);
 const spread=Math.max(0,second.pt-q.pt);
 const quality=100*clamp01((.105-q.pt)/.045);

 return{
  q,
  second,
  spread,
  quality,
  regime,
  regimeProb,
  entropy,
  surprise:hmm.surprise,
  updates:hmm.updates
 };
}
function resetCycle(){
 lastDecision=null;
 if($('evidenceCandidates'))$('evidenceCandidates').textContent='V11 · esperando siguiente predicción HMM.';
}

function showSignal(s){
 lastSignal=s;
 const buy=$('buy');

 if(!s){
  $('decision').textContent=running?'V11 · CALIBRANDO HMM':'AUTO DETENIDO';
  $('reason').textContent=running?'Inicializando estados ocultos y emisiones.':'Pulsa INICIAR AUTO para comenzar.';
  if($('sepPick'))$('sepPick').textContent='—';
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='V11 · esperando historial suficiente.';
  buy.textContent=running?'AUTO V11 · CALIBRANDO':'INICIAR AUTO';
  buy.disabled=running;
  return;
 }

 $('risk').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+s.q.d;
 $('spread').textContent=(s.spread*100).toFixed(1);
 $('entropy').textContent=s.entropy.toFixed(2);
 $('phase').textContent='ESTADO '+s.regime;
 $('meter').style.width=Math.min(100,s.quality)+'%';

 if($('finalScore'))$('finalScore').textContent=s.quality.toFixed(0)+'/100';
 if($('delayState'))$('delayState').textContent='T+1';
 if($('regimeState'))$('regimeState').textContent='S'+s.regime+' '+(s.regimeProb*100).toFixed(0)+'%';
 if($('techScore'))$('techScore').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('techRsi'))$('techRsi').textContent=(s.second.pt*100).toFixed(1)+'%';
 if($('techMacd'))$('techMacd').textContent=s.surprise.toFixed(2);
 if($('techTrend'))$('techTrend').textContent='UPD '+s.updates;

 if(pending){
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='OPERACIÓN EN CURSO · HMM sigue aprendiendo';
  $('decision').textContent='OPERACIÓN EN CURSO · DIFFER D'+pending.d;
  $('reason').textContent='La compra automática espera la liquidación antes de abrir otra.';
  buy.textContent='OPERACIÓN EN CURSO';
  buy.disabled=true;
  return;
 }

 if($('evidenceCandidates')){
  $('evidenceCandidates').textContent='HMM T+1 · DIFFER D'+s.q.d+' · P estimada '+(s.q.pt*100).toFixed(1)+'% · SIN FILTROS';
 }

 if(!running){
  $('decision').textContent='V11 · CANDIDATO D'+s.q.d;
  $('reason').textContent='HMM adaptativo listo · AUTO detenido.';
  buy.textContent='INICIAR AUTO';
  buy.disabled=false;
 }else{
  $('decision').textContent='V11 · COMPRA D'+s.q.d;
  $('reason').textContent='El HMM eligió el dígito con menor probabilidad T+1.';
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

// COMPRA AUTOMÁTICA: mecanismo preservado.
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
 $('status').textContent='ERROR · NUEVA BÚSQUEDA V10';
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
  $('status').textContent='V10 · BUSCANDO NUEVA SEÑAL';
 }
 ui();
}

function tick(d){
 if(hmm)updateHMM(d);

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

 // SIN FILTROS HEREDADOS: el candidato HMM se compra automáticamente.
 if(running&&!pending&&s){
  enter(s);
 }
}
// CONEXIÓN PÚBLICA DE TICKS: preservada.
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
 lastPick=null;lastSignal=null;candidateHistory=[];lastDecision=null;hmm=null;running=true;

 $('status').textContent='V11 ACTIVA · HMM ADAPTATIVO · AUTO DIRECTO T+1';
 $('buy').textContent='AUTO V11 · ACTIVO';
 $('buy').disabled=true;
 log('V9 INICIADA '+mode+' · STAKE $'+stake.toFixed(2)+' · META STOP $'+target().toFixed(2));
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
