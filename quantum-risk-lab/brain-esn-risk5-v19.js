const $=x=>document.getElementById(x);
let running=false,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,lastDecision=null,esn=null;

const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const clamp01=x=>Math.max(0,Math.min(1,x));
function log(s){$('log').textContent=s+'\n'+$('log').textContent}

// RESEARCH V19 · DUAL-TIMESCALE ECHO STATE NETWORK + ONLINE RLS
// Reservoir recurrente fijo con unidades rápidas y lentas.
// Dos readouts RLS con distintas memorias (rápida/lenta).
// Predicción T+1; aprende de cada tick de forma prequential.
// V16 queda guardada como candidata separada.

const ESN_N=28,ESN_K=10,ESN_D=1+ESN_N+ESN_K;

function rngFactory(seed){
 let x=seed>>>0;
 return()=>{
  x^=x<<13;x^=x>>>17;x^=x<<5;
  return((x>>>0)/4294967296);
 };
}
function softmax(raw){
 const mx=Math.max(...raw);
 const ex=raw.map(v=>Math.exp(Math.max(-30,Math.min(30,(v-mx)*2.4))));
 const z=ex.reduce((a,b)=>a+b,0)||1;
 return ex.map(v=>v/z);
}
function makeReadout(lambda){
 const W=Array.from({length:ESN_K},()=>new Float64Array(ESN_D));
 const P=Array.from({length:ESN_D},()=>new Float64Array(ESN_D));
 for(let i=0;i<ESN_D;i++)P[i][i]=4;
 return{W,P,lambda};
}
function rawOut(ro,phi){
 const y=Array(ESN_K).fill(0);
 for(let k=0;k<ESN_K;k++){
  let s=0;
  const w=ro.W[k];
  for(let i=0;i<ESN_D;i++)s+=w[i]*phi[i];
  y[k]=s;
 }
 return y;
}
function updateRLS(ro,phi,target){
 const D=ESN_D;
 const Px=new Float64Array(D);
 for(let i=0;i<D;i++){
  let s=0;
  for(let j=0;j<D;j++)s+=ro.P[i][j]*phi[j];
  Px[i]=s;
 }
 let den=ro.lambda;
 for(let i=0;i<D;i++)den+=phi[i]*Px[i];
 den=Math.max(1e-9,den);

 const gain=new Float64Array(D);
 for(let i=0;i<D;i++)gain[i]=Px[i]/den;

 const raw=rawOut(ro,phi);
 for(let k=0;k<ESN_K;k++){
  const err=(k===target?1:0)-raw[k];
  for(let i=0;i<D;i++)ro.W[k][i]+=err*gain[i];
 }

 const xTP=new Float64Array(D);
 for(let j=0;j<D;j++){
  let s=0;
  for(let i=0;i<D;i++)s+=phi[i]*ro.P[i][j];
  xTP[j]=s;
 }
 for(let i=0;i<D;i++){
  for(let j=0;j<D;j++){
   ro.P[i][j]=(ro.P[i][j]-gain[i]*xTP[j])/ro.lambda;
  }
 }
}
function initReservoir(){
 const rnd=rngFactory(0x51a7c3);
 const W=Array.from({length:ESN_N},()=>new Float64Array(ESN_N));
 const Win=Array.from({length:ESN_N},()=>new Float64Array(ESN_K));

 for(let i=0;i<ESN_N;i++){
  let sumAbs=0;
  for(let j=0;j<ESN_N;j++){
   if(rnd()<.18){
    const v=(rnd()*2-1);
    W[i][j]=v;
    sumAbs+=Math.abs(v);
   }
  }
  if(sumAbs>0){
   const sc=.88/sumAbs;
   for(let j=0;j<ESN_N;j++)W[i][j]*=sc;
  }
  for(let d=0;d<ESN_K;d++)Win[i][d]=(rnd()*2-1)*.65;
 }

 esn={
  W,Win,
  state:new Float64Array(ESN_N),
  fast:makeReadout(.972),
  slow:makeReadout(.995),
  evals:[],
  updates:0,
  phi:null,
  pred:null
 };
}
function advanceState(d){
 const old=esn.state,next=new Float64Array(ESN_N);
 for(let i=0;i<ESN_N;i++){
  let s=esn.Win[i][d];
  for(let j=0;j<ESN_N;j++)s+=esn.W[i][j]*old[j];
  const leak=i<ESN_N/2?.66:.24;
  next[i]=(1-leak)*old[i]+leak*Math.tanh(s);
 }
 esn.state=next;
}
function makePhi(lastDigit){
 const phi=new Float64Array(ESN_D);
 phi[0]=1;
 for(let i=0;i<ESN_N;i++)phi[1+i]=esn.state[i];
 if(lastDigit!==null&&lastDigit!==undefined)phi[1+ESN_N+lastDigit]=1;
 return phi;
}
function ensemblePredict(phi){
 const pf=softmax(rawOut(esn.fast,phi));
 const ps=softmax(rawOut(esn.slow,phi));
 const p=pf.map((v,d)=>(v+ps[d])*.5);
 return{p,pf,ps};
}
function rank(p){
 const rows=p.map((pt,d)=>({d,pt})).sort((a,b)=>a.pt-b.pt||a.d-b.d);
 return{q:rows[0],second:rows[1]};
}
function rawScore(q,second){
 return .72*(.10-q.pt)+.28*Math.max(0,second.pt-q.pt);
}
function recordEval(pred,y){
 const r=rank(pred.p),rf=rank(pred.pf),rs=rank(pred.ps);
 esn.evals.push({
  pred:r.q.pt,
  match:r.q.d===y?1:0,
  score:rawScore(r.q,r.second),
  agree:rf.q.d===rs.q.d?1:0
 });
 if(esn.evals.length>180)esn.evals.shift();
}
function percentile(a,q){
 if(!a.length)return 0;
 const b=a.slice().sort((x,y)=>x-y);
 const pos=(b.length-1)*q,lo=Math.floor(pos),hi=Math.ceil(pos);
 if(lo===hi)return b[lo];
 const t=pos-lo;
 return b[lo]*(1-t)+b[hi]*t;
}
function stats(){
 const e=esn.evals.slice(-140),n=e.length;
 if(!n)return{n:0,matchRate:.10,avgPred:.10,bias:0,agreeRate:0,threshold:0};

 let m=0,ap=0,ag=0;
 for(const x of e){m+=x.match;ap+=x.pred;ag+=x.agree}
 ap/=n;
 const matchRate=(m+1)/(n+10);
 return{
  n,
  matchRate,
  avgPred:ap,
  bias:matchRate-ap,
  agreeRate:ag/n,
  threshold:percentile(e.map(x=>x.score),.68)
 };
}
function warmESN(){
 initReservoir();
 const seq=hist.slice(-320);
 if(seq.length<2)return;

 advanceState(seq[0]);
 esn.phi=makePhi(seq[0]);

 for(let t=1;t<seq.length;t++){
  const pred=ensemblePredict(esn.phi);
  recordEval(pred,seq[t]);
  updateRLS(esn.fast,esn.phi,seq[t]);
  updateRLS(esn.slow,esn.phi,seq[t]);
  esn.updates++;
  advanceState(seq[t]);
  esn.phi=makePhi(seq[t]);
 }
 esn.pred=ensemblePredict(esn.phi);
}
function learnObserved(y){
 if(!esn)warmESN();
 if(!esn||!esn.phi)return;

 const pred=esn.pred||ensemblePredict(esn.phi);
 recordEval(pred,y);
 updateRLS(esn.fast,esn.phi,y);
 updateRLS(esn.slow,esn.phi,y);
 esn.updates++;

 advanceState(y);
 esn.phi=makePhi(y);
 esn.pred=ensemblePredict(esn.phi);
}
function analyse(){
 if(hist.length<80)return null;
 if(!esn)warmESN();
 if(!esn||!esn.pred)return null;

 const pred=esn.pred;
 const r=rank(pred.p),rf=rank(pred.pf),rs=rank(pred.ps);
 const st=stats();

 const calibratedRisk=clamp01(r.q.pt+st.bias);
 const margin=Math.max(0,r.second.pt-r.q.pt);
 const score=rawScore(r.q,r.second);
 const modelAgree=rf.q.d===rs.q.d&&rf.q.d===r.q.d;

 // Regla V19: solo son elegibles candidatos con riesgo calibrado < 5%.
 const calibratedRows=pred.p.map((pt,d)=>({
  d,
  pt,
  calibratedRisk:clamp01(pt+st.bias)
 })).sort((a,b)=>a.calibratedRisk-b.calibratedRisk||a.d-b.d);

 const eligible=calibratedRows.filter(x=>x.calibratedRisk<.05);
 const under5=eligible.length>0&&eligible.some(x=>x.d===r.q.d);

 const edge=clamp01((.05-calibratedRisk)/.025);
 const sep=clamp01(margin/.012);
 const histQuality=clamp01((.118-st.matchRate)/.04);
 const quality=100*(.46*edge+.26*sep+.18*histQuality+.10*(modelAgree?1:0));

 const buySignal=
  st.n>=65&&
  under5&&
  modelAgree&&
  calibratedRisk<.05&&
  margin>=.0015&&
  score>=st.threshold&&
  st.matchRate<.12;

 return{
  q:r.q,
  second:r.second,
  quality,
  buySignal,
  calibratedRisk,
  eligibleCount:eligible.length,
  under5,
  margin,
  score,
  threshold:st.threshold,
  evalN:st.n,
  matchRate:st.matchRate,
  agreeRate:st.agreeRate,
  modelAgree,
  updates:esn.updates
 };
}
function resetCycle(){
 lastDecision=null;
 if($('evidenceCandidates'))$('evidenceCandidates').textContent='V19 · esperando señal ESN-RLS validada.';
}
function showSignal(s){
 lastSignal=s;
 const buy=$('buy');

 if(!s){
  $('decision').textContent=running?'V19 · ENTRENANDO ESN':'AUTO DETENIDO';
  $('reason').textContent=running?'Reservorio recurrente aprendiendo de cada tick.':'Pulsa INICIAR AUTO para comenzar.';
  if($('sepPick'))$('sepPick').textContent='—';
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='V19 · esperando historial mínimo.';
  buy.textContent=running?'AUTO V19 · ENTRENANDO':'INICIAR AUTO';
  buy.disabled=running;
  return;
 }

 $('risk').textContent=(s.calibratedRisk*100).toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+s.q.d;
 $('spread').textContent=(s.margin*100).toFixed(2);
 $('entropy').textContent=(s.matchRate*100).toFixed(1)+'%';
 $('phase').textContent=s.modelAgree?'ESN ACUERDO':'ESN DISCREPA';
 $('meter').style.width=Math.min(100,s.quality)+'%';

 if($('finalScore'))$('finalScore').textContent=s.quality.toFixed(0)+'/100';
 if($('delayState'))$('delayState').textContent='T+1';
 if($('regimeState'))$('regimeState').textContent='EVAL '+s.evalN;
 if($('techScore'))$('techScore').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('techRsi'))$('techRsi').textContent=(s.matchRate*100).toFixed(1)+'%';
 if($('techMacd'))$('techMacd').textContent=(s.score*100).toFixed(2);
 if($('techTrend'))$('techTrend').textContent='UPD '+s.updates;

 if(pending){
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='OPERACIÓN EN CURSO · ESN sigue aprendiendo';
  $('decision').textContent='OPERACIÓN EN CURSO · DIFFER D'+pending.d;
  $('reason').textContent='No abre otra operación hasta liquidación.';
  buy.textContent='OPERACIÓN EN CURSO';
  buy.disabled=true;
  return;
 }

 if($('evidenceCandidates')){
  $('evidenceCandidates').textContent=
   'ESN-RLS · D'+s.q.d+
   ' · P '+(s.q.pt*100).toFixed(1)+'%'+
   ' · RIESGO CAL '+(s.calibratedRisk*100).toFixed(1)+'%'+
   ' · MATCH PREQ '+(s.matchRate*100).toFixed(1)+'%'+
   ' · <5% '+s.eligibleCount+
   ' · '+(s.modelAgree?'2/2 ACUERDO':'DISCREPA')+
   ' · '+(s.buySignal?'COMPRAR':'ESPERAR');
 }

 if(!running){
  $('decision').textContent='V19 · CANDIDATO D'+s.q.d;
  $('reason').textContent='Echo State Network online listo · AUTO detenido.';
  buy.textContent='INICIAR AUTO';
  buy.disabled=false;
 }else if(s.buySignal){
  $('decision').textContent='V19 · COMPRA D'+s.q.d;
  $('reason').textContent='Reservorio rápido/lento coincide, la señal está bajo 5% y supera validación reciente.';
  buy.textContent='AUTO · COMPRA D'+s.q.d;
  buy.disabled=true;
 }else{
  $('decision').textContent='V19 · ESPERANDO';
  $('reason').textContent=s.calibratedRisk>=.05?'No compra: el mejor candidato todavía tiene riesgo de 5% o más.':'La red tiene candidato <5%, pero todavía no cumple acuerdo y validación.';
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

// COMPRA AUTOMÁTICA: función preservada exactamente desde V17.
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
 $('status').textContent='ERROR · V19 SIGUE APRENDIENDO';
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
  $('status').textContent='V19 · BUSCANDO SIGUIENTE SEÑAL';
 }
 ui();
}
function tick(d){
 if(hist.length>=80)learnObserved(d);

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
  enter(s);
 }
}

// CONEXIÓN PÚBLICA DE TICKS: función preservada exactamente desde V17.
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
 lastPick=null;lastSignal=null;lastDecision=null;
 if(!esn&&hist.length>=80)warmESN();
 running=true;

 $('status').textContent='V19 ACTIVA · ESN-RLS · SOLO RIESGO <5% · T+1';
 $('buy').textContent='AUTO V19 · BUSCANDO SEÑAL';
 $('buy').disabled=true;
 log('V19 ESN-RLS INICIADA '+mode+' · STAKE $'+stake.toFixed(2)+' · META STOP $'+target().toFixed(2));
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
