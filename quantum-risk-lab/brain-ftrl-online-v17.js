const $=x=>document.getElementById(x);
let running=false,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,lastDecision=null,ftrl=null;

const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const clamp01=x=>Math.max(0,Math.min(1,x));
function log(s){$('log').textContent=s+'\n'+$('log').textContent}

// RESEARCH V17 · ONLINE MULTICLASS FTRL-PROXIMAL
// Cada tick es un ejemplo de entrenamiento.
// Predice los 10 dígitos, observa el resultado real y actualiza inmediatamente.
// Evaluación prequential: siempre se evalúa la predicción hecha ANTES de conocer el tick.
// V16 PPM-C Learning queda intacta como candidata separada.

const DIM=2048,CLASSES=10;
const F_ALPHA=.075,F_BETA=1,F_L1=0,F_L2=1.15;

function hashFeature(s){
 let h=2166136261>>>0;
 for(let i=0;i<s.length;i++){
  h^=s.charCodeAt(i);
  h=Math.imul(h,16777619)>>>0;
 }
 return 1+(h%(DIM-1));
}
function addFeature(map,key,val=1){
 const i=key==='BIAS'?0:hashFeature(key);
 map.set(i,(map.get(i)||0)+val);
}
function featuresFrom(seq){
 const m=new Map();
 addFeature(m,'BIAS',1);
 const n=seq.length;

 for(let pos=1;pos<=8;pos++){
  if(n>=pos)addFeature(m,'P'+pos+'='+seq[n-pos],1);
 }
 for(let len=2;len<=5;len++){
  if(n>=len)addFeature(m,'S'+len+'='+seq.slice(-len).join(''),1);
 }

 for(const w of [10,30,80]){
  const tail=seq.slice(-w);
  const den=Math.max(1,tail.length);
  const c=Array(10).fill(0);
  for(const d of tail)c[d]++;
  for(let d=0;d<10;d++)addFeature(m,'F'+w+'D'+d,c[d]/den);
 }

 for(let d=0;d<10;d++){
  let gap=30;
  for(let k=1;k<=Math.min(30,n);k++){
   if(seq[n-k]===d){gap=k-1;break}
  }
  addFeature(m,'GAP'+d,Math.min(30,gap)/30);
 }

 if(n){
  let run=1;
  for(let i=n-2;i>=0&&seq[i]===seq[n-1]&&run<8;i--)run++;
  addFeature(m,'RUN',run/8);
 }
 return [...m.entries()];
}
function initFTRLState(){
 ftrl={
  z:Array.from({length:CLASSES},()=>new Float64Array(DIM)),
  n:Array.from({length:CLASSES},()=>new Float64Array(DIM)),
  evals:[],
  updates:0
 };
}
function weightFor(k,i){
 const z=ftrl.z[k][i],n=ftrl.n[k][i];
 if(Math.abs(z)<=F_L1)return 0;
 return-(z-Math.sign(z)*F_L1)/((F_BETA+Math.sqrt(n))/F_ALPHA+F_L2);
}
function predictFTRL(x){
 const logits=Array(CLASSES).fill(0);
 for(let k=0;k<CLASSES;k++){
  let s=0;
  for(const [i,v] of x)s+=weightFor(k,i)*v;
  logits[k]=s;
 }
 const mx=Math.max(...logits);
 const ex=logits.map(v=>Math.exp(Math.max(-35,Math.min(35,v-mx))));
 const z=ex.reduce((a,b)=>a+b,0)||1;
 return ex.map(v=>v/z);
}
function trainFTRL(x,p,y){
 for(let k=0;k<CLASSES;k++){
  const err=p[k]-(k===y?1:0);
  for(const [i,v] of x){
   const g=err*v;
   const oldN=ftrl.n[k][i];
   const w=weightFor(k,i);
   const newN=oldN+g*g;
   const sigma=(Math.sqrt(newN)-Math.sqrt(oldN))/F_ALPHA;
   ftrl.z[k][i]+=g-sigma*w;
   ftrl.n[k][i]=newN;
  }
 }
 ftrl.updates++;
}
function rankedCandidate(p,seq){
 let hash=2166136261>>>0;
 for(const d of seq.slice(-8)){hash^=(d+31);hash=Math.imul(hash,16777619)>>>0}
 const rows=p.map((pt,d)=>({d,pt,tie:(d+hash)%10}))
  .sort((a,b)=>Math.abs(a.pt-b.pt)>1e-12?a.pt-b.pt:a.tie-b.tie);
 return{q:rows[0],second:rows[1]};
}
function rawDecisionScore(q,second){
 const edge=.10-q.pt;
 const margin=second.pt-q.pt;
 return .72*edge+.28*margin;
}
function recordPrequential(p,y,seq){
 const r=rankedCandidate(p,seq);
 const score=rawDecisionScore(r.q,r.second);
 ftrl.evals.push({
  pred:r.q.pt,
  match:r.q.d===y?1:0,
  score
 });
 if(ftrl.evals.length>180)ftrl.evals.shift();
}
function percentile(a,q){
 if(!a.length)return 0;
 const b=a.slice().sort((x,y)=>x-y);
 const pos=(b.length-1)*q;
 const lo=Math.floor(pos),hi=Math.ceil(pos);
 if(lo===hi)return b[lo];
 const t=pos-lo;
 return b[lo]*(1-t)+b[hi]*t;
}
function reliabilityStats(){
 const e=ftrl.evals.slice(-140);
 const n=e.length;
 if(!n)return{n:0,posteriorMatch:.10,avgPred:.10,bias:0,threshold:0};

 let matches=0,avgPred=0;
 for(const x of e){matches+=x.match;avgPred+=x.pred}
 avgPred/=n;

 // Beta(1,9) prior has mean 10%; used only as a stabilizer for prequential outcomes.
 const posteriorMatch=(matches+1)/(n+10);
 const bias=posteriorMatch-avgPred;
 const threshold=percentile(e.map(x=>x.score),.62);
 return{n,posteriorMatch,avgPred,bias,threshold};
}
function warmFTRL(){
 initFTRLState();
 const seq=hist.slice(-320);
 const ctx=[];

 for(const y of seq){
  if(ctx.length>=8){
   const x=featuresFrom(ctx);
   const p=predictFTRL(x);
   recordPrequential(p,y,ctx);
   trainFTRL(x,p,y);
  }
  ctx.push(y);
 }
}
function learnTick(y){
 if(!ftrl)warmFTRL();
 const x=featuresFrom(hist);
 const p=predictFTRL(x);
 recordPrequential(p,y,hist);
 trainFTRL(x,p,y);
}
function analyse(){
 if(hist.length<80)return null;
 if(!ftrl)warmFTRL();

 const x=featuresFrom(hist);
 const p=predictFTRL(x);
 const {q,second}=rankedCandidate(p,hist);
 const stats=reliabilityStats();

 const calibratedRisk=clamp01(q.pt+stats.bias);
 const margin=Math.max(0,second.pt-q.pt);
 const score=rawDecisionScore(q,second);
 const adaptiveThreshold=stats.threshold;

 // Calidad puramente diagnóstica.
 const edge=clamp01((.10-calibratedRisk)/.045);
 const sep=clamp01(margin/.010);
 const reliability=clamp01((.115-stats.posteriorMatch)/.035);
 const quality=100*(.46*edge+.30*sep+.24*reliability);

 // Entrada adaptativa propia de V17:
 // 1) suficiente evaluación prequential,
 // 2) riesgo calibrado por debajo del 10% neutral,
 // 3) señal actual mejor que el 62% reciente del propio modelo.
 const buySignal=
  stats.n>=55&&
  calibratedRisk<.10&&
  score>=adaptiveThreshold&&
  q.pt<second.pt;

 return{
  q,
  second,
  margin,
  quality,
  buySignal,
  calibratedRisk,
  posteriorMatch:stats.posteriorMatch,
  avgPred:stats.avgPred,
  bias:stats.bias,
  threshold:adaptiveThreshold,
  score,
  evalN:stats.n,
  updates:ftrl.updates
 };
}

function resetCycle(){
 lastDecision=null;
 if($('evidenceCandidates'))$('evidenceCandidates').textContent='V17 · esperando señal FTRL validada online.';
}

function showSignal(s){
 lastSignal=s;
 const buy=$('buy');

 if(!s){
  $('decision').textContent=running?'V17 · ENTRENANDO FTRL':'AUTO DETENIDO';
  $('reason').textContent=running?'Aprendiendo online de cada tick.':'Pulsa INICIAR AUTO para comenzar.';
  if($('sepPick'))$('sepPick').textContent='—';
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='V17 · esperando historial mínimo.';
  buy.textContent=running?'AUTO V17 · ENTRENANDO':'INICIAR AUTO';
  buy.disabled=running;
  return;
 }

 $('risk').textContent=(s.calibratedRisk*100).toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+s.q.d;
 $('spread').textContent=(s.margin*100).toFixed(2);
 $('entropy').textContent=(s.posteriorMatch*100).toFixed(1)+'%';
 $('phase').textContent='FTRL ONLINE';
 $('meter').style.width=Math.min(100,s.quality)+'%';

 if($('finalScore'))$('finalScore').textContent=s.quality.toFixed(0)+'/100';
 if($('delayState'))$('delayState').textContent='T+1';
 if($('regimeState'))$('regimeState').textContent='EVAL '+s.evalN;
 if($('techScore'))$('techScore').textContent=(s.q.pt*100).toFixed(1)+'%';
 if($('techRsi'))$('techRsi').textContent=(s.posteriorMatch*100).toFixed(1)+'%';
 if($('techMacd'))$('techMacd').textContent=(s.score*100).toFixed(2);
 if($('techTrend'))$('techTrend').textContent='UPD '+s.updates;

 if(pending){
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='OPERACIÓN EN CURSO · FTRL sigue aprendiendo con cada tick';
  $('decision').textContent='OPERACIÓN EN CURSO · DIFFER D'+pending.d;
  $('reason').textContent='No abre otra operación hasta liquidación, pero el modelo continúa aprendiendo.';
  buy.textContent='OPERACIÓN EN CURSO';
  buy.disabled=true;
  return;
 }

 if($('evidenceCandidates')){
  $('evidenceCandidates').textContent=
   'FTRL · D'+s.q.d+
   ' · P '+(s.q.pt*100).toFixed(1)+'%'+
   ' · RIESGO CAL '+(s.calibratedRisk*100).toFixed(1)+'%'+
   ' · MATCH PREQ '+(s.posteriorMatch*100).toFixed(1)+'%'+
   ' · '+(s.buySignal?'COMPRAR':'ESPERAR');
 }

 if(!running){
  $('decision').textContent='V17 · CANDIDATO D'+s.q.d;
  $('reason').textContent='FTRL online listo · AUTO detenido.';
  buy.textContent='INICIAR AUTO';
  buy.disabled=false;
 }else if(s.buySignal){
  $('decision').textContent='V17 · COMPRA D'+s.q.d;
  $('reason').textContent='La señal supera su umbral adaptativo y su validación prequential.';
  buy.textContent='AUTO · COMPRA D'+s.q.d;
  buy.disabled=true;
 }else{
  $('decision').textContent='V17 · ESPERANDO';
  $('reason').textContent='FTRL encontró candidato, pero todavía no tiene suficiente ventaja validada.';
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

// COMPRA AUTOMÁTICA: función preservada exactamente desde V16.
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
 $('status').textContent='ERROR · V17 SIGUE APRENDIENDO';
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
  $('status').textContent='V17 · BUSCANDO SIGUIENTE SEÑAL';
 }
 ui();
}

function tick(d){
 // El modelo aprende del tick real ANTES de usarlo como contexto para T+1.
 if(hist.length>=80)learnTick(d);

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

// CONEXIÓN PÚBLICA DE TICKS: función preservada exactamente desde V16.
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
 if(!ftrl&&hist.length>=80)warmFTRL();
 running=true;

 $('status').textContent='V17 ACTIVA · FTRL ONLINE MULTICLASE · T+1';
 $('buy').textContent='AUTO V17 · BUSCANDO SEÑAL';
 $('buy').disabled=true;
 log('V17 FTRL INICIADA '+mode+' · STAKE $'+stake.toFixed(2)+' · META STOP $'+target().toFixed(2));
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
