const $=x=>document.getElementById(x);
let running=false,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,candidateHistory=[],autoCandidateCount=0,lastCandidateSeen=null,cycleScores=[],bestCycleScore=0,lastDecision=null,lastExpertPreds=null,expertPerf=null;
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function clamp01(x){return Math.max(0,Math.min(1,x))}

// RESEARCH V8
// - Predicción alineada con el contrato real: T+1 (1 tick).
// - Estimación KT (alpha=1/2) para evitar probabilidades extremas por muestras pequeñas.
// - Ensamble de contextos 0,1,2,3 y ventanas cortas/medias/largas.
// - Los expertos se ponderan por log-loss + Brier observados online.
// - Detección de drift por divergencia Jensen-Shannon entre ventanas recientes.
// - Selección conservadora: probabilidad estimada de MATCH + desacuerdo + drift + soporte.
// - Espera adaptativa de hasta 4 candidatos. El 4.º NO fuerza compra si sigue débil.
const EXPERTS=[
 {o:0,w:80},{o:0,w:240},{o:0,w:700},
 {o:1,w:100},{o:1,w:300},{o:1,w:800},
 {o:2,w:160},{o:2,w:420},
 {o:3,w:240},{o:3,w:700}
];

function initExpertPerf(){
 if(!expertPerf||expertPerf.length!==EXPERTS.length){
  expertPerf=EXPERTS.map(()=>({n:0,log:Math.log(10),brier:.90}));
 }
}
function updateExpertPerformance(actual){
 initExpertPerf();
 if(!lastExpertPreds||lastExpertPreds.length!==EXPERTS.length)return;
 for(let i=0;i<EXPERTS.length;i++){
  const p=lastExpertPreds[i];
  if(!p)continue;
  const pa=Math.max(1e-9,p[actual]||0);
  const ll=-Math.log(pa);
  let bs=0;
  for(let d=0;d<10;d++){
   const y=d===actual?1:0;
   bs+=(p[d]-y)*(p[d]-y);
  }
  const st=expertPerf[i];
  const a=st.n<25 ? .16 : .07;
  st.log=(1-a)*st.log+a*ll;
  st.brier=(1-a)*st.brier+a*bs;
  st.n++;
 }
 lastExpertPreds=null;
}
function empiricalDist(arr){
 const c=Array(10).fill(.5);
 for(const d of arr)c[d]++;
 const z=c.reduce((a,b)=>a+b,0);
 return c.map(x=>x/z);
}
function jsDivergence(a,b){
 const m=a.map((x,i)=>(x+b[i])/2);
 const kl=(p,q)=>p.reduce((s,x,i)=>x>0?s+x*Math.log(x/Math.max(q[i],1e-12)):s,0);
 return .5*kl(a,m)+.5*kl(b,m);
}
function driftLevel(){
 if(hist.length<180)return 0;
 const recent=empiricalDist(hist.slice(-50));
 const prior=empiricalDist(hist.slice(-170,-50));
 return clamp01(jsDivergence(recent,prior)/.12);
}
function ktPredict(order,window){
 const n=hist.length,alpha=.5,counts=Array(10).fill(0);
 if(n<Math.max(1,order+1))return{p:Array(10).fill(.1),support:0};
 const ctx=order?hist.slice(n-order):[];
 const start=Math.max(order,n-window);
 let support=0;
 for(let t=start;t<n;t++){
  if(order){
   let ok=true;
   for(let k=0;k<order;k++){
    if(hist[t-order+k]!==ctx[k]){ok=false;break}
   }
   if(!ok)continue;
  }
  counts[hist[t]]++;
  support++;
 }
 const den=support+10*alpha;
 return{p:counts.map(c=>(c+alpha)/den),support};
}
function analyse(){
 if(hist.length<260)return null;
 initExpertPerf();
 const drift=driftLevel();
 const models=EXPERTS.map((e,i)=>{
  const r=ktPredict(e.o,e.w),perf=expertPerf[i];
  const normLog=perf.log/Math.log(10);
  const normBrier=perf.brier/.90;
  const loss=.65*normLog+.35*normBrier;
  let reliability=Math.exp(-2.2*(loss-1));
  const needed=e.o===0 ? 18 : Math.max(4,6*e.o);
  reliability*=.20+.80*clamp01(r.support/needed);
  if(drift>.20){
   if(e.w<=160)reliability*=1+.85*drift;
   if(e.w>=700)reliability*=1-.45*drift;
  }
  return{...e,p:r.p,support:r.support,reliability,loss};
 });
 let z=models.reduce((s,m)=>s+m.reliability,0);
 if(!Number.isFinite(z)||z<=0)z=1;
 models.forEach(m=>m.weight=m.reliability/z);
 const mix=Array(10).fill(0);
 for(const m of models)for(let d=0;d<10;d++)mix[d]+=m.weight*m.p[d];
 lastExpertPreds=models.map(m=>m.p.slice());
 const variance=Array(10).fill(0);
 for(let d=0;d<10;d++){
  for(const m of models){
   const dx=m.p[d]-mix[d];
   variance[d]+=m.weight*dx*dx;
  }
 }
 const entropy=-mix.reduce((s,p)=>p>0?s+p*Math.log(p):s,0);
 const entropyNorm=entropy/Math.log(10);
 const avgLoss=models.reduce((s,m)=>s+m.weight*m.loss,0);
 const calibration=clamp01((1.10-avgLoss)/.22);
 const rows=[];
 for(let d=0;d<10;d++){
  const disagreement=Math.sqrt(Math.max(0,variance[d]));
  let support=0;
  for(const m of models){
   const scale=m.o===0 ? 40 : Math.max(6,8*m.o);
   support+=m.weight*clamp01(m.support/scale);
  }
  const conservativeRisk=
   mix[d]+
   .70*disagreement+
   .004*drift+
   .004*(1-support);
  rows.push({
   d,
   pt:mix[d],
   pe:mix[d],
   risk:conservativeRisk,
   disagreement,
   support,
   confidence:clamp01(.55*calibration+.30*(1-Math.min(1,disagreement/.03))+.15*support)
  });
 }
 rows.sort((a,b)=>a.risk-b.risk||a.pt-b.pt);
 const q=rows[0],second=rows[1];
 if(!q||!second)return null;
 const spread=Math.max(0,second.risk-q.risk);
 const safe=
  q.risk<.101&&
  q.pt<.10&&
  q.disagreement<.024&&
  q.support>=.18&&
  calibration>=.22&&
  drift<.92;
 return{
  q,
  spread,
  H:entropy/Math.log(2),
  entropyNorm,
  calibration,
  drift,
  safe,
  experts:models
 };
}
function candidateQuality(s){
 if(!s||!s.q)return 0;
 const q=s.q;
 const edge=clamp01((.102-q.risk)/.026);
 const rawEdge=clamp01((.102-q.pt)/.024);
 const agreement=clamp01(1-q.disagreement/.025);
 const support=clamp01(q.support/.65);
 const calibration=clamp01(s.calibration);
 const separation=clamp01(s.spread/.010);
 const stability=clamp01(1-s.drift);
 const entropyStructure=clamp01((1-s.entropyNorm)/.05);
 const score=100*(
  .28*edge+
  .18*rawEdge+
  .15*agreement+
  .12*calibration+
  .10*support+
  .07*separation+
  .06*stability+
  .04*entropyStructure
 );
 return Math.max(0,Math.min(100,score));
}
function evaluateCandidate(s,count){
 const score=candidateQuality(s),previousBest=bestCycleScore;
 let buy=false,threshold=0,reason='',restart=false;
 if(count===1){
  threshold=70;
  buy=score>=threshold&&s.safe;
  reason=buy?'Candidato 1 con evidencia fuerte.':'Candidato 1 todavía no justifica entrada.';
 }else if(count===2){
  threshold=63;
  const clearImprovement=score>=previousBest+8;
  buy=(score>=threshold&&s.q.risk<.101)||(clearImprovement&&score>=60&&s.q.risk<.10);
  reason=buy?'Candidato 2 supera el balance riesgo/incertidumbre.':'Candidato 2: conviene seguir esperando.';
 }else if(count===3){
  threshold=57;
  const improvement=score>=previousBest+5;
  buy=(score>=threshold&&s.q.risk<.102)||(improvement&&score>=54&&s.q.risk<.101);
  reason=buy?'Candidato 3 alcanza evidencia suficiente.':'Candidato 3 sigue débil; queda una evaluación.';
 }else{
  threshold=51;
  buy=score>=threshold&&s.q.risk<.103&&s.calibration>=.20;
  restart=!buy;
  reason=buy?'Candidato 4 alcanza el mínimo estadístico de entrada.':'Cuatro candidatos débiles: ciclo descartado sin comprar.';
 }
 cycleScores.push({n:count,d:s.q.d,score,risk:s.q.risk,pt:s.q.pt});
 bestCycleScore=Math.max(previousBest,score);
 return{buy,score,threshold,reason,restart};
}
function resetAdaptiveCycle(anchor=null){
 autoCandidateCount=0;
 lastCandidateSeen=anchor;
 cycleScores=[];
 bestCycleScore=0;
 lastDecision=null;
 if($('evidenceCandidates'))$('evidenceCandidates').textContent='CICLO 0/4 · evaluando evidencia DIFFER.';
}
function showSignal(s){
 lastSignal=s;
 const buy=$('buy');
 if(!s){
  $('decision').textContent=running?'V8 · CALIBRANDO':'AUTO DETENIDO';
  $('reason').textContent=running?'Construyendo contexto y calibración T+1.':'Pulsa INICIAR AUTO para comenzar.';
  if($('sepPick'))$('sepPick').textContent='—';
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='CICLO '+autoCandidateCount+'/4 · esperando evidencia DIFFER.';
  buy.textContent=running?'AUTO V8 · ESPERANDO':'INICIAR AUTO';
  buy.disabled=running;
  return;
 }
 const riskPct=s.q.risk*100,rawPct=s.q.pt*100,score=lastDecision?.score??candidateQuality(s);
 $('risk').textContent=riskPct.toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+s.q.d;
 $('spread').textContent=(s.spread*100).toFixed(1);
 $('entropy').textContent=s.H.toFixed(2);
 $('phase').textContent=s.drift>.55?'CAMBIO':'ESTABLE';
 $('meter').style.width=Math.min(100,score)+'%';
 if($('finalScore'))$('finalScore').textContent=score.toFixed(0)+'/100';
 if($('delayState'))$('delayState').textContent='T+1 · '+autoCandidateCount+'/4';
 if($('regimeState'))$('regimeState').textContent='DRIFT '+(s.drift*100).toFixed(0)+'%';
 if($('techScore'))$('techScore').textContent=(s.calibration*100).toFixed(0)+'%';
 if($('techRsi'))$('techRsi').textContent=rawPct.toFixed(1)+'%';
 if($('techMacd'))$('techMacd').textContent=(s.q.disagreement*100).toFixed(1)+'%';
 if($('techTrend'))$('techTrend').textContent='SOP '+(s.q.support*100).toFixed(0)+'%';
 if(pending){
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='OPERACIÓN EN CURSO · V8 pausada hasta liquidación';
  $('decision').textContent='OPERACIÓN EN CURSO · DIFFER D'+pending.d;
  $('reason').textContent='No se evalúan nuevas entradas hasta WIN o MATCH.';
  buy.textContent='OPERACIÓN EN CURSO';
  buy.disabled=true;
  return;
 }
 if($('evidenceCandidates')){
  $('evidenceCandidates').textContent='CANDIDATO '+Math.max(1,autoCandidateCount)+'/4 · D'+s.q.d+' · MATCH T+1 '+rawPct.toFixed(1)+'% · RIESGO CONS. '+riskPct.toFixed(1)+'%';
 }
 if(!running){
  $('decision').textContent='V8 · CANDIDATO D'+s.q.d;
  $('reason').textContent='Ensamble KT T+1 listo · AUTO detenido.';
  buy.textContent='INICIAR AUTO';
  buy.disabled=false;
 }else if(lastDecision?.buy){
  $('decision').textContent='V8 · COMPRA D'+s.q.d;
  $('reason').textContent=lastDecision.reason+' Calidad '+score.toFixed(0)+'/100.';
  buy.textContent='AUTO · COMPRA D'+s.q.d;
  buy.disabled=true;
 }else if(lastDecision?.restart){
  $('decision').textContent='V8 · SIN ENTRADA';
  $('reason').textContent=lastDecision.reason;
  buy.textContent='AUTO · DESCARTA CICLO';
  buy.disabled=true;
 }else{
  $('decision').textContent='V8 · ESPERA '+autoCandidateCount+'/4';
  $('reason').textContent=(lastDecision?.reason||'Evaluando evidencia T+1.')+' Calidad '+score.toFixed(0)+'/100.';
  buy.textContent='AUTO · ESPERANDO EVIDENCIA';
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

function enter(s){
 if(!running||pending||!s)return;
 const d=s.q.d,mode=$('mode').value;
 if(mode==='DEMO'&&!window.demoReady){$('status').textContent='CONECTA DEMO DERIV PRIMERO';return}
 lastPick=d;observe=0;pending={d,stake,mode};ops++;
 const score=lastDecision?.score??candidateQuality(s);
 $('decision').textContent='AUTO '+mode+' · DIFFER D'+d;
 $('reason').textContent='V8 T+1 · candidato '+autoCandidateCount+'/4 · calidad '+score.toFixed(0)+'/100 · $'+stake.toFixed(2);
 log('V8 KT-ENSEMBLE '+autoCandidateCount+'/4 · D'+d+' · Q'+score.toFixed(0)+' · $'+stake.toFixed(2));
 ui();
 if(mode==='DEMO'){
  $('status').textContent='AUTO · ENVIANDO A DERIV…';
  window.sendDemoTrade(d,stake).catch(e=>tradeError(e));
 }else $('status').textContent='AUTO SIM ABIERTA';
}
function tradeError(e){
 log('ERROR DERIV '+(e?.message||e));
 pending=null;observe=0;
 resetAdaptiveCycle();
 $('status').textContent='ERROR · NUEVO CICLO V8';
 ui();
}
function finish(profit,label){
 profit=Number(profit);
 if(!Number.isFinite(profit)){tradeError(new Error('Resultado inválido'));return}
 pnl+=profit;
 if(profit>0){wins++;stake=Math.max(baseStake(),stake+profit);log('WIN '+label+' +$'+profit.toFixed(2))}
 else{losses++;stake=baseStake();log('MATCH '+label+' $'+profit.toFixed(2))}
 pending=null;observe=0;
 resetAdaptiveCycle();
 if(pnl>=target()){
  running=false;
  $('status').textContent='META +$'+target().toFixed(2)+' · STOP';
  $('phase').textContent='META';
  $('buy').textContent='INICIAR AUTO';
  $('buy').disabled=false;
 }else if(running)$('status').textContent='V8 · NUEVO CICLO DE EVIDENCIA';
 ui();
}
function tick(d){
 updateExpertPerformance(d);
 if(pending&&pending.mode==='SIM'){
  const p=pending;
  finish(d===p.d?-p.stake:p.stake*.10,'SIM');
 }
 hist.push(d);
 if(hist.length>1000)hist.shift();
 if(running&&!pending)observe++;
 ui(d);
 const s=analyse();
 let candidateEvent=false;
 if(s){
  candidateHistory.push(s.q.d);
  if(candidateHistory.length>40)candidateHistory.shift();
  if(s.q.d!==lastCandidateSeen){candidateEvent=true;lastCandidateSeen=s.q.d}
 }
 if(running&&!pending&&s&&candidateEvent){
  autoCandidateCount=Math.min(4,autoCandidateCount+1);
  lastDecision=evaluateCandidate(s,autoCandidateCount);
 }
 showSignal(s);
 if(running&&!pending&&s&&candidateEvent&&lastDecision?.buy){enter(s);return}
 if(running&&!pending&&s&&candidateEvent&&lastDecision?.restart){
  log('V8 · CICLO DESCARTADO · sin evidencia suficiente');
  resetAdaptiveCycle(s.q.d);
  $('status').textContent='V8 · CICLO DESCARTADO · BUSCANDO NUEVA EVIDENCIA';
 }
}

function connect(){
 clearTimeout(retry);ws=new WebSocket('wss://api.derivws.com/trading/v1/options/ws/public');
 ws.onopen=()=>ws.send(JSON.stringify({ticks_history:'R_75',count:300,end:'latest',style:'ticks'}));
 ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.history&&m.history.prices){
   const p=Number(m.pip_size||4);
   hist=m.history.prices.map(x=>Number(Number(x).toFixed(p).slice(-1)));
   ws.send(JSON.stringify({ticks:'R_75',subscribe:1}));
   $('status').textContent='LISTO · '+hist.length+' TICKS';
  }
  if(m.tick){
   const ep=+m.tick.epoch;if(ep===lastEpoch)return;lastEpoch=ep;
   const p=Number(m.tick.pip_size||4),d=Number(Number(m.tick.quote).toFixed(p).slice(-1));
   tick(d);
  }
 };
 ws.onclose=()=>retry=setTimeout(connect,2500);
}
function startAuto(){
 const mode=$('mode').value;
 if(mode==='DEMO'&&!window.demoReady){$('status').textContent='CONECTA DEMO DERIV PRIMERO';return}
 pnl=0;stake=baseStake();wins=0;losses=0;ops=0;pending=null;observe=0;
 lastPick=null;lastSignal=null;candidateHistory=[];autoCandidateCount=0;lastCandidateSeen=null;cycleScores=[];bestCycleScore=0;lastDecision=null;lastExpertPreds=null;
 initExpertPerf();
 running=true;
 $('status').textContent='V8 ACTIVA · KT + CONTEXTOS + EXPERTOS · T+1';
 $('buy').textContent='AUTO V8 · 0/4';
 $('buy').disabled=true;
 log('V8 INICIADA '+mode+' · STAKE $'+stake.toFixed(2)+' · META STOP $'+target().toFixed(2));
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
 if(running){$('status').textContent='AUTO YA ESTÁ ACTIVO';return}
 startAuto();
};
window.demoSettlement=p=>finish(p,'DERIV '+($('mode').value||''));
window.demoTradeError=tradeError;
stake=baseStake();initExpertPerf();$('status').textContent='AUTO DETENIDO · CONECTANDO TICKS';$('buy').textContent='INICIAR AUTO';$('buy').disabled=false;ui();connect();
