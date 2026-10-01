const $=x=>document.getElementById(x);
let running=false,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,candidateHistory=[],lastCandidateSeen=null,lastDecision=null;

const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const clamp01=x=>Math.max(0,Math.min(1,x));
function log(s){$('log').textContent=s+'\n'+$('log').textContent}

// RESEARCH V9
// Bayesian Online Changepoint Detection (BOCPD) for categorical digits.
// Dirichlet-categorical posterior within each possible current run.
// Hierarchical transition row shrunk toward the current regime distribution.
// Bayes-factor evidence against a uniform 10% transition row.
// Candidate selection uses posterior probability P(p_match < 0.10 | data).
// Contract horizon is exactly T+1 (one tick).

function logGamma(z){
 const c=[
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.6150291621406,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7
 ];
 if(z<.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);
 z-=1;
 let x=c[0];
 for(let i=1;i<c.length;i++)x+=c[i]/(z+i);
 const t=z+c.length-1.5;
 return .5*Math.log(2*Math.PI)+(z+.5)*Math.log(t)-t+Math.log(x);
}

function betaCF(a,b,x){
 const MAX=120,EPS=3e-10,FPMIN=1e-30;
 const qab=a+b,qap=a+1,qam=a-1;
 let c=1,d=1-qab*x/qap;
 if(Math.abs(d)<FPMIN)d=FPMIN;
 d=1/d;
 let h=d;
 for(let m=1;m<=MAX;m++){
  const m2=2*m;
  let aa=m*(b-m)*x/((qam+m2)*(a+m2));
  d=1+aa*d;if(Math.abs(d)<FPMIN)d=FPMIN;
  c=1+aa/c;if(Math.abs(c)<FPMIN)c=FPMIN;
  d=1/d;h*=d*c;

  aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
  d=1+aa*d;if(Math.abs(d)<FPMIN)d=FPMIN;
  c=1+aa/c;if(Math.abs(c)<FPMIN)c=FPMIN;
  d=1/d;
  const del=d*c;
  h*=del;
  if(Math.abs(del-1)<EPS)break;
 }
 return h;
}

function ibeta(x,a,b){
 if(x<=0)return 0;
 if(x>=1)return 1;
 const bt=Math.exp(logGamma(a+b)-logGamma(a)-logGamma(b)+a*Math.log(x)+b*Math.log(1-x));
 if(x<(a+1)/(a+b+2))return bt*betaCF(a,b,x)/a;
 return 1-bt*betaCF(b,a,1-x)/b;
}

function boCPD(seq){
 const K=10,alpha=.5;
 const hazard=1/110;
 let states=[{p:1,len:0,c:Array(K).fill(0)}];

 for(const x of seq){
  const next=[];
  const basePred=.1;
  let changeMass=hazard*basePred;

  const cc=Array(K).fill(0);
  cc[x]=1;
  next.push({p:changeMass,len:1,c:cc});

  for(const st of states){
   const pred=(st.c[x]+alpha)/(st.len+K*alpha);
   const mass=st.p*(1-hazard)*pred;
   const nc=st.c.slice();
   nc[x]++;
   next.push({p:mass,len:st.len+1,c:nc});
  }

  let z=next.reduce((s,r)=>s+r.p,0);
  if(!Number.isFinite(z)||z<=0)z=1;
  for(const r of next)r.p/=z;

  // Keep the posterior compact on mobile without discarding material mass.
  next.sort((a,b)=>b.p-a.p);
  states=next.slice(0,180);
  const zr=states.reduce((s,r)=>s+r.p,0)||1;
  for(const r of states)r.p/=zr;
 }

 const mix=Array(K).fill(0);
 const second=Array(K).fill(0);
 const under10=Array(K).fill(0);
 let expectedRun=0,cpProb=0,runEntropy=0;

 for(const st of states){
  expectedRun+=st.p*st.len;
  if(st.len===1)cpProb+=st.p;
  if(st.p>0)runEntropy-=st.p*Math.log(st.p);

  const A=st.len+K*alpha;
  for(let d=0;d<K;d++){
   const ad=st.c[d]+alpha,bd=A-ad;
   const m=ad/A;
   const v=ad*bd/(A*A*(A+1));
   mix[d]+=st.p*m;
   second[d]+=st.p*(v+m*m);
   under10[d]+=st.p*ibeta(.10,ad,bd);
  }
 }

 const variance=second.map((x,d)=>Math.max(0,x-mix[d]*mix[d]));
 const entropyNorm=states.length>1?runEntropy/Math.log(states.length):0;

 return{states,mix,variance,under10,expectedRun,cpProb,runEntropy:entropyNorm};
}

function transitionStats(windowLen,baseP){
 const K=10,lambda=14;
 const prev=hist[hist.length-1];
 const start=Math.max(0,hist.length-windowLen-1);
 const counts=Array(K).fill(0);
 let n=0;

 for(let i=start;i<hist.length-1;i++){
  if(hist[i]===prev){
   counts[hist[i+1]]++;
   n++;
  }
 }

 const A=n+lambda;
 const mean=Array(K),variance=Array(K),under10=Array(K);
 for(let d=0;d<K;d++){
  const ad=counts[d]+lambda*baseP[d];
  const bd=A-ad;
  mean[d]=ad/A;
  variance[d]=ad*bd/(A*A*(A+1));
  under10[d]=ibeta(.10,Math.max(ad,1e-6),Math.max(bd,1e-6));
 }

 // Bayes factor: transition row with symmetric Dirichlet prior vs uniform 10%.
 const a=.5;
 let logAlt=logGamma(K*a)-logGamma(K*a+n);
 for(let d=0;d<K;d++)logAlt+=logGamma(a+counts[d])-logGamma(a);
 const logNull=n*Math.log(.1);
 const logBF=logAlt-logNull;

 return{prev,counts,n,mean,variance,under10,logBF};
}

function analyse(){
 if(hist.length<280)return null;

 const seq=hist.slice(-240);
 const bo=boCPD(seq);

 // BOCPD supplies the regime length instead of imposing a fixed window.
 const windowLen=Math.max(25,Math.min(180,Math.round(bo.expectedRun)));
 const tr=transitionStats(windowLen,bo.mix);

 // If a changepoint is plausible or transition support is weak,
 // shrink more strongly to the BOCPD regime predictive distribution.
 const supportWeight=tr.n/(tr.n+14);
 const regimeTrust=(1-bo.cpProb)*(1-.55*bo.runEntropy);
 const rowWeight=clamp01(supportWeight*regimeTrust);

 const rows=[];
 for(let d=0;d<10;d++){
  const mean=rowWeight*tr.mean[d]+(1-rowWeight)*bo.mix[d];

  const second=
   rowWeight*(tr.variance[d]+tr.mean[d]*tr.mean[d])+
   (1-rowWeight)*(bo.variance[d]+bo.mix[d]*bo.mix[d]);
  const variance=Math.max(0,second-mean*mean);
  const sd=Math.sqrt(variance);

  const pBelow=
   rowWeight*tr.under10[d]+
   (1-rowWeight)*bo.under10[d];

  // Conservative MATCH risk: posterior mean + uncertainty allowance.
  const conservativeRisk=
   mean+
   .80*sd+
   .005*bo.cpProb+
   .004*bo.runEntropy;

  rows.push({
   d,
   pt:mean,
   risk:conservativeRisk,
   sd,
   pBelow,
   support:tr.n,
   logBF:tr.logBF,
   rowWeight
  });
 }

 // Prefer posterior evidence that MATCH probability is below 10%;
 // break ties using the conservative upper risk.
 rows.sort((a,b)=>b.pBelow-a.pBelow||a.risk-b.risk||a.pt-b.pt);
 const q=rows[0],second=rows[1];
 if(!q||!second)return null;

 const spread=Math.max(0,q.pBelow-second.pBelow);
 const bfStrength=clamp01((tr.logBF+1.5)/5);
 const supportScore=clamp01(tr.n/28);
 const regimeScore=clamp01(1-bo.cpProb);
 const runCertainty=clamp01(1-bo.runEntropy);

 const quality=100*(
  .40*q.pBelow+
  .20*clamp01((.105-q.risk)/.025)+
  .15*bfStrength+
  .10*supportScore+
  .08*regimeScore+
  .07*runCertainty
 );

 const safe=
  q.pBelow>=.68&&
  q.risk<.104&&
  q.pt<.102&&
  tr.n>=4&&
  bo.cpProb<.65;

 return{
  q,second,spread,quality,safe,
  cpProb:bo.cpProb,
  expectedRun:bo.expectedRun,
  runEntropy:bo.runEntropy,
  windowLen,
  logBF:tr.logBF,
  transitionSupport:tr.n,
  rowWeight
 };
}


function resetCycle(anchor=null){
 lastCandidateSeen=anchor;
 lastDecision=null;
 if($('evidenceCandidates'))$('evidenceCandidates').textContent='V10 · esperando señal bayesiana válida.';
}
function showSignal(s){
 lastSignal=s;
 const buy=$('buy');
 if(!s){
  $('decision').textContent=running?'V10 · CALIBRANDO':'AUTO DETENIDO';
  $('reason').textContent=running?'Inicializando posterior bayesiano.':'Pulsa INICIAR AUTO para comenzar.';
  if($('sepPick'))$('sepPick').textContent='—';
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='V10 · esperando historial suficiente.';
  buy.textContent=running?'AUTO V10 · CALIBRANDO':'INICIAR AUTO';
  buy.disabled=running;
  return;
 }
 const q=s.q;
 $('risk').textContent=(q.risk*100).toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+q.d;
 $('spread').textContent=(s.spread*100).toFixed(1);
 $('entropy').textContent=s.runEntropy.toFixed(2);
 $('phase').textContent=s.cpProb>.35?'CAMBIO':'RÉGIMEN';
 $('meter').style.width=Math.min(100,s.quality)+'%';
 if($('finalScore'))$('finalScore').textContent=s.quality.toFixed(0)+'/100';
 if($('delayState'))$('delayState').textContent='T+1';
 if($('regimeState'))$('regimeState').textContent='CP '+(s.cpProb*100).toFixed(0)+'%';
 if($('techScore'))$('techScore').textContent='P<10 '+(q.pBelow*100).toFixed(0)+'%';
 if($('techRsi'))$('techRsi').textContent=(q.pt*100).toFixed(1)+'%';
 if($('techMacd'))$('techMacd').textContent='BF '+s.logBF.toFixed(1);
 if($('techTrend'))$('techTrend').textContent='RUN '+s.expectedRun.toFixed(0)+' · N'+s.transitionSupport;

 if(pending){
  if($('evidenceCandidates'))$('evidenceCandidates').textContent='OPERACIÓN EN CURSO · esperando liquidación Deriv';
  $('decision').textContent='OPERACIÓN EN CURSO · DIFFER D'+pending.d;
  $('reason').textContent='No se abre otra operación hasta recibir WIN o MATCH.';
  buy.textContent='OPERACIÓN EN CURSO';
  buy.disabled=true;
  return;
 }

 if($('evidenceCandidates')){
  $('evidenceCandidates').textContent=
   'D'+q.d+
   ' · P(MATCH<10%) '+(q.pBelow*100).toFixed(0)+'%'+
   ' · RIESGO '+(q.risk*100).toFixed(1)+'%'+
   ' · '+(s.safe?'SEÑAL VÁLIDA':'ESPERAR');
 }

 if(!running){
  $('decision').textContent='V10 · CANDIDATO D'+q.d;
  $('reason').textContent='BOCPD + Dirichlet + Bayes Factor · AUTO detenido.';
  buy.textContent='INICIAR AUTO';
  buy.disabled=false;
 }else if(s.safe){
  $('decision').textContent='V10 · COMPRA D'+q.d;
  $('reason').textContent='La señal cumple el filtro bayesiano completo.';
  buy.textContent='AUTO · COMPRA D'+q.d;
  buy.disabled=true;
 }else{
  $('decision').textContent='V10 · ESPERANDO';
  $('reason').textContent='El candidato actual todavía no cumple el filtro bayesiano.';
  buy.textContent='AUTO · ESPERANDO SEÑAL VÁLIDA';
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
 log('V10 BOCPD · COMPRA D'+d+' · Q'+s.quality.toFixed(0)+' · $'+stake.toFixed(2));
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
  if(s.q.d!==lastCandidateSeen){
   candidateEvent=true;
   lastCandidateSeen=s.q.d;
  }
 }

 lastDecision=s?{buy:s.safe,score:s.quality}:null;
 showSignal(s);

 // AUTO DIRECTO: compra cada nueva señal bayesiana que cumpla el filtro safe.
 if(running&&!pending&&s&&candidateEvent&&s.safe){
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
 lastPick=null;lastSignal=null;candidateHistory=[];lastCandidateSeen=null;lastDecision=null;running=true;

 $('status').textContent='V10 ACTIVA · AUTO BAYESIANO DIRECTO · T+1';
 $('buy').textContent='AUTO V10 · ESPERANDO SEÑAL';
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
