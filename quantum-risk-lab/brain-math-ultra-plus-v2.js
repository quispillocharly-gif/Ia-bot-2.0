const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],quotes=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,candidateHistory=[];
let tickCounter=0,arming=null,cooldownTicks=0,deteriorationTicks=0,shadowQueue=[];
let shadowStats={rawWins:0,rawLosses:0,confirmedWins:0,confirmedLosses:0,confirmedRecent:[]};
try{let z=JSON.parse(localStorage.getItem('quantumShadowMathUltra2')||'null');if(z&&typeof z==='object')shadowStats={...shadowStats,...z}}catch(_){}
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}

function clamp(x,a=0,b=1){return Math.max(a,Math.min(b,x))}
function logSafe(x){return Math.log(Math.max(x,1e-15))}
function renyi2(dist){
 let s=dist.reduce((z,p)=>z+p*p,0);
 return -Math.log2(Math.max(s,1e-15));
}
function entropyNorm(dist){
 let h=0;
 for(const p of dist)if(p>0)h-=p*Math.log2(p);
 return h/Math.log2(10);
}
function jsDivergence(p,q){
 let m=p.map((x,i)=>(x+q[i])/2),kl=(a,b)=>{
  let s=0;
  for(let i=0;i<a.length;i++)if(a[i]>0)s+=a[i]*Math.log(a[i]/Math.max(b[i],1e-15));
  return s;
 };
 return .5*kl(p,m)+.5*kl(q,m);
}
function digitDistribution(a,alpha=.5){
 let c=Array(10).fill(alpha),n=10*alpha;
 for(const d of a){c[d]++;n++}
 return c.map(x=>x/n);
}
function pairDistribution(a,alpha=.05){
 let c=Array(100).fill(alpha),n=100*alpha;
 for(let i=1;i<a.length;i++){c[a[i-1]*10+a[i]]++;n++}
 return c.map(x=>x/n);
}
function regimeMath(){
 if(hist.length<180)return{shift:1,score:0,gate:false};
 let recent=hist.slice(-70),long=hist.slice(-260),
     older=long.slice(0,Math.max(1,long.length-recent.length)),
     dm=jsDivergence(digitDistribution(recent),digitDistribution(older)),
     dp=jsDivergence(pairDistribution(recent),pairDistribution(older)),
     shift=.4*dm+.6*dp,
     score=100*(1-clamp(shift/.12)),
     gate=shift<.095;
 return{shift,score,gate};
}
function ctxKey(seq,i,depth){
 if(depth===0)return'*';
 if(i<depth)return null;
 return seq.slice(i-depth,i).join('');
}
function tailKey(tail,depth){
 if(depth===0)return'*';
 if(tail.length<depth)return null;
 return tail.slice(-depth).join('');
}
function makeExpert(seq,depth,window,evalN=140){
 const alpha=.5,start=Math.max(0,seq.length-window),table=new Map();
 let loss=0,evals=0;
 for(let i=start;i<seq.length;i++){
  let k=ctxKey(seq,i,depth);
  if(k===null)continue;
  let node=table.get(k);
  if(!node){node={c:Array(10).fill(0),n:0};table.set(k,node)}
  let y=seq[i],p=(node.c[y]+alpha)/(node.n+10*alpha);
  if(i>=Math.max(start,seq.length-evalN)){loss-=logSafe(p);evals++}
  node.c[y]++;node.n++;
 }
 let prior=1/Math.pow(depth+1,1.15);
 return{depth,window,table,loss,evals,prior,avgLoss:loss/Math.max(1,evals)};
}
function expertDist(expert,tail){
 const alpha=.5,k=tailKey(tail,expert.depth);
 if(k===null)return{p:Array(10).fill(.1),support:0};
 let node=expert.table.get(k);
 if(!node)return{p:Array(10).fill(.1),support:0};
 let den=node.n+10*alpha;
 return{p:node.c.map(x=>(x+alpha)/den),support:node.n/(node.n+18)};
}
function expertWeights(experts){
 let min=Math.min(...experts.map(e=>e.avgLoss)),eta=3.2,raw=[];
 for(const e of experts){
  let z=e.prior*Math.exp(-eta*(e.avgLoss-min));
  raw.push(Number.isFinite(z)?z:0);
 }
 let s=raw.reduce((a,b)=>a+b,0)||1;
 return raw.map(x=>x/s);
}
function mixtureDist(experts,weights,tail){
 let p=Array(10).fill(0),support=0;
 for(let i=0;i<experts.length;i++){
  let z=expertDist(experts[i],tail),w=weights[i];
  support+=w*z.support;
  for(let d=0;d<10;d++)p[d]+=w*z.p[d];
 }
 return{p,support};
}
function forecastWith(experts,weights,tail,steps,maxDepth=3){
 let states=new Map([[tail.slice(-maxDepth).join(''),{tail:tail.slice(-maxDepth),prob:1}]]);
 let horizons=[];
 for(let step=1;step<=steps;step++){
  let next=new Map(),agg=Array(10).fill(0);
  for(const st of states.values()){
   let dist=mixtureDist(experts,weights,st.tail).p;
   for(let d=0;d<10;d++){
    let pr=st.prob*dist[d];
    if(pr<1e-12)continue;
    agg[d]+=pr;
    let nt=st.tail.concat(d).slice(-maxDepth),key=nt.join('');
    let old=next.get(key);
    if(old)old.prob+=pr;else next.set(key,{tail:nt,prob:pr});
   }
  }
  horizons.push(agg);
  states=next;
 }
 return horizons;
}
function forecastSingle(expert,tail,steps,maxDepth=3){
 return forecastWith([expert],[1],tail,steps,maxDepth)[steps-1];
}
function weightedStd(vals,weights,mean){
 let s=0;
 for(let i=0;i<vals.length;i++)s+=weights[i]*(vals[i]-mean)*(vals[i]-mean);
 return Math.sqrt(Math.max(0,s));
}
function buildMathModel(seq){
 const depths=[0,1,2,3],windows=[120,240,480],experts=[];
 for(const w of windows)for(const d of depths)experts.push(makeExpert(seq,d,w,140));
 let weights=expertWeights(experts),tail=seq.slice(-3),
     horizons=forecastWith(experts,weights,tail,3,3),
     p1=horizons[0],p2=horizons[1],p3=horizons[2],
     p1Experts=experts.map(e=>forecastSingle(e,tail,1,3)),
     p2Experts=experts.map(e=>forecastSingle(e,tail,2,3)),
     support=mixtureDist(experts,weights,tail).support,
     eff=1/weights.reduce((s,w)=>s+w*w,0),
     bestI=weights.indexOf(Math.max(...weights));
 return{
  experts,weights,p1,p2,p3,p1Experts,p2Experts,
  support,eff,best:experts[bestI],bestWeight:weights[bestI]
 };
}
function mathScore(row,model,regime,hNorm){
 let edge=clamp((.10-row.upperRisk)/.016),
     meanEdge=clamp((.10-row.pt)/.018),
     cert=1-clamp(row.disagreement/.018),
     horizon=1-clamp(row.horizonDisagreement/.020),
     support=clamp(model.support/.72),
     stable=clamp(regime.score/100),
     modelMix=clamp((model.eff-1)/5),
     info=clamp((1-hNorm)/.025);
 return 100*(.24*edge+.14*meanEdge+.16*cert+.13*horizon+.11*support+.10*stable+.07*modelMix+.05*info);
}
function saveShadow(){
 try{localStorage.setItem('quantumShadowMathUltra2',JSON.stringify(shadowStats))}catch(_){}
}
function queueShadow(d,type,due){
 shadowQueue.push({d,type,due});
 if(shadowQueue.length>120)shadowQueue.shift();
}
function settleShadow(actual){
 let keep=[];
 for(const x of shadowQueue){
  if(x.due>tickCounter){keep.push(x);continue}
  let win=actual!==x.d;
  if(x.type==='confirmed'){
   if(win)shadowStats.confirmedWins++;else shadowStats.confirmedLosses++;
   shadowStats.confirmedRecent.push(win?1:0);
   if(shadowStats.confirmedRecent.length>20)shadowStats.confirmedRecent.shift();
  }else{
   if(win)shadowStats.rawWins++;else shadowStats.rawLosses++;
  }
 }
 shadowQueue=keep;
 let r=shadowStats.confirmedRecent;
 if(r.length>=20){
  let losses=r.filter(x=>x===0).length;
  if(losses>=5)deteriorationTicks=Math.max(deteriorationTicks,8);
 }
 saveShadow();
}

function analyse(){
 if(hist.length<240)return null;

 // MATH SUPER ULTRA PLUS v2 · DELAY REAL 1 TICK
 // Nueva preseñal en T: pronostica T+2.
 // En T+1 se vuelve a validar el MISMO dígito para T+2 usando pronóstico T+1.
 const seq=hist.slice(-900),model=buildMathModel(seq),regime=regimeMath(),
       p1=model.p1,p2=model.p2,p3=model.p3,
       H=renyi2(p2),hNorm=H/Math.log2(10),rows=[],recheck=[];

 for(let d=0;d<10;d++){
  // Candidato inicial: objetivo T+2.
  let vals=model.p2Experts.map(x=>x[d]),
      disagreement=weightedStd(vals,model.weights,p2[d]),
      hMean=(p1[d]+2*p2[d]+p3[d])/4,
      horizonDisagreement=Math.sqrt(((p1[d]-hMean)**2+(p2[d]-hMean)**2+(p3[d]-hMean)**2)/3),
      upperRisk=p2[d]+.55*disagreement+.35*horizonDisagreement,
      lowerRisk=Math.max(0,p2[d]-.55*disagreement),
      row={
       d,risk:upperRisk,upperRisk,lowerRisk,
       pt:p2[d],p1:p1[d],p3:p3[d],
       disagreement,horizonDisagreement,
       support:model.support,effModels:model.eff
      };
  row.score=mathScore(row,model,regime,hNorm);
  rows.push(row);

  // Revalidación un tick después: ahora el mismo objetivo está a T+1.
  let vals1=model.p1Experts.map(x=>x[d]),
      disagreement1=weightedStd(vals1,model.weights,p1[d]),
      bridge=Math.abs(p1[d]-p2[d]),
      upperRisk1=p1[d]+.55*disagreement1+.25*bridge,
      row1={
       d,risk:upperRisk1,upperRisk:upperRisk1,
       lowerRisk:Math.max(0,p1[d]-.55*disagreement1),
       pt:p1[d],p1:p1[d],p3:p2[d],
       disagreement:disagreement1,
       horizonDisagreement:bridge,
       support:model.support,effModels:model.eff
      };
  row1.score=mathScore(row1,model,regime,hNorm);
  row1.safe=true;
  recheck[d]=row1;
 }

 rows.sort((a,b)=>a.upperRisk-b.upperRisk||a.pt-b.pt);
 let q=rows[0],second=rows[1];
 if(!q||!second)return null;

 let spread=Math.max(0,second.upperRisk-q.upperRisk),
     finalScore=q.score,
     baseSafe=true,
     regimeSafe=true,
     safe=true,
     leader='D'+model.best.depth+' W'+model.best.window;

 return{
  q,spread,H,near:pnl>=target()*.75,
  safe,baseSafe,techSafe:true,regimeSafe,
  tech:null,regime,finalScore,recheck,
  math:{
   eff:model.eff,
   support:model.support,
   bestDepth:model.best.depth,
   bestWindow:model.best.window,
   bestWeight:model.bestWeight,
   leader,hNorm
  }
 };
}
function showSignal(raw,ready,waitReason=''){
 const buy=$('buy');
 lastSignal=ready||null;

 if(!raw){
  $('decision').textContent='MATH ULTRA · ANALIZANDO';
  $('reason').textContent=waitReason||'KT + contextos variables + mezcla por log-loss.';
  $('sepPick').textContent='—';
  $('risk').textContent='—';$('spread').textContent='—';$('entropy').textContent='—';
  $('phase').textContent='OBSERVAR';$('meter').style.width='0%';
  buy.textContent='ESPERANDO SEÑAL';buy.disabled=true;
  if($('techScore'))$('techScore').textContent='—';
  if($('techRsi'))$('techRsi').textContent='—';
  if($('techMacd'))$('techMacd').textContent='—';
  if($('techTrend'))$('techTrend').textContent='—';
  if($('finalScore'))$('finalScore').textContent='—';
  if($('regimeState'))$('regimeState').textContent='—';
  updateShadowUI();
  return;
 }

 $('risk').textContent=(raw.q.upperRisk*100).toFixed(2)+'%';
 $('spread').textContent=(raw.spread*100).toFixed(2);
 $('entropy').textContent=raw.H.toFixed(3);
 $('phase').textContent=raw.near?'MODO META':'MATH ULTRA';

 if($('techScore'))$('techScore').textContent=raw.math.eff.toFixed(1);
 if($('techRsi'))$('techRsi').textContent=(raw.q.pt*100).toFixed(2)+'%';
 if($('techMacd'))$('techMacd').textContent=(raw.q.disagreement*100).toFixed(2)+'%';
 if($('techTrend'))$('techTrend').textContent=raw.math.leader;
 if($('finalScore'))$('finalScore').textContent=raw.finalScore.toFixed(0)+'/100';
 if($('regimeState'))$('regimeState').textContent=raw.regime.gate?'ESTABLE':'CAMBIO';
 updateShadowUI();

 if(pending){
  $('sepPick').textContent='—';
  $('decision').textContent='OPERACIÓN EN CURSO';
  $('reason').textContent='Esperando liquidación.';
  buy.textContent='OPERACIÓN EN CURSO';buy.disabled=true;
  return;
 }

 if(!ready){
  $('sepPick').textContent='—';
  $('decision').textContent='MATH ULTRA · ANALIZANDO';
  $('reason').textContent=waitReason||'Todavía no hay señal matemática completa.';
  buy.textContent='ESPERANDO SEÑAL';buy.disabled=true;
  $('meter').style.width=Math.min(95,raw.finalScore)+'%';
  return;
 }

 $('sepPick').textContent='D'+ready.q.d;
 $('decision').textContent='MATH ULTRA · SEÑAL D'+ready.q.d;
 $('reason').textContent='Lógica matemática + T+2 · score informativo '+ready.finalScore.toFixed(0)+'/100.';
 buy.textContent='COMPRAR AHORA · D'+ready.q.d+' · RIESGO '+(ready.q.upperRisk*100).toFixed(2)+'%';
 buy.disabled=false;
 $('meter').style.width='100%';
}
function updateShadowUI(){
 if(!$('shadowState'))return;
 let w=shadowStats.confirmedWins,l=shadowStats.confirmedLosses,n=w+l;
 $('shadowState').textContent=n?((100*w/n).toFixed(1)+'% · '+n+' señales'):'SIN DATOS';
}
function ui(d){
 if(d!==undefined)$('tick').textContent='D'+d;
 $('pnl').textContent=(pnl>=0?'+':'')+'$'+pnl.toFixed(2);$('stake').textContent='$'+stake.toFixed(2);$('wins').textContent=wins;$('losses').textContent=losses;$('ops').textContent=ops;$('pick').textContent=lastPick===null?'—':'D'+lastPick;
}
function enter(s){
 if(!running||pending||!s||!s.safe||s!==lastSignal){$('status').textContent='ESPERANDO SEÑAL COMPLETA';return}
 let d=s.q.d,mode=$('mode').value;
 if(mode==='DEMO'&&!window.demoReady){$('status').textContent='CONECTA DEMO DERIV';return}
 lastPick=d;observe=0;pending={d,stake,mode};ops++;$('buy').disabled=true;
 $('decision').textContent='COMPRA '+mode+' · DIFFER D'+d;$('reason').textContent='$'+stake.toFixed(2)+' · duración 1 tick';log('COMPRA '+mode+' D'+d+' $'+stake.toFixed(2));ui();
 if(mode==='DEMO'){ $('status').textContent='ENVIANDO A DERIV…'; window.sendDemoTrade(d,stake).catch(e=>tradeError(e));}
 else $('status').textContent='SIM ABIERTA';
}
function tradeError(e){
 log('ERROR DEMO '+(e?.message||e));pending=null;observe=0;$('status').textContent='ERROR DEMO · REVISA LOG';ui();
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

 pending=null;
 observe=0;
 arming=null;
 lastSignal=null;
 $('buy').disabled=true;
 $('buy').textContent='ESPERANDO SEÑAL';
 $('sepPick').textContent='—';

 if(pnl>=target()){
  running=false;
  $('status').textContent='META +$'+target().toFixed(2)+' · STOP';
  $('phase').textContent='META';
 }else if(running){
  $('status').textContent=profit>0?'REVALIDANDO':'RECALIBRANDO TRAS MATCH';
 }
 ui();
}
function tick(d,price){
 tickCounter++;
 settleShadow(d);

 if(pending&&pending.mode==='SIM'){
  let p=pending;
  finish(d===p.d?-p.stake:p.stake*.10,'SIM');
 }

 hist.push(d);if(hist.length>1000)hist.shift();
 if(Number.isFinite(price)){quotes.push(price);if(quotes.length>1000)quotes.shift()}
 if(running&&!pending)observe++;
 ui(d);

 let raw=analyse(),ready=null,reason='';

 if(raw){
  candidateHistory.push(raw.q.d);
  if(candidateHistory.length>40)candidateHistory.shift();
 }

 if(!running){
  arming=null;
  reason='STOP MANUAL';
 }else if(pending){
  arming=null;
  reason='Operación en curso.';
 }else if(!raw){
  arming=null;
  reason='Recolectando historial matemático.';
 }else if(!arming){
  arming={d:raw.q.d,confirmedShadow:false};
  reason='Mejor dígito calculado · esperando 1 tick real.';
 }else if(raw.q.d===arming.d){
  ready={...raw,safe:true};
  reason='Mismo mejor dígito confirmado tras 1 tick real.';
  if(!arming.confirmedShadow){
   queueShadow(arming.d,'confirmed',tickCounter+1);
   arming.confirmedShadow=true;
  }
 }else{
  arming={d:raw.q.d,confirmedShadow:false};
  reason='Cambió el mejor dígito · nuevo delay de 1 tick.';
 }

 if($('delayState'))$('delayState').textContent=ready?'1/1 LISTO':arming?'0/1':'—';
 showSignal(raw,ready,reason);
}
function connect(){
 clearTimeout(retry);ws=new WebSocket('wss://api.derivws.com/trading/v1/options/ws/public');
 ws.onopen=()=>ws.send(JSON.stringify({ticks_history:'R_75',count:300,end:'latest',style:'ticks'}));
 ws.onmessage=e=>{let m=JSON.parse(e.data);
  if(m.history&&m.history.prices){let p=Number(m.pip_size||4);quotes=m.history.prices.map(Number);hist=quotes.map(x=>Number(Number(x).toFixed(p).slice(-1)));ws.send(JSON.stringify({ticks:'R_75',subscribe:1}));$('status').textContent='LISTO · '+hist.length+' TICKS · HÍBRIDO'}
  if(m.tick){let ep=+m.tick.epoch;if(ep===lastEpoch)return;lastEpoch=ep;let p=Number(m.tick.pip_size||4),price=Number(m.tick.quote),d=Number(price.toFixed(p).slice(-1));tick(d,price)}
 };
 ws.onclose=()=>retry=setTimeout(connect,2500);
}
$('start').onclick=()=>{
 if($('mode').value==='DEMO'&&!window.demoReady){$('status').textContent='CONECTA DEMO DERIV PRIMERO';return}
 pnl=0;stake=baseStake();wins=0;losses=0;ops=0;pending=null;observe=0;lastPick=null;lastSignal=null;candidateHistory=[];arming=null;cooldownTicks=0;deteriorationTicks=0;running=true;$('status').textContent='ANALIZANDO';log('NUEVA SESIÓN '+$('mode').value+' · STAKE $'+stake.toFixed(2)+' · META $'+target().toFixed(2));ui();
};
$('stop').onclick=()=>{running=false;$('status').textContent='STOP MANUAL'};
$('buy').onclick=()=>{if(!running){$('status').textContent='PULSA REINICIAR SESIÓN';return}if(pending){$('status').textContent='OPERACIÓN EN CURSO';return}let s=lastSignal;if(!s){$('status').textContent='AÚN CALIBRANDO';return}enter(s)};
window.demoSettlement=p=>finish(p,'DERIV DEMO');
window.demoTradeError=tradeError;
stake=baseStake();$('status').textContent='MATH ULTRA SIMPLE · LÓGICA + 1T';$('buy').disabled=true;ui();updateShadowUI();connect();
