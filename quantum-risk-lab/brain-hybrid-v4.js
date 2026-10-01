const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],quotes=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,candidateHistory=[];
let tickCounter=0,arming=null,cooldownTicks=0,deteriorationTicks=0,shadowQueue=[];
let shadowStats={rawWins:0,rawLosses:0,confirmedWins:0,confirmedLosses:0,confirmedRecent:[]};
try{let z=JSON.parse(localStorage.getItem('quantumShadowV4')||'null');if(z&&typeof z==='object')shadowStats={...shadowStats,...z}}catch(_){}
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}

function avg(a){return a.length?a.reduce((s,x)=>s+x,0)/a.length:0}
function sd(a){if(a.length<2)return 0;let m=avg(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/a.length)}
function emaSeries(a,p){if(!a.length)return[];let k=2/(p+1),o=[a[0]];for(let i=1;i<a.length;i++)o.push(a[i]*k+o[i-1]*(1-k));return o}
function calcRSI(a,p=14){
 if(a.length<=p)return 50;
 let g=0,l=0;
 for(let i=a.length-p;i<a.length;i++){let d=a[i]-a[i-1];if(d>0)g+=d;else l-=d}
 if(l===0)return g===0?50:100;
 let rs=(g/p)/(l/p);return 100-(100/(1+rs));
}
function calcMACD(a){
 let e12=emaSeries(a,12),e26=emaSeries(a,26),line=[];
 for(let i=0;i<a.length;i++)line.push((e12[i]??a[i])-(e26[i]??a[i]));
 let sig=emaSeries(line,9),i=line.length-1;
 return{line:line[i]||0,signal:sig[i]||0,hist:(line[i]||0)-(sig[i]||0)};
}
function technicalSnapshot(a){
 if(a.length<60)return null;
 let last=a[a.length-1],
     e9=emaSeries(a,9),e21=emaSeries(a,21),e50=emaSeries(a,50),
     ema9=e9[e9.length-1],ema21=e21[e21.length-1],ema50=e50[e50.length-1],
     R=calcRSI(a,14),M=calcMACD(a),
     w20=a.slice(-20),mid=avg(w20),dev=sd(w20),
     upper=mid+2*dev,lower=mid-2*dev,
     bbPos=dev?(last-mid)/(2*dev):0,
     diffs=[];
 for(let i=Math.max(1,a.length-40);i<a.length;i++)diffs.push(a[i]-a[i-1]);
 let vol=sd(diffs),scale=Math.max(vol,Math.abs(last)*1e-7,1e-8),
     eShort=(ema9-ema21)/scale,
     eLong=(ema21-ema50)/scale,
     macdN=M.hist/scale,
     momentum=(last-a[Math.max(0,a.length-11)])/scale,
     bull=0,bear=0;
 if(ema9>ema21)bull++;else if(ema9<ema21)bear++;
 if(ema21>ema50)bull++;else if(ema21<ema50)bear++;
 if(M.hist>0)bull++;else if(M.hist<0)bear++;
 if(R>52)bull++;else if(R<48)bear++;
 if(momentum>0)bull++;else if(momentum<0)bear++;
 let trend=bull>=4&&bull>bear?'ALCISTA':bear>=4&&bear>bull?'BAJISTA':'LATERAL',
     alignment=Math.max(bull,bear)/5,
     emaStrength=Math.min(1,(Math.abs(eShort)+Math.abs(eLong))/4),
     macdStrength=Math.min(1,Math.abs(macdN)/2),
     rsiStrength=Math.min(1,Math.abs(R-50)/25),
     momentumStrength=Math.min(1,Math.abs(momentum)/4),
     bbStrength=Math.min(1,Math.abs(bbPos)),
     score=100*(.28*alignment+.24*emaStrength+.20*macdStrength+.12*rsiStrength+.10*momentumStrength+.06*bbStrength);
 score=Math.max(0,Math.min(100,score));
 let extreme=R>=82||R<=18||Math.abs(bbPos)>=1.45;
 return{score,R,M,trend,ema9,ema21,ema50,bbPos,upper,lower,momentum,extreme,vol};
}
function technicalAnalysis(){
 if(quotes.length<60)return null;
 let a=quotes.slice(-600),
     now=technicalSnapshot(a);
 if(!now)return null;

 // COMPENSACIÓN TÉCNICA DEL DELAY DE 2 TICKS.
 // Se estima la deriva inmediata con los últimos cambios y se proyectan
 // dos ticks sintéticos. La señal técnica solo pasa si el contexto actual
 // y el contexto proyectado a +2 ticks siguen siendo compatibles.
 let diffs=[];
 for(let i=Math.max(1,a.length-9);i<a.length;i++)diffs.push(a[i]-a[i-1]);
 let weighted=0,weights=0;
 for(let i=0;i<diffs.length;i++){let w=i+1;weighted+=diffs[i]*w;weights+=w}
 let drift=weights?weighted/weights:0,
     cap=Math.max(now.vol*1.35,Math.abs(a[a.length-1])*1e-7,1e-8);
 drift=Math.max(-cap,Math.min(cap,drift));

 let p1=a[a.length-1]+drift,
     p2=p1+drift,
     future=technicalSnapshot(a.concat([p1,p2]));

 let trendCompatible=!!future&&(
   now.trend===future.trend ||
   now.trend==='LATERAL' ||
   future.trend==='LATERAL'
 );
 let gate=!!future &&
          now.score>=45 &&
          future.score>=45 &&
          !now.extreme &&
          !future.extreme &&
          trendCompatible;

 return{
  score:now.score,
  R:now.R,
  M:now.M,
  trend:now.trend,
  ema9:now.ema9,
  ema21:now.ema21,
  ema50:now.ema50,
  bbPos:now.bbPos,
  upper:now.upper,
  lower:now.lower,
  momentum:now.momentum,
  projectedScore:future?future.score:0,
  projectedTrend:future?future.trend:'—',
  projectedRSI:future?future.R:50,
  drift,
  delayTicks:2,
  gate
 };
}
function digitDistribution(a){
 let c=Array(10).fill(0),n=Math.max(1,a.length);
 a.forEach(d=>c[d]++);
 return c.map(x=>x/n);
}
function tvDistance(a,b){
 let s=0;
 for(let i=0;i<10;i++)s+=Math.abs(a[i]-b[i]);
 return .5*s;
}
function regimeAnalysis(){
 if(hist.length<300)return{score:0,shift:1,gate:false};
 let d50=digitDistribution(hist.slice(-50)),
     d100=digitDistribution(hist.slice(-100)),
     d300=digitDistribution(hist.slice(-300)),
     s1=tvDistance(d50,d100),
     s2=tvDistance(d100,d300),
     shift=.6*s1+.4*s2,
     score=Math.max(0,100*(1-Math.min(1,shift/.24))),
     gate=shift<.20;
 return{score,shift,gate};
}
function qualityScore(q,tech,regime){
 if(!q||!tech||!regime)return 0;
 let riskQ=Math.max(0,Math.min(1,(.10-q.risk)/.045)),
     probQ=Math.max(0,Math.min(1,(.10-q.pt)/.025)),
     supportQ=Math.max(0,Math.min(1,(q.support-.35)/.45)),
     horizonQ=Math.max(0,1-Math.min(1,q.horizonDisagreement/.035)),
     shiftQ=Math.max(0,1-Math.min(1,q.regimeShift/.045)),
     techQ=Math.max(0,Math.min(1,((tech.score+tech.projectedScore)/2)/100)),
     regimeQ=Math.max(0,Math.min(1,regime.score/100));
 return 100*(.20*riskQ+.12*probQ+.12*supportQ+.12*horizonQ+.10*shiftQ+.22*techQ+.12*regimeQ);
}
function saveShadow(){
 try{localStorage.setItem('quantumShadowV4',JSON.stringify(shadowStats))}catch(_){}
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
 if(hist.length<240||quotes.length<60)return null;

 // QUANTUM ANALYSIS V13 · V9 + REFUERZOS ADITIVOS
 // CADENA DE MARKOV DE PRIMER ORDEN + ENTROPÍA RÉNYI (alpha = 2)
 // Base original: Markov + Rényi. Refuerzo nuevo aditivo: EMA 9/21/50 + RSI 14 + MACD 12/26/9 + Bollinger.
 //
 // El modelo es estrictamente de primer orden:
 // P(X[t+1]=j | X[t]=i)
 //
 // Compensación de delay: 2 ticks reales.
 // La señal se proyecta 3 pasos hacia delante (t+3):
 // señal t -> delay 1 -> delay 2 -> tick objetivo t+3.
 // P³(i,d) = sum_j sum_k P(i,j) * P(j,k) * P(k,d)
 //
 // La Entropía Rényi de orden 2 mide cuán concentrada o uniforme es
 // la distribución proyectada. Si está demasiado cerca de uniforme,
 // el modelo pierde confianza y eleva el riesgo interno.

 const STATES=10;
 const ALPHA=.5; // suavizado Jeffreys/Dirichlet
 const RENYI_ALPHA=2;

 let n=hist.length,
     last=hist[n-1],
     counts=Array.from({length:STATES},()=>Array(STATES).fill(0)),
     rowN=Array(STATES).fill(0),
     recentCounts=Array.from({length:STATES},()=>Array(STATES).fill(0)),
     recentRowN=Array(STATES).fill(0);

 // Matriz de transición de primer orden con hasta 1000 transiciones recientes.
 for(let i=Math.max(0,n-1001);i<n-1;i++){
  let a=hist[i],b=hist[i+1];
  counts[a][b]++;
  rowN[a]++;
 }
 // Segunda matriz, solo reciente, para medir estabilidad del patrón.
 for(let i=Math.max(0,n-301);i<n-1;i++){
  let a=hist[i],b=hist[i+1];
  recentCounts[a][b]++;
  recentRowN[a]++;
 }

 function prob(a,b){
  return (counts[a][b]+ALPHA)/(rowN[a]+STATES*ALPHA);
 }
 function recentProb(a,b){
  return (recentCounts[a][b]+ALPHA)/(recentRowN[a]+STATES*ALPHA);
 }

 // Matriz P suavizada.
 let P=Array.from({length:STATES},(_,a)=>
  Array.from({length:STATES},(_,b)=>prob(a,b))
 );
 let PRecent=Array.from({length:STATES},(_,a)=>
  Array.from({length:STATES},(_,b)=>recentProb(a,b))
 );

 function advance(dist,matrix){
  let out=Array(STATES).fill(0);
  for(let d=0;d<STATES;d++){
   let s=0;
   for(let j=0;j<STATES;j++)s+=dist[j]*matrix[j][d];
   out[d]=s;
  }
  return out;
 }

 // P³ es el objetivo principal para compensar 2 ticks de delay.
 // P² y P⁴ solo validan la estabilidad alrededor del objetivo.
 let p1=P[last].slice(),
     p2=advance(p1,P),
     p3=advance(p2,P),
     p4=advance(p3,P);

 // Proyección P³ con la matriz reciente para detectar cambios de régimen.
 let rp1=PRecent[last].slice(),
     rp2=advance(rp1,PRecent),
     p3Recent=advance(rp2,PRecent);

 // Entropía Rényi H_alpha(p) = 1/(1-alpha) log2(sum p_i^alpha)
 // Para alpha=2: H2 = -log2(sum p_i^2)
 function renyi2(dist){
  let sumSq=dist.reduce((s,p)=>s+p*p,0);
  return -Math.log2(Math.max(sumSq,1e-12));
 }

 let H2=renyi2(p3),
     H2MAX=Math.log2(STATES),
     h2Norm=H2/H2MAX;

 // Entropía Rényi de la fila actual (1 paso) como segunda medida
 // de estructura del estado presente.
 let rowEntropy=renyi2(P[last]),
     rowEntropyNorm=rowEntropy/H2MAX;

 // Soporte estadístico del estado actual.
 let support=rowN[last]/(rowN[last]+25);

 // Confianza estructural: baja cuando la distribución es casi uniforme.
 let structure=Math.max(0,1-h2Norm),
     rowStructure=Math.max(0,1-rowEntropyNorm),
     confidence=Math.min(1,
       .55*support+
       .27*Math.min(1,structure/.08)+
       .18*Math.min(1,rowStructure/.08)
     );

 let rows=[];

 for(let d=0;d<STATES;d++){
  let targetP=p3[d];

  // Ventaja matemática frente al 10% teórico.
  // Un candidato con P³ menor que 0.10 recibe menor riesgo.
  let excess=Math.max(0,targetP-.10);

  // Cuando Rényi está muy cerca del máximo, la cadena parece casi uniforme.
  // El filtro evita convertir pequeñas diferencias aleatorias en señales fuertes.
  let entropyPenalty=Math.max(0,h2Norm-.965),
      rowEntropyPenalty=Math.max(0,rowEntropyNorm-.965);

  // Penalización por poco soporte de la fila del estado actual.
  let supportPenalty=1-support;

  // V13 · REFUERZOS ADITIVOS, sin sustituir P³:
  // 1) consenso entre P², P³ y P⁴;
  // 2) estabilidad entre la matriz larga y la matriz reciente;
  // 3) incertidumbre de la ruta Markov por soporte efectivo.
  let horizonMean=(p2[d]+2*targetP+p4[d])/4,
      horizonDisagreement=Math.sqrt(
       ((p2[d]-horizonMean)**2+
        (targetP-horizonMean)**2+
        (p4[d]-horizonMean)**2)/3
      ),
      regimeShift=Math.abs(targetP-p3Recent[d]);

  let routeSupport=0;
  for(let j=0;j<STATES;j++){
   routeSupport+=p2[j]*(rowN[j]/(rowN[j]+25));
  }
  let routeUncertainty=1-Math.min(1,routeSupport);

  // Riesgo interno V9 + capas V13 aditivas.
  // Escala calibrada para conservar el filtro visual existente <= 3.5%.
  let risk=.010+
           .145*targetP+
           .075*excess+
           .115*entropyPenalty+
           .055*rowEntropyPenalty+
           .008*supportPenalty+
           .006*(1-confidence)+
           .030*horizonDisagreement+
           .024*regimeShift+
           .004*routeUncertainty;

  rows.push({
   d,
   risk,
   p12:P[last][d],
   p36:targetP,
   p120:targetP,
   pt:targetP,
   pe:targetP,
   renyi:H2,
   renyiNorm:h2Norm,
   rowRenyi:rowEntropy,
   rowRenyiNorm:rowEntropyNorm,
   support,
   confidence,
   p2:p2[d],
   p4:p4[d],
   p3Recent:p3Recent[d],
   horizonDisagreement,
   regimeShift,
   routeUncertainty
  });
 }

 rows.sort((a,b)=>a.risk-b.risk);

 let pool=rows.filter(x=>x.d!==lastPick);
 if(pool.length<2)return null;

 // V10 · DIVERSIFICACIÓN MATEMÁTICA DE CANDIDATOS
 // Antes se elegía siempre el mínimo absoluto, lo que podía encerrar
 // la salida en 1-2 dígitos. Ahora:
 // - se consideran los 4 mejores según Markov + Rényi,
 // - se cuenta cuántas veces apareció cada uno como candidato en las
 //   últimas 24 decisiones,
 // - se aplica una penalización determinista por sobreuso.
 // No hay aleatoriedad: el riesgo Markov sigue siendo la base.
 let shortlist=pool.slice(0,Math.min(4,pool.length)),
     recent=candidateHistory.slice(-24),
     use=Array(10).fill(0);

 recent.forEach(d=>use[d]++);

 let lastCandidate=recent.length?recent[recent.length-1]:null;

 let ranked=shortlist.map(x=>{
  let repetitionPenalty=use[x.d]*.00135,
      immediatePenalty=x.d===lastCandidate?.0022:0,
      adjusted=x.risk+repetitionPenalty+immediatePenalty;
  return{x,adjusted};
 }).sort((a,b)=>a.adjusted-b.adjusted||a.x.risk-b.x.risk);

 let q=ranked[0].x,
     second=ranked[1]?ranked[1].x:pool.find(x=>x.d!==q.d);

 if(!q||!second)return null;

 let spread=Math.max(0,ranked[1]?ranked[1].adjusted-ranked[0].adjusted:Math.abs(second.risk-q.risk)),
     near=pnl>=target()*.75;

 // Filtro interno Rényi:
 // - riesgo Markov a t+3 razonablemente bajo
 // - suficiente soporte del estado actual
 // - evita estados casi completamente uniformes
 let baseSafe=q.risk<.085&&
          q.pt<.10&&
          q.support>=.45&&
          q.renyiNorm<.992&&
          q.horizonDisagreement<.035&&
          q.regimeShift<.045;

 // CAPA TÉCNICA ADITIVA:
 // No decide el dígito. El dígito sigue saliendo de Markov + Rényi.
 // Solo confirma si el contexto del precio tiene estructura suficiente.
 let tech=technicalAnalysis(),
     techSafe=!!tech&&tech.gate,
     regime=regimeAnalysis(),
     finalScore=qualityScore(q,tech,regime),
     regimeSafe=regime.gate,
     safe=baseSafe&&techSafe&&regimeSafe&&finalScore>=65;

 // V4: la señal cruda todavía NO se muestra. Debe sobrevivir 2 ticks reales.
 return{q,spread,H:H2,near,safe,baseSafe,techSafe,regimeSafe,tech,regime,finalScore};
}
function showSignal(raw,ready,waitReason=''){
 const buy=$('buy');
 lastSignal=ready||null;

 if(!raw){
  $('decision').textContent='ANALIZANDO';
  $('reason').textContent=waitReason||'Buscando confirmación estadística y técnica.';
  $('sepPick').textContent='—';
  $('risk').textContent='—';$('spread').textContent='—';$('entropy').textContent='—';
  $('phase').textContent='OBSERVAR';$('meter').style.width='0%';
  buy.textContent='ESPERANDO SEÑAL';buy.disabled=true;
  if($('techScore'))$('techScore').textContent='—';
  if($('techRsi'))$('techRsi').textContent='—';
  if($('techMacd'))$('techMacd').textContent='—';
  if($('techTrend'))$('techTrend').textContent='—';
  if($('finalScore'))$('finalScore').textContent='—';
  if($('delayState'))$('delayState').textContent='—';
  if($('regimeState'))$('regimeState').textContent='—';
  updateShadowUI();
  return;
 }

 let t=raw.tech;
 $('risk').textContent=(raw.q.risk*100).toFixed(1)+'%';
 $('spread').textContent=(raw.spread*100).toFixed(1);
 $('entropy').textContent=raw.H.toFixed(2);
 $('phase').textContent=raw.near?'MODO META':'ANÁLISIS';
 if(t){
  if($('techScore'))$('techScore').textContent=t.score.toFixed(0)+'/100';
  if($('techRsi'))$('techRsi').textContent=t.R.toFixed(1);
  if($('techMacd'))$('techMacd').textContent=t.M.hist.toFixed(4);
  if($('techTrend'))$('techTrend').textContent=t.trend;
 }
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
  $('decision').textContent='ANALIZANDO';
  $('reason').textContent=waitReason||'Todavía no hay una señal completa.';
  buy.textContent='ESPERANDO SEÑAL';buy.disabled=true;
  $('meter').style.width=Math.min(95,raw.finalScore)+'%';
  return;
 }

 $('sepPick').textContent='D'+ready.q.d;
 $('decision').textContent='SEÑAL LISTA · D'+ready.q.d;
 $('reason').textContent='Confirmada tras 2 ticks reales · score '+ready.finalScore.toFixed(0)+'/100.';
 buy.textContent='COMPRAR AHORA · D'+ready.q.d+' · RIESGO '+(ready.q.risk*100).toFixed(1)+'%';
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
  cooldownTicks=Math.max(cooldownTicks,3);
  log('MATCH '+label+' $'+profit.toFixed(2)+' · recalibración 3T');
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

 // Descuenta protecciones que ya venían activas antes de este tick.
 if(cooldownTicks>0)cooldownTicks--;
 if(deteriorationTicks>0)deteriorationTicks--;

 // Evalúa resultados shadow cuyo vencimiento corresponde a este tick.
 settleShadow(d);

 // Liquida simulación antes de recalcular la nueva señal.
 if(pending&&pending.mode==='SIM'){
  let p=pending;
  finish(d===p.d?-p.stake:p.stake*.10,'SIM');
 }

 hist.push(d);
 if(hist.length>1000)hist.shift();
 if(Number.isFinite(price)){
  quotes.push(price);
  if(quotes.length>1000)quotes.shift();
 }
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
 }else if(cooldownTicks>0){
  arming=null;
  reason='Recalibrando después de MATCH · '+cooldownTicks+' tick'+(cooldownTicks===1?'':'s')+'.';
 }else if(deteriorationTicks>0){
  arming=null;
  reason='Protección activa por deterioro reciente · '+deteriorationTicks+'T.';
 }else if(!raw||!raw.safe){
  arming=null;
  if(!raw)reason='Recolectando datos.';
  else if(!raw.baseSafe)reason='Markov/Rényi aún no confirma.';
  else if(!raw.techSafe)reason='Análisis técnico aún no confirma.';
  else if(!raw.regimeSafe)reason='Cambio de régimen detectado.';
  else reason='Score final insuficiente: '+raw.finalScore.toFixed(0)+'/100.';
 }else{
  if(!arming||arming.d!==raw.q.d){
   arming={d:raw.q.d,age:0,minScore:raw.finalScore,confirmedShadow:false};
   queueShadow(raw.q.d,'raw',tickCounter+3);
   reason='Preseñal detectada · validando 2 ticks reales.';
  }else{
   arming.age++;
   arming.minScore=Math.min(arming.minScore,raw.finalScore);

   if(arming.age<2){
    reason='Validación real '+arming.age+'/2 ticks.';
   }else if(arming.minScore>=65){
    ready={...raw,finalScore:Math.min(raw.finalScore,arming.minScore)};
    reason='Confirmación completa.';
    if(!arming.confirmedShadow){
     queueShadow(raw.q.d,'confirmed',tickCounter+1);
     arming.confirmedShadow=true;
    }
   }else{
    arming=null;
    reason='La calidad cayó durante el delay.';
   }
  }
 }

 if($('delayState')){
  $('delayState').textContent=ready?'2/2 LISTO':arming?(Math.min(2,arming.age)+'/2'):'—';
 }
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
stake=baseStake();$('status').textContent='ANÁLISIS ACTIVO · V4';$('buy').disabled=true;ui();updateShadowUI();connect();
