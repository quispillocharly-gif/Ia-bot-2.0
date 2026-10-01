const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null,candidateHistory=[];
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}

function analyse(){
 if(hist.length<240)return null;

 // QUANTUM ANALYSIS V13 · V9 + REFUERZOS ADITIVOS
 // CADENA DE MARKOV DE PRIMER ORDEN + ENTROPÍA RÉNYI (alpha = 2)
 // Sin GAP, EMA/EWMA ni indicadores técnicos.
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
 let safe=q.risk<.085&&
          q.pt<.10&&
          q.support>=.45&&
          q.renyiNorm<.992&&
          q.horizonDisagreement<.035&&
          q.regimeShift<.045;

 // La UI ya tiene un campo ENTROPÍA; ahora muestra H2 de Rényi.
 return{q,spread,H:H2,near,safe};
}
function showSignal(s){
 lastSignal=s;
 if(!s){
  $('decision').textContent='OBSERVANDO';
  $('reason').textContent='Aún no existe suficiente historial.';
  $('buy').textContent='COMPRAR AHORA · CALIBRANDO';
  if($('sepPick'))$('sepPick').textContent='—';
  return;
 }
 let sep=s.spread*100,riskPct=s.q.risk*100;
 $('risk').textContent=riskPct.toFixed(1)+'%';
 if($('sepPick'))$('sepPick').textContent='D'+s.q.d;
 $('buy').textContent='COMPRAR AHORA · D'+s.q.d+' · RIESGO '+riskPct.toFixed(1)+'%';
 $('spread').textContent=sep.toFixed(1);
 $('entropy').textContent=s.H.toFixed(2);
 $('phase').textContent=s.near?'MODO META':'ANÁLISIS';
 $('meter').style.width=Math.min(100,s.spread*1000)+'%';
 if(!s.safe){
  $('decision').textContent='CANDIDATO D'+s.q.d;
  $('reason').textContent='Candidato calculado con Markov + Rényi y validaciones multi-horizonte.';
 }else{
  $('decision').textContent='SEÑAL D'+s.q.d;
  $('reason').textContent='Candidato de menor riesgo interno; entrada habilitada.';
 }
}

function ui(d){
 if(d!==undefined)$('tick').textContent='D'+d;
 $('pnl').textContent=(pnl>=0?'+':'')+'$'+pnl.toFixed(2);$('stake').textContent='$'+stake.toFixed(2);$('wins').textContent=wins;$('losses').textContent=losses;$('ops').textContent=ops;$('pick').textContent=lastPick===null?'—':'D'+lastPick;
}

function enter(s){
 if(!running||pending||!s)return;
 let d=s.q.d,mode=$('mode').value;
 if((mode==='DEMO'||mode==='REAL')&&!window.demoReady){
  $('status').textContent='CONECTA DERIV PRIMERO';return;
 }
 if((mode==='DEMO'||mode==='REAL')&&window.currentAccountType!==mode.toLowerCase()){
  $('status').textContent='RECONECTA EN MODO '+mode;return;
 }
 lastPick=d;observe=0;pending={d,stake,mode};ops++;
 $('decision').textContent='COMPRA '+mode+' · DIFFER D'+d;$('reason').textContent='$'+stake.toFixed(2)+' · duración 1 tick';log('COMPRA '+mode+' D'+d+' $'+stake.toFixed(2));ui();
 if(mode==='DEMO'||mode==='REAL'){
  $('status').textContent='ENVIANDO A DERIV…';
  window.sendDemoTrade(d,stake).catch(e=>tradeError(e));
 } else {
  $('status').textContent='SIM ABIERTA';
 }
}

function tradeError(e){
 log('ERROR DERIV '+(e?.message||e));pending=null;observe=0;$('status').textContent='ERROR · REVISA LOG';ui();
}

function finish(profit,label){
 profit=Number(profit);if(!Number.isFinite(profit)){tradeError(new Error('Resultado inválido'));return}
 pnl+=profit;
 if(profit>0){wins++;stake=Math.max(baseStake(),stake+profit);log('WIN '+label+' +$'+profit.toFixed(2))}
 else{losses++;stake=baseStake();log('MATCH '+label+' $'+profit.toFixed(2))}
 pending=null;observe=0;
 if(pnl>=target()){running=false;$('status').textContent='META +$'+target().toFixed(2)+' · STOP';$('phase').textContent='META'}
 else if(running)$('status').textContent='OBSERVANDO';
 ui();
}

function tick(d){
 if(pending&&pending.mode==='SIM'){let p=pending;finish(d===p.d?-p.stake:p.stake*.10,'SIM')}
 hist.push(d);if(hist.length>1000)hist.shift();if(running&&!pending)observe++;ui(d);
 let s=analyse();
 if(s){
  candidateHistory.push(s.q.d);
  if(candidateHistory.length>40)candidateHistory.shift();
 }
 showSignal(s);
}

function connect(){
 clearTimeout(retry);ws=new WebSocket('wss://api.derivws.com/trading/v1/options/ws/public');
 ws.onopen=()=>ws.send(JSON.stringify({ticks_history:'R_75',count:300,end:'latest',style:'ticks'}));
 ws.onmessage=e=>{let m=JSON.parse(e.data);
  if(m.history&&m.history.prices){let p=Number(m.pip_size||4);hist=m.history.prices.map(x=>Number(Number(x).toFixed(p).slice(-1)));ws.send(JSON.stringify({ticks:'R_75',subscribe:1}));$('status').textContent='LISTO · '+hist.length+' TICKS'}
  if(m.tick){let ep=+m.tick.epoch;if(ep===lastEpoch)return;lastEpoch=ep;let p=Number(m.tick.pip_size||4),d=Number(Number(m.tick.quote).toFixed(p).slice(-1));tick(d)}
 };
 ws.onclose=()=>retry=setTimeout(connect,2500);
}

$('start').onclick=()=>{
 let mode=$('mode').value;
 if((mode==='DEMO'||mode==='REAL')&&!window.demoReady){
  $('status').textContent='CONECTA DERIV PRIMERO';return;
 }
 pnl=0;stake=baseStake();wins=0;losses=0;ops=0;pending=null;observe=0;lastPick=null;candidateHistory=[];running=true;
 $('status').textContent='ANALIZANDO';
 log('NUEVA SESIÓN '+mode+' · STAKE $'+stake.toFixed(2)+' · META $'+target().toFixed(2));
 ui();
};
$('stop').onclick=()=>{running=false;$('status').textContent='STOP MANUAL'};
$('buy').onclick=()=>{
 if(!running){$('status').textContent='PULSA REINICIAR SESIÓN';return}
 if(pending){$('status').textContent='OPERACIÓN EN CURSO';return}
 let s=lastSignal;if(!s){$('status').textContent='AÚN CALIBRANDO';return}
 enter(s);
};

window.demoSettlement=p=>finish(p,'DERIV '+($('mode').value||''));
window.demoTradeError=tradeError;
stake=baseStake();$('status').textContent='ANÁLISIS ACTIVO';ui();connect();