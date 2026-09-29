const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null;
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const DISPLAY_MAX_RISK=.035;
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}
function analyse(){
 if(hist.length<240)return null;

 // QUANTUM ANALYSIS V8 · MATEMÁTICA BAYESIANA PARA DELAY 1 TICK
 // Sin GAP, EMA/EWMA ni indicadores técnicos.
 // Objetivo: estimar el riesgo del dígito en t+2, porque existe 1 tick de delay.
 // Combina:
 // 1) P² de una matriz de transición,
 // 2) conteo directo a t+2 según el último dígito,
 // 3) conteo directo a t+2 según los últimos 2 y 3 dígitos,
 // 4) pesos por tamaño de muestra,
 // 5) penalización por incertidumbre y desacuerdo entre estimadores.

 let n=hist.length,
     last=hist[n-1],
     prev=hist[n-2],
     prev2=hist[n-3],
     w12=hist.slice(-12),
     w36=hist.slice(-36),
     w120=hist.slice(-120),
     f12=Array(10).fill(0),
     f36=Array(10).fill(0),
     f120=Array(10).fill(0),
     t1=Array.from({length:10},()=>Array(10).fill(0)),
     rowN=Array(10).fill(0),
     next2Last=Array(10).fill(0),
     next2Pair=Array(10).fill(0),
     next2Triple=Array(10).fill(0),
     cLast=0,cPair=0,cTriple=0;

 w12.forEach(x=>f12[x]++);
 w36.forEach(x=>f36[x]++);
 w120.forEach(x=>f120[x]++);

 // Matriz de transición de un paso.
 for(let i=Math.max(0,n-1000);i<n-1;i++){
  let a=hist[i],b=hist[i+1];
  t1[a][b]++;
  rowN[a]++;
 }

 // Observación directa del resultado a t+2.
 for(let i=Math.max(0,n-1000);i<n-2;i++){
  if(hist[i]===last){
   next2Last[hist[i+2]]++;
   cLast++;
  }

  if(i>=1&&hist[i-1]===prev&&hist[i]===last){
   next2Pair[hist[i+2]]++;
   cPair++;
  }

  if(i>=2&&hist[i-2]===prev2&&hist[i-1]===prev&&hist[i]===last){
   next2Triple[hist[i+2]]++;
   cTriple++;
  }
 }

 let H=ent(hist.slice(-36)),rows=[];

 function dirichletMean(count,total,alpha=1){
  return (count+alpha)/(total+10*alpha);
 }

 // Varianza posterior aproximada para una categoría de un Dirichlet simétrico.
 function dirichletVar(count,total,alpha=1){
  let A=total+10*alpha,
      a=count+alpha;
  return (a*(A-a))/(A*A*(A+1));
 }

 function p1(a,b){
  return dirichletMean(t1[a][b],rowN[a],1);
 }

 function reliability(total,k){
  return total/(total+k);
 }

 for(let d=0;d<10;d++){
  // Frecuencias suavizadas en tres escalas.
  let p12=dirichletMean(f12[d],12,.8),
      p36=dirichletMean(f36[d],36,1),
      p120=dirichletMean(f120[d],120,1.5);

  // Estimador A: transición de dos pasos P².
  let p2Matrix=0;
  for(let j=0;j<10;j++) p2Matrix+=p1(last,j)*p1(j,d);

  // Estimadores B/C/D: conteo directo a t+2.
  let pLast=dirichletMean(next2Last[d],cLast,1),
      pPair=dirichletMean(next2Pair[d],cPair,1),
      pTriple=dirichletMean(next2Triple[d],cTriple,1);

  // Cada contexto recibe peso según la cantidad de evidencia.
  let rLast=reliability(cLast,16),
      rPair=reliability(cPair,14),
      rTriple=reliability(cTriple,10),
      rMatrix=.72;

  // El contexto más largo no puede dominar si la muestra es pequeña.
  let wLast=.34*rLast,
      wPair=.26*rPair,
      wTriple=.18*rTriple,
      wMatrix=.22*rMatrix,
      wSum=wLast+wPair+wTriple+wMatrix;

  let targetP=(wLast*pLast+wPair*pPair+wTriple*pTriple+wMatrix*p2Matrix)/wSum;

  // Incertidumbre posterior de los tres conteos directos.
  let vLast=dirichletVar(next2Last[d],cLast,1),
      vPair=dirichletVar(next2Pair[d],cPair,1),
      vTriple=dirichletVar(next2Triple[d],cTriple,1);

  let uncertainty=Math.sqrt(
   (wLast*vLast+wPair*vPair+wTriple*vTriple)/Math.max(.0001,wLast+wPair+wTriple)
  );

  // Frecuencia general suavizada.
  let freq=.50*p12+.30*p36+.20*p120;

  // Desacuerdo entre estimadores: si no coinciden, se penaliza.
  let est=[pLast,pPair,pTriple,p2Matrix],
      estMean=est.reduce((a,b)=>a+b,0)/est.length,
      disagreement=Math.sqrt(
       est.reduce((s,x)=>s+(x-estMean)*(x-estMean),0)/est.length
      );

  // Inestabilidad entre ventanas de frecuencia.
  let fMean=(p12+p36+p120)/3,
      instability=Math.sqrt(
       ((p12-fMean)**2+(p36-fMean)**2+(p120-fMean)**2)/3
      );

  // Penalizaciones solo por exceso sobre el 10% teórico.
  let excessTarget=Math.max(0,targetP-.10),
      excess12=Math.max(0,p12-.10),
      excess36=Math.max(0,p36-.10),
      excess120=Math.max(0,p120-.10);

  // Chi-cuadrado local sobre 36 ticks.
  let expected36=3.6,
      chiOver=f36[d]>expected36
       ? ((f36[d]-expected36)**2)/expected36
       : 0;

  // Riesgo interno V8.
  // Mantiene una escala compatible con el filtro visual actual (<= 3.5%).
  let risk=.19*targetP+
           .05*freq+
           .10*excessTarget+
           .07*excess12+
           .045*excess36+
           .02*excess120+
           .025*uncertainty+
           .05*disagreement+
           .05*instability+
           .0012*chiOver;

  rows.push({
   d,risk,
   p12,
   p36,
   p120,
   pt:targetP,
   pe:freq,
   p2Matrix,
   pLast,
   pPair,
   pTriple,
   uncertainty
  });
 }

 rows.sort((a,b)=>a.risk-b.risk);

 let pool=rows.filter(x=>x.d!==lastPick),
     q=pool[0],
     second=pool[1];

 if(!q||!second)return null;

 let spread=second.risk-q.risk,
     near=pnl>=target()*.75,
     need=near?.05:.03,
     maxRisk=near?.085:.105;

 // Mantiene el mismo contrato para no tocar la interfaz ni el resto del programa.
 let safe=q.risk<maxRisk&&
          spread>=need&&
          q.p12<=.10&&
          q.p36<=.12&&
          q.pt<=.13&&
          q.pe<=.12;

 return{q,spread,H,near,safe};
}
function showSignal(s){
 lastSignal=s;
 if(!s){$('decision').textContent='OBSERVANDO';$('reason').textContent='Aún no existe suficiente historial.';$('buy').textContent='COMPRAR AHORA · CALIBRANDO';$('sepPick').textContent='—';return}
 let sep=s.spread*100,riskPct=s.q.risk*100,visible=s.q.risk<=DISPLAY_MAX_RISK;
 $('risk').textContent=riskPct.toFixed(1)+'%';
 $('sepPick').textContent=visible?'D'+s.q.d:'—';
 $('buy').textContent=visible?'COMPRAR AHORA · D'+s.q.d+' · RIESGO '+riskPct.toFixed(1)+'%':'ESPERANDO · RIESGO ≤ 3.5%';
 $('spread').textContent=sep.toFixed(1);$('entropy').textContent=s.H.toFixed(2);$('phase').textContent=s.near?'MODO META':'ANÁLISIS';$('meter').style.width=Math.min(100,s.spread*1000)+'%';
 if(!visible){$('decision').textContent='NO MOSTRAR DÍGITO';$('reason').textContent='Requiere riesgo interno de 3.5% o menor. Actual: '+riskPct.toFixed(1)+'%.'}
 else if(!s.safe){$('decision').textContent='CANDIDATO D'+s.q.d;$('reason').textContent='Cumple el filtro visual de riesgo, pero aún no supera el filtro interno de seguridad.'}
 else{$('decision').textContent='SEÑAL D'+s.q.d;$('reason').textContent='Candidato de menor riesgo interno; entrada habilitada.'}
}
function ui(d){
 if(d!==undefined)$('tick').textContent='D'+d;
 $('pnl').textContent=(pnl>=0?'+':'')+'$'+pnl.toFixed(2);$('stake').textContent='$'+stake.toFixed(2);$('wins').textContent=wins;$('losses').textContent=losses;$('ops').textContent=ops;$('pick').textContent=lastPick===null?'—':'D'+lastPick;
}
function enter(s){
 if(!running||pending||!s)return;
 let d=s.q.d,mode=$('mode').value;
 if(mode==='DEMO'&&!window.demoReady){$('status').textContent='CONECTA DEMO DERIV';return}
 lastPick=d;observe=0;pending={d,stake,mode};ops++;
 $('decision').textContent='COMPRA '+mode+' · DIFFER D'+d;$('reason').textContent='$'+stake.toFixed(2)+' · duración 1 tick';log('COMPRA '+mode+' D'+d+' $'+stake.toFixed(2));ui();
 if(mode==='DEMO'){ $('status').textContent='ENVIANDO A DERIV…'; window.sendDemoTrade(d,stake).catch(e=>tradeError(e));}
 else $('status').textContent='SIM ABIERTA';
}
function tradeError(e){
 log('ERROR DEMO '+(e?.message||e));pending=null;observe=0;$('status').textContent='ERROR DEMO · REVISA LOG';ui();
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
 hist.push(d);if(hist.length>1000)hist.shift();if(running&&!pending)observe++;ui(d);showSignal(analyse());
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
 if($('mode').value==='DEMO'&&!window.demoReady){$('status').textContent='CONECTA DEMO DERIV PRIMERO';return}
 pnl=0;stake=baseStake();wins=0;losses=0;ops=0;pending=null;observe=0;lastPick=null;running=true;$('status').textContent='ANALIZANDO';log('NUEVA SESIÓN '+$('mode').value+' · STAKE $'+stake.toFixed(2)+' · META $'+target().toFixed(2));ui();
};
$('stop').onclick=()=>{running=false;$('status').textContent='STOP MANUAL'};
$('buy').onclick=()=>{if(!running){$('status').textContent='PULSA REINICIAR SESIÓN';return}if(pending){$('status').textContent='OPERACIÓN EN CURSO';return}let s=analyse();if(!s){$('status').textContent='AÚN CALIBRANDO';return}if(s.q.risk>DISPLAY_MAX_RISK){$('status').textContent='ESPERANDO RIESGO ≤ 3.5%';return}enter(s)};
window.demoSettlement=p=>finish(p,'DERIV DEMO');
window.demoTradeError=tradeError;
stake=baseStake();$('status').textContent='ANÁLISIS ACTIVO';ui();connect();