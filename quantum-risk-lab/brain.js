const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null;
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const DISPLAY_MAX_RISK=.035;
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}
function analyse(){
 if(hist.length<220)return null;

 // QUANTUM ANALYSIS V7 · DELAY 1 TICK
 // La señal se calcula ahora, pero la entrada efectiva ocurre 1 tick después.
 // Por eso el objetivo matemático principal es minimizar la probabilidad
 // de que el candidato aparezca en t+2 respecto al momento de la señal.
 // Sin GAP, EMA/EWMA ni indicadores de mercado.

 let n=hist.length,
     last=hist[n-1],
     prev=hist[n-2],
     w10=hist.slice(-10),
     w30=hist.slice(-30),
     w100=hist.slice(-100),
     f10=Array(10).fill(0),
     f30=Array(10).fill(0),
     f100=Array(10).fill(0),
     t1=Array.from({length:10},()=>Array(10).fill(0)),
     rowN=Array(10).fill(0),
     next2Last=Array(10).fill(0),
     next2Pair=Array(10).fill(0),
     cLast2=0,
     cPair2=0;

 w10.forEach(x=>f10[x]++);
 w30.forEach(x=>f30[x]++);
 w100.forEach(x=>f100[x]++);

 // Matriz de transición de 1 paso P(a->b).
 for(let i=Math.max(0,n-900);i<n-1;i++){
  let a=hist[i],b=hist[i+1];
  t1[a][b]++;
  rowN[a]++;
 }

 // Frecuencia empírica del dígito en t+2 dado el último dígito actual.
 for(let i=Math.max(0,n-900);i<n-2;i++){
  if(hist[i]===last){
   next2Last[hist[i+2]]++;
   cLast2++;
  }
 }

 // Frecuencia empírica del dígito en t+2 dado el par [prev,last].
 for(let i=Math.max(1,n-900);i<n-2;i++){
  if(hist[i-1]===prev&&hist[i]===last){
   next2Pair[hist[i+2]]++;
   cPair2++;
  }
 }

 let H=ent(hist.slice(-36)),rows=[];

 function laplace(count,total,alpha){
  return (count+alpha)/(total+10*alpha);
 }

 function p1(a,b){
  return laplace(t1[a][b],rowN[a],1);
 }

 for(let d=0;d<10;d++){
  let p10=f10[d]/10,
      p30=f30[d]/30,
      p100=f100[d]/100;

  // Método A: probabilidad a 2 pasos mediante P².
  // P(X[t+2]=d | X[t]=last) = Σ_j P(last->j)P(j->d)
  let p2Matrix=0;
  for(let j=0;j<10;j++) p2Matrix+=p1(last,j)*p1(j,d);

  // Método B: observación empírica directa a t+2.
  let p2Last=laplace(next2Last[d],cLast2,1),
      p2Pair=laplace(next2Pair[d],cPair2,1);

  // El contexto de 2 dígitos pesa más solo si tiene suficiente soporte.
  let pairSupport=cPair2/(cPair2+12),
      p2Emp=(1-.40*pairSupport)*p2Last+(.40*pairSupport)*p2Pair;

  // Consenso de dos métodos independientes para el tick objetivo.
  let targetP=.56*p2Emp+.44*p2Matrix;

  // Frecuencia base multi-muestra.
  let freq=.50*p10+.30*p30+.20*p100;

  // Penalizaciones matemáticas puras por sobre-representación.
  let excess10=Math.max(0,p10-.10),
      excess30=Math.max(0,p30-.10),
      excess100=Math.max(0,p100-.10),
      excessTarget=Math.max(0,targetP-.10);

  // Estabilidad entre las tres ventanas de frecuencia.
  let mean=(p10+p30+p100)/3,
      variance=((p10-mean)**2+(p30-mean)**2+(p100-mean)**2)/3,
      instability=Math.sqrt(variance);

  // Desacuerdo entre estimación empírica t+2 y P².
  let modelDisagreement=Math.abs(p2Emp-p2Matrix);

  // Chi-cuadrado local, solo si el dígito está sobre-representado.
  let expected30=3,
      chiOver=f30[d]>expected30
       ? ((f30[d]-expected30)**2)/expected30
       : 0;

  // Riesgo interno V7:
  // mayor peso al tick t+2 para compensar el delay real de 1 tick.
  let risk=.30*targetP+
           .10*freq+
           .14*excessTarget+
           .10*excess10+
           .07*excess30+
           .03*excess100+
           .08*instability+
           .08*modelDisagreement+
           .0015*chiOver;

  rows.push({
   d,risk,
   p12:p10,
   p36:p30,
   p120:p100,
   pt:targetP,
   pe:freq,
   p2Emp,
   p2Matrix
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

 // Mantiene la misma salida para no tocar ninguna otra parte del programa.
 // El filtro interno ahora interpreta q.pt como riesgo matemático del tick t+2.
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