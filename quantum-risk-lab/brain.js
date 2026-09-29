const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null;
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const DISPLAY_SEP=.02;
const DISPLAY_MAX_RISK=.035;
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}
function analyse(){
 if(hist.length<180)return null;

 // QUANTUM ANALYSIS V6 · MATEMÁTICA DISCRETA PURA
 // Sin GAP, EMA/EWMA ni indicadores de mercado.
 // Solo usa: conteos, frecuencias, probabilidades condicionadas,
 // desviación frente al 10% teórico y consistencia entre muestras.

 let n=hist.length,
     last=hist[n-1],
     prev=hist[n-2],
     w10=hist.slice(-10),
     w30=hist.slice(-30),
     w100=hist.slice(-100),
     f10=Array(10).fill(0),
     f30=Array(10).fill(0),
     f100=Array(10).fill(0),
     tr1=Array(10).fill(0),
     tr2=Array(10).fill(0);

 w10.forEach(x=>f10[x]++);
 w30.forEach(x=>f30[x]++);
 w100.forEach(x=>f100[x]++);

 // Conteo condicional de orden 1 y 2.
 let c1=0,c2=0;
 for(let i=Math.max(1,n-600);i<n-1;i++){
  if(hist[i]===last){
   tr1[hist[i+1]]++;
   c1++;
  }
  if(i>=1&&hist[i-1]===prev&&hist[i]===last){
   tr2[hist[i+1]]++;
   c2++;
  }
 }

 let H=ent(hist.slice(-36)),rows=[];

 // Suavizado de Laplace puro para evitar divisiones extremas
 // cuando una combinación aparece pocas veces.
 function laplace(count,total,alpha){
  return (count+alpha)/(total+10*alpha);
 }

 for(let d=0;d<10;d++){
  let p10=f10[d]/10,
      p30=f30[d]/30,
      p100=f100[d]/100;

  let p1=laplace(tr1[d],c1,1),
      p2=laplace(tr2[d],c2,1);

  // El orden 2 recibe más peso solamente cuando hay más observaciones.
  let support2=c2/(c2+10),
      pc=(1-.35*support2)*p1+(.35*support2)*p2;

  // Frecuencia pura multi-muestra.
  let pf=.50*p10+.30*p30+.20*p100;

  // Penaliza únicamente exceso sobre el 10% teórico.
  let excess10=Math.max(0,p10-.10),
      excess30=Math.max(0,p30-.10),
      excess100=Math.max(0,p100-.10),
      excessCond=Math.max(0,pc-.10);

  // Desviación/consistencia: si las tres muestras discrepan mucho,
  // aumenta el score porque la estimación es menos estable.
  let mean=(p10+p30+p100)/3,
      variance=((p10-mean)**2+(p30-mean)**2+(p100-mean)**2)/3,
      instability=Math.sqrt(variance);

  // Chi cuadrado local: castiga dígitos sobrerrepresentados en 30 ticks.
  // Esperado = 3 apariciones por dígito.
  let expected30=3,
      chiOver=f30[d]>expected30
       ? ((f30[d]-expected30)**2)/expected30
       : 0;

  // Score interno: cuanto menor, mejor candidato.
  // Se mantiene en una escala compatible con los filtros actuales de la UI.
  let risk=.18*pf+
           .12*pc+
           .17*excess10+
           .10*excess30+
           .05*excess100+
           .12*excessCond+
           .10*instability+
           .0018*chiOver;

  rows.push({
   d,risk,
   p12:p10,
   p36:p30,
   p120:p100,
   pt:pc,
   pe:pf
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

 // Conserva el mismo formato y filtros internos esperados
 // por el resto del programa.
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
 let sep=s.spread*100,riskPct=s.q.risk*100,visible=s.spread>=DISPLAY_SEP&&s.q.risk<=DISPLAY_MAX_RISK;
 $('risk').textContent=riskPct.toFixed(1)+'%';
 $('sepPick').textContent=visible?'D'+s.q.d:'—';
 $('buy').textContent=visible?'COMPRAR AHORA · D'+s.q.d+' · RIESGO '+riskPct.toFixed(1)+'%':'ESPERANDO · SEP ≥ 2.0 Y RIESGO ≤ 3.5%';
 $('spread').textContent=sep.toFixed(1);$('entropy').textContent=s.H.toFixed(2);$('phase').textContent=s.near?'MODO META':'ANÁLISIS';$('meter').style.width=Math.min(100,s.spread*1000)+'%';
 if(!visible){$('decision').textContent='NO MOSTRAR DÍGITO';$('reason').textContent='Requiere separación ≥ 2.0 y riesgo interno de 3.5% o menor. Actual: sep '+sep.toFixed(1)+' · riesgo '+riskPct.toFixed(1)+'%.'}
 else if(!s.safe){$('decision').textContent='CANDIDATO D'+s.q.d;$('reason').textContent='Cumple separación visual ≥ 2.0, pero aún no supera el filtro de seguridad para operar.'}
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
$('buy').onclick=()=>{if(!running){$('status').textContent='PULSA REINICIAR SESIÓN';return}if(pending){$('status').textContent='OPERACIÓN EN CURSO';return}let s=analyse();if(!s){$('status').textContent='AÚN CALIBRANDO';return}if(s.spread<DISPLAY_SEP||s.q.risk>DISPLAY_MAX_RISK){$('status').textContent='ESPERANDO SEP ≥ 2.0 Y RIESGO ≤ 3.5%';return}enter(s)};
window.demoSettlement=p=>finish(p,'DERIV DEMO');
window.demoTradeError=tradeError;
stake=baseStake();$('status').textContent='ANÁLISIS ACTIVO';ui();connect();