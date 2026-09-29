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
 if(hist.length<220)return null;

 // QUANTUM ANALYSIS V5 · GAP HAZARD
 // Lógica distinta a la anterior:
 // - mide el tiempo entre apariciones de cada dígito;
 // - estima el "hazard" de que vuelva justo en el siguiente tick;
 // - compara contextos históricos de 1, 2 y 3 dígitos;
 // - estabiliza con frecuencia reciente y castigo por ráfagas.
 let n=hist.length,
     last=hist[n-1],
     prev=hist[n-2],
     prev2=hist[n-3],
     w12=hist.slice(-12),
     w36=hist.slice(-36),
     w96=hist.slice(-96),
     f12=Array(10).fill(0),
     f36=Array(10).fill(0),
     f96=Array(10).fill(0),
     next1=Array(10).fill(0),
     next2=Array(10).fill(0),
     next3=Array(10).fill(0),
     c1=0,c2=0,c3=0;

 w12.forEach(x=>f12[x]++);
 w36.forEach(x=>f36[x]++);
 w96.forEach(x=>f96[x]++);

 // Contexto exacto de 1, 2 y 3 dígitos.
 for(let i=Math.max(3,n-900);i<n-1;i++){
  if(hist[i]===last){
   next1[hist[i+1]]++; c1++;
  }
  if(hist[i-1]===prev&&hist[i]===last){
   next2[hist[i+1]]++; c2++;
  }
  if(hist[i-2]===prev2&&hist[i-1]===prev&&hist[i]===last){
   next3[hist[i+1]]++; c3++;
  }
 }

 let H=ent(w36),rows=[];

 function smooth(count,total,k,base=.10){
  return (count+k*base)/(total+k);
 }

 function currentGap(d){
  let g=0;
  for(let i=n-1;i>=0&&g<120;i--){
   if(hist[i]===d)break;
   g++;
  }
  return g;
 }

 // Riesgo de reaparición en el próximo tick condicionado al gap actual.
 // Se calcula con gaps históricos del mismo dígito y supervivencia del ciclo.
 function gapHazard(d,gapNow){
  let positions=[];
  for(let i=Math.max(0,n-950);i<n;i++) if(hist[i]===d) positions.push(i);
  if(positions.length<6)return .10;

  let gaps=[];
  for(let i=1;i<positions.length;i++) gaps.push(positions[i]-positions[i-1]-1);
  if(!gaps.length)return .10;

  let band=Math.min(3,Math.max(1,Math.floor(Math.sqrt(gapNow+1)/2))),
      survived=0,endedNext=0;

  for(let g of gaps){
   if(g>=Math.max(0,gapNow-band)){
    survived++;
    if(Math.abs(g-gapNow)<=band) endedNext++;
   }
  }

  // Suavizado fuerte: evita creer demasiado en pocos ciclos.
  return (endedNext+8*.10)/(survived+8);
 }

 // Detecta si el dígito está entrando en una ráfaga reciente.
 function burstScore(d){
  let a=hist.slice(-8).filter(x=>x===d).length/8;
  let b=hist.slice(-24).filter(x=>x===d).length/24;
  return Math.max(0,a-b);
 }

 for(let d=0;d<10;d++){
  let gap=currentGap(d),
      p12=f12[d]/12,
      p36=f36[d]/36,
      p96=f96[d]/96,
      pg=gapHazard(d,gap),

      p1=smooth(next1[d],c1,14),
      p2=smooth(next2[d],c2,10),
      p3=smooth(next3[d],c3,7);

  // Los contextos más largos pesan solo si tienen soporte suficiente.
  let r2=c2/(c2+12),
      r3=c3/(c3+8),
      w3=.34*r3,
      w2=.30*r2,
      w1=1-w2-w3,
      pc=w1*p1+w2*p2+w3*p3;

  // Tendencia de frecuencia: si el dígito se está calentando, sube riesgo.
  let trend=Math.max(0,p12-p36)*.42+
            Math.max(0,p36-p96)*.16;

  // Ráfaga y cercanía inmediata.
  let burst=burstScore(d)*.40,
      recent=gap===0?.060:
             gap===1?.035:
             gap===2?.018:
             gap===3?.008:0;

  // Consenso entre modelos: si contexto + hazard + frecuencia coinciden
  // en que el dígito está bajo, el score baja; si discrepan, se penaliza.
  let recentFreq=.62*p12+.38*p36,
      mean=(pc+pg+recentFreq)/3,
      disagreement=(
       Math.abs(pc-mean)+Math.abs(pg-mean)+Math.abs(recentFreq-mean)
      )*.14;

  // Núcleo del riesgo interno V5.
  let risk=.34*pc+
           .30*pg+
           .18*recentFreq+
           .10*p96+
           trend+burst+recent+disagreement;

  rows.push({
   d,risk,
   p12,p36,p120:p96,
   pt:pc,
   pe:pg,
   gap,
   ctx1:p1,ctx2:p2,ctx3:p3
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

 // Mismo contrato de salida para no tocar nada más de la página.
 let safe=q.risk<maxRisk&&
          spread>=need&&
          q.p12<=.10&&
          q.p36<=.12&&
          q.pt<=.13&&
          q.pe<=.13;

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