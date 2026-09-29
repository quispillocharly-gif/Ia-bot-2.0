const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null;
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
const DISPLAY_SEP=.03;
const DISPLAY_MAX_RISK=.035;
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}
function analyse(){
 if(hist.length<180)return null;

 // QUANTUM ANALYSIS V4 · ANTI-APARICIÓN EN LOS PRÓXIMOS 3 TICKS
 // Objetivo: si el candidato es D4, favorecer contextos históricos donde,
 // después de un contexto parecido al actual, D4 NO apareció en t+1, t+2 ni t+3.
 let last=hist[hist.length-1],prev=hist[hist.length-2],
     w12=hist.slice(-12),w36=hist.slice(-36),w120=hist.slice(-120),
     f12=Array(10).fill(0),f36=Array(10).fill(0),f120=Array(10).fill(0),
     r18=Array(10).fill(0),p18=Array(10).fill(0),
     tr1=Array(10).fill(0),tr2=Array(10).fill(0);

 w12.forEach(x=>f12[x]++);
 w36.forEach(x=>f36[x]++);
 w120.forEach(x=>f120[x]++);
 w36.slice(-18).forEach(x=>r18[x]++);
 w36.slice(0,18).forEach(x=>p18[x]++);

 for(let i=Math.max(1,hist.length-420);i<hist.length;i++){
  if(hist[i-1]===last)tr1[hist[i]]++;
 }
 for(let i=Math.max(2,hist.length-700);i<hist.length;i++){
  if(hist[i-2]===prev&&hist[i-1]===last)tr2[hist[i]]++;
 }

 let t1=tr1.reduce((a,b)=>a+b,0),
     t2=tr2.reduce((a,b)=>a+b,0),
     H=ent(w36),
     entropyNorm=H/Math.log2(10),
     rows=[];

 function shrink(count,n,k){
  return (count+k*.10)/(n+k);
 }

 function ewmaProb(d,lookback,decay){
  let start=Math.max(0,hist.length-lookback),num=0,den=0,w=1;
  for(let i=hist.length-1;i>=start;i--){
   if(hist[i]===d)num+=w;
   den+=w;
   w*=decay;
  }
  return den?num/den:.10;
 }

 // Estima el riesgo de que el candidato aparezca AL MENOS UNA VEZ
 // dentro de los próximos 3 ticks desde un contexto parecido al actual.
 function future3Risk(d){
  const baseline=1-Math.pow(.9,3); // 27.1% si los dígitos fueran uniformes e independientes.
  let n1=0,h1=0,n2=0,h2=0;
  let begin=Math.max(1,hist.length-850);

  for(let i=begin;i<=hist.length-4;i++){
   let hit=hist[i+1]===d||hist[i+2]===d||hist[i+3]===d;

   if(hist[i]===last){
    n1++;
    if(hit)h1++;
   }

   if(i>=1&&hist[i-1]===prev&&hist[i]===last){
    n2++;
    if(hit)h2++;
   }
  }

  // Suavizado hacia el 27.1% base para evitar extremos por muestras pequeñas.
  let p1=(h1+10*baseline)/(n1+10);
  let p2=(h2+7*baseline)/(n2+7);

  // El contexto de 2 dígitos pesa más solo cuando tiene suficientes observaciones.
  let rel2=n2/(n2+12);
  let w2=Math.min(.42,.42*rel2);
  let p=(1-w2)*p1+w2*p2;

  return{p,n1,n2};
 }

 for(let d=0;d<10;d++){
  let gap=0;
  for(let i=hist.length-1;i>=0&&gap<60;i--){if(hist[i]===d)break;gap++}

  let streak=0;
  for(let i=hist.length-1;i>=0&&hist[i]===d;i--)streak++;

  let p12=shrink(f12[d],12,.7),
      p36=shrink(f36[d],36,1.2),
      p120=shrink(f120[d],120,2.0),
      peFast=ewmaProb(d,48,.91),
      peSlow=ewmaProb(d,120,.972),
      pe=.64*peFast+.36*peSlow;

  let raw1=t1?tr1[d]/t1:.10,
      rel1=t1/(t1+14),
      pt1=rel1*raw1+(1-rel1)*.10;

  let raw2=t2?tr2[d]/t2:.10,
      rel2=t2/(t2+10),
      pt2=rel2*raw2+(1-rel2)*.10;

  let pairWeight=Math.min(.32,.32*rel2),
      pt=(1-pairWeight)*pt1+pairWeight*pt2;

  let recent18=r18[d]/18,
      prior18=p18[d]/18,
      accel18=Math.max(0,recent18-prior18);

  let short=.80*p12+.20*p36,
      mid=.88*p36+.12*p120;

  let rise=Math.max(0,p12-p36)*.24+
           Math.max(0,p36-p120)*.08+
           accel18*.16+
           Math.max(0,peFast-peSlow)*.20;

  let hot=Math.max(0,p12-.10)*.72+
          Math.max(0,p36-.10)*.28+
          Math.max(0,pe-.10)*.34;

  let patternTrust=Math.max(.45,Math.min(1,1.65-entropyNorm));
  let transition=Math.max(0,pt-.10)*.58*patternTrust;

  let recency=gap===0?.060:
              gap===1?.034:
              gap===2?.017:
              gap===3?.008:
              gap===4?.003:0;

  let repeat=Math.min(streak,3)*.028;

  let hi=Math.max(short,mid,pe,pt),
      lo=Math.min(short,mid,pe,pt),
      disagreement=Math.max(0,(hi-lo)-.055)*.15;

  let h3=future3Risk(d);

  // Anti-3: 27.1% es el nivel aleatorio aproximado. Queremos candidatos
  // claramente por debajo de ese nivel. A partir de 20% la penalización
  // crece de forma no lineal para castigar apariciones tempranas.
  let anti3Excess=Math.max(0,h3.p-.20),
      anti3=anti3Excess*.18+anti3Excess*anti3Excess*1.25;

  let risk=.28*short+
           .17*mid+
           .11*p120+
           .15*pt+
           .21*pe+
           hot+transition+rise+recency+repeat+disagreement+anti3;

  rows.push({d,risk,p12,p36,p120,pt,pe,peFast,peSlow,gap,h3:h3.p,h3n1:h3.n1,h3n2:h3.n2});
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

 // El filtro interno ahora también exige que la estimación de aparición
 // dentro de 3 ticks sea menor que 25%. No garantiza el resultado:
 // es un filtro estadístico adicional.
 let safe=q.risk<maxRisk&&
          spread>=need&&
          q.p12<=.10&&
          q.p36<=.12&&
          q.pt<=.13&&
          q.pe<=.12&&
          q.h3<.25;

 return{q,spread,H,near,safe};
}
function showSignal(s){
 lastSignal=s;
 if(!s){$('decision').textContent='OBSERVANDO';$('reason').textContent='Aún no existe suficiente historial.';$('buy').textContent='COMPRAR AHORA · CALIBRANDO';$('sepPick').textContent='—';return}
 let sep=s.spread*100,riskPct=s.q.risk*100,visible=s.spread>=DISPLAY_SEP&&s.q.risk<=DISPLAY_MAX_RISK;
 $('risk').textContent=riskPct.toFixed(1)+'%';
 $('sepPick').textContent=visible?'D'+s.q.d:'—';
 $('buy').textContent=visible?'COMPRAR AHORA · D'+s.q.d+' · RIESGO '+riskPct.toFixed(1)+'%':'ESPERANDO · SEP ≥ 3.0 Y RIESGO ≤ 3.5%';
 $('spread').textContent=sep.toFixed(1);$('entropy').textContent=s.H.toFixed(2);$('phase').textContent=s.near?'MODO META':'ANÁLISIS';$('meter').style.width=Math.min(100,s.spread*1000)+'%';
 if(!visible){$('decision').textContent='NO MOSTRAR DÍGITO';$('reason').textContent='Requiere separación ≥ 3.0 y riesgo interno de 3.5% o menor. Actual: sep '+sep.toFixed(1)+' · riesgo '+riskPct.toFixed(1)+'%.'}
 else if(!s.safe){$('decision').textContent='CANDIDATO D'+s.q.d;$('reason').textContent='Cumple separación visual ≥ 3.0, pero aún no supera el filtro de seguridad para operar.'}
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
$('buy').onclick=()=>{if(!running){$('status').textContent='PULSA REINICIAR SESIÓN';return}if(pending){$('status').textContent='OPERACIÓN EN CURSO';return}let s=analyse();if(!s){$('status').textContent='AÚN CALIBRANDO';return}if(s.spread<DISPLAY_SEP||s.q.risk>DISPLAY_MAX_RISK){$('status').textContent='ESPERANDO SEP ≥ 3.0 Y RIESGO ≤ 3.5%';return}enter(s)};
window.demoSettlement=p=>finish(p,'DERIV DEMO');
window.demoTradeError=tradeError;
stake=baseStake();$('status').textContent='ANÁLISIS ACTIVO';ui();connect();