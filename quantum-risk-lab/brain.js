const $=x=>document.getElementById(x);

let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,
    hist=[],quotes=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null;

const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);

function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function mean(a){return a.length?a.reduce((s,x)=>s+x,0)/a.length:0}
function stdev(a){
 if(a.length<2)return 0;
 const m=mean(a);
 return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/a.length);
}
function emaSeries(a,p){
 if(!a.length)return [];
 const k=2/(p+1),out=[a[0]];
 for(let i=1;i<a.length;i++)out.push(a[i]*k+out[i-1]*(1-k));
 return out;
}
function rsi(a,p=14){
 if(a.length<=p)return 50;
 let gains=0,losses=0;
 for(let i=a.length-p;i<a.length;i++){
  const d=a[i]-a[i-1];
  if(d>0)gains+=d; else losses-=d;
 }
 if(losses===0)return gains===0?50:100;
 const rs=(gains/p)/(losses/p);
 return 100-(100/(1+rs));
}
function macd(a){
 const e12=emaSeries(a,12),e26=emaSeries(a,26),line=[];
 for(let i=0;i<a.length;i++)line.push((e12[i]||a[i])-(e26[i]||a[i]));
 const sig=emaSeries(line,9);
 const i=line.length-1;
 return {line:line[i]||0,signal:sig[i]||0,hist:(line[i]||0)-(sig[i]||0)};
}
function freqProb(arr,d,alpha=.75){
 let c=alpha;
 for(const x of arr)if(x===d)c++;
 return c/(arr.length+10*alpha);
}
function transProb(arr,from,d,alpha=.5){
 let row=10*alpha,c=alpha;
 for(let i=0;i<arr.length-1;i++){
  if(arr[i]!==from)continue;
  row++;
  if(arr[i+1]===d)c++;
 }
 return c/row;
}

function analyse(){
 if(hist.length<240||quotes.length<240)return null;

 const q=quotes.slice(-600);
 const h=hist.slice(-600);
 const lastPrice=q[q.length-1];
 const lastDigit=h[h.length-1];

 const e9=emaSeries(q,9),e21=emaSeries(q,21),e50=emaSeries(q,50);
 const ema9=e9[e9.length-1],ema21=e21[e21.length-1],ema50=e50[e50.length-1];
 const R=rsi(q,14);
 const M=macd(q);

 const w20=q.slice(-20),mid=mean(w20),sd20=stdev(w20);
 const upper=mid+2*sd20,lower=mid-2*sd20;
 const bbPos=sd20?((lastPrice-mid)/(2*sd20)):0;

 const returns=[];
 for(let i=Math.max(1,q.length-40);i<q.length;i++)returns.push(q[i]-q[i-1]);
 const vol=stdev(returns);
 const scale=Math.max(vol,Math.abs(lastPrice)*1e-7,1e-8);

 const emaSpread=(ema9-ema21)/scale;
 const emaLong=(ema21-ema50)/scale;
 const macdNorm=M.hist/scale;
 const momentum10=(lastPrice-q[q.length-11])/scale;

 let bull=0,bear=0;
 if(ema9>ema21)bull++; else if(ema9<ema21)bear++;
 if(ema21>ema50)bull++; else if(ema21<ema50)bear++;
 if(M.hist>0)bull++; else if(M.hist<0)bear++;
 if(R>52)bull++; else if(R<48)bear++;
 if(momentum10>0)bull++; else if(momentum10<0)bear++;

 let trend='LATERAL';
 if(bull>=4&&bull>bear)trend='ALCISTA';
 else if(bear>=4&&bear>bull)trend='BAJISTA';

 const alignment=Math.max(bull,bear)/5;
 const emaStrength=Math.min(1,(Math.abs(emaSpread)+Math.abs(emaLong))/4);
 const macdStrength=Math.min(1,Math.abs(macdNorm)/2);
 const rsiStrength=Math.min(1,Math.abs(R-50)/20);
 const momentumStrength=Math.min(1,Math.abs(momentum10)/4);
 const bbStrength=Math.min(1,Math.abs(bbPos));

 let techScore=100*(
   .30*alignment+
   .22*emaStrength+
   .20*macdStrength+
   .14*rsiStrength+
   .09*momentumStrength+
   .05*bbStrength
 );
 techScore=Math.max(0,Math.min(100,techScore));

 const extreme=(R>=78||R<=22||Math.abs(bbPos)>=1.25);
 const coherent=(trend==='ALCISTA' ? (M.hist>=0&&ema9>=ema21) :
                 trend==='BAJISTA' ? (M.hist<=0&&ema9<=ema21) : false);

 const rows=[];
 for(let d=0;d<10;d++){
  const p60=freqProb(h.slice(-60),d);
  const p180=freqProb(h.slice(-180),d);
  const p500=freqProb(h.slice(-500),d);
  const pt=transProb(h.slice(-500),lastDigit,d);

  const matchRisk=.46*p60+.29*p180+.15*p500+.10*pt;
  rows.push({d,matchRisk,p60,p180,p500,pt});
 }
 rows.sort((a,b)=>a.matchRisk-b.matchRisk);

 let pool=rows.filter(x=>x.d!==lastPick);
 if(!pool.length)return null;

 const best=pool[0];
 const second=pool[1]||rows.find(x=>x.d!==best.d)||best;
 const separation=Math.max(0,second.matchRisk-best.matchRisk);

 const gateScore=trend==='LATERAL'?72:64;
 const safe=techScore>=gateScore &&
            coherent &&
            !extreme &&
            best.matchRisk<.105 &&
            separation>=.0010;

 return {
  q:best,
  second,
  safe,
  techScore,
  rsi:R,
  macd:M,
  trend,
  ema9,ema21,ema50,
  bbPos,upper,lower,
  momentum10,
  separation,
  gateScore
 };
}

function showSignal(s){
 lastSignal=s;
 if(!s){
  $('decision').textContent='OBSERVANDO';
  $('reason').textContent='Recolectando datos para EMA, RSI y MACD.';
  $('buy').textContent='COMPRAR AHORA · CALIBRANDO';
  $('sepPick').textContent='—';
  $('risk').textContent='—';
  $('spread').textContent='—';
  $('entropy').textContent='—';
  $('phase').textContent='OBSERVAR';
  $('meter').style.width='0%';
  return;
 }

 const riskPct=s.q.matchRisk*100;
 const sepPct=s.separation*100;
 $('risk').textContent=riskPct.toFixed(2)+'%';
 $('sepPick').textContent='D'+s.q.d;
 $('spread').textContent=sepPct.toFixed(2)+' pp';
 $('entropy').textContent=s.rsi.toFixed(1);
 $('phase').textContent=s.trend;
 $('meter').style.width=Math.min(100,s.techScore)+'%';
 $('buy').textContent='COMPRAR AHORA · D'+s.q.d+' · SCORE '+s.techScore.toFixed(0)+'/100';

 const emaTxt='EMA9 '+s.ema9.toFixed(3)+' · EMA21 '+s.ema21.toFixed(3);
 const macdTxt='MACD-H '+s.macd.hist.toFixed(4)+' · RSI '+s.rsi.toFixed(1);

 if(s.safe){
  $('decision').textContent='SEÑAL TÉCNICA · DIFFER D'+s.q.d;
  $('reason').textContent='Score '+s.techScore.toFixed(0)+'/100 · '+s.trend+' · '+emaTxt+' · '+macdTxt;
 }else{
  $('decision').textContent='ESPERAR · CANDIDATO D'+s.q.d;
  $('reason').textContent='Score '+s.techScore.toFixed(0)+'/100 (mín. '+s.gateScore+') · '+s.trend+' · '+emaTxt+' · '+macdTxt;
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
 if(!s.safe){
  $('status').textContent='SIN CONFIRMACIÓN TÉCNICA';
  return;
 }
 const d=s.q.d,mode=$('mode').value;
 if(mode==='DEMO'&&!window.demoReady){
  $('status').textContent='CONECTA DEMO DERIV';
  return;
 }
 lastPick=d;
 observe=0;
 pending={d,stake,mode};
 ops++;
 $('decision').textContent='COMPRA '+mode+' · DIFFER D'+d;
 $('reason').textContent='$'+stake.toFixed(2)+' · análisis técnico confirmado · duración 1 tick';
 log('COMPRA '+mode+' D'+d+' $'+stake.toFixed(2)+' · SCORE '+s.techScore.toFixed(0));
 ui();

 if(mode==='DEMO'){
  $('status').textContent='ENVIANDO A DERIV…';
  window.sendDemoTrade(d,stake).catch(e=>tradeError(e));
 }else{
  $('status').textContent='SIM ABIERTA';
 }
}

function tradeError(e){
 log('ERROR DEMO '+(e?.message||e));
 pending=null;
 observe=0;
 $('status').textContent='ERROR DEMO · REVISA LOG';
 ui();
}

function finish(profit,label){
 profit=Number(profit);
 if(!Number.isFinite(profit)){
  tradeError(new Error('Resultado inválido'));
  return;
 }
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

 if(pnl>=target()){
  running=false;
  $('status').textContent='META +$'+target().toFixed(2)+' · STOP';
  $('phase').textContent='META';
 }else if(running){
  $('status').textContent='ANÁLISIS TÉCNICO';
 }
 ui();
}

function tick(price,pip){
 const d=Number(Number(price).toFixed(pip).slice(-1));

 if(pending&&pending.mode==='SIM'){
  const p=pending;
  finish(d===p.d?-p.stake:p.stake*.10,'SIM');
 }

 hist.push(d);
 quotes.push(Number(price));
 if(hist.length>1200)hist.shift();
 if(quotes.length>1200)quotes.shift();

 if(running&&!pending)observe++;
 ui(d);
 showSignal(analyse());
}

function connect(){
 clearTimeout(retry);
 ws=new WebSocket('wss://api.derivws.com/trading/v1/options/ws/public');

 ws.onopen=()=>ws.send(JSON.stringify({
  ticks_history:'R_75',
  count:600,
  end:'latest',
  style:'ticks'
 }));

 ws.onmessage=e=>{
  const m=JSON.parse(e.data);

  if(m.history&&m.history.prices){
   const p=Number(m.pip_size||4);
   quotes=m.history.prices.map(Number);
   hist=quotes.map(x=>Number(Number(x).toFixed(p).slice(-1)));
   ws.send(JSON.stringify({ticks:'R_75',subscribe:1}));
   $('status').textContent='ANÁLISIS TÉCNICO · '+hist.length+' TICKS';
   showSignal(analyse());
  }

  if(m.tick){
   const ep=+m.tick.epoch;
   if(ep===lastEpoch)return;
   lastEpoch=ep;
   const p=Number(m.tick.pip_size||4);
   tick(Number(m.tick.quote),p);
  }
 };

 ws.onclose=()=>retry=setTimeout(connect,2500);
}

$('start').onclick=()=>{
 if($('mode').value==='DEMO'&&!window.demoReady){
  $('status').textContent='CONECTA DEMO DERIV PRIMERO';
  return;
 }
 pnl=0;
 stake=baseStake();
 wins=0;
 losses=0;
 ops=0;
 pending=null;
 observe=0;
 lastPick=null;
 running=true;
 $('status').textContent='ANÁLISIS TÉCNICO';
 log('NUEVA SESIÓN '+$('mode').value+' · STAKE $'+stake.toFixed(2)+' · META $'+target().toFixed(2));
 ui();
};

$('stop').onclick=()=>{
 running=false;
 $('status').textContent='STOP MANUAL';
};

$('buy').onclick=()=>{
 if(!running){
  $('status').textContent='PULSA REINICIAR SESIÓN';
  return;
 }
 if(pending){
  $('status').textContent='OPERACIÓN EN CURSO';
  return;
 }
 const s=lastSignal;
 if(!s){
  $('status').textContent='AÚN CALIBRANDO';
  return;
 }
 enter(s);
};

window.demoSettlement=p=>finish(p,'DERIV DEMO');
window.demoTradeError=tradeError;

stake=baseStake();
$('status').textContent='ANÁLISIS TÉCNICO ACTIVO';
ui();
connect();
