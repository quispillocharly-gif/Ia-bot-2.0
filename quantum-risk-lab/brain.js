const $=x=>document.getElementById(x);
let running=true,pending=null,pnl=0,stake=1,wins=0,losses=0,ops=0,hist=[],lastEpoch=0,ws,retry,observe=0,lastPick=null,lastSignal=null;
const cfg=()=>window.QUANTUM_CONFIG||{baseStake:1,target:3};
const baseStake=()=>Math.max(.01,Number(cfg().baseStake)||1);
const target=()=>Math.max(.01,Number(cfg().target)||3);
function log(s){$('log').textContent=s+'\n'+$('log').textContent}
function ent(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);let h=0;c.forEach(v=>{if(v){let p=v/a.length;h-=p*Math.log2(p)}});return h}
function analyse(){
 if(hist.length<180)return null;
 let last=hist[hist.length-1],w12=hist.slice(-12),w36=hist.slice(-36),w120=hist.slice(-120),f12=Array(10).fill(0),f36=Array(10).fill(0),f120=Array(10).fill(0),tr=Array(10).fill(0);
 w12.forEach(x=>f12[x]++);w36.forEach(x=>f36[x]++);w120.forEach(x=>f120[x]++);
 for(let i=Math.max(1,hist.length-300);i<hist.length;i++)if(hist[i-1]===last)tr[hist[i]]++;
 let tt=tr.reduce((a,b)=>a+b,0),H=ent(w36),rows=[];
 for(let d=0;d<10;d++){
  let gap=0;for(let i=hist.length-1;i>=0&&gap<50;i--){if(hist[i]===d)break;gap++}
  let streak=0;for(let i=hist.length-1;i>=0&&hist[i]===d;i--)streak++;
  let p12=f12[d]/12,p36=f36[d]/36,p120=f120[d]/120,pt=tt?tr[d]/tt:.1;
  let hot=Math.max(0,p12-.10)*1.7+Math.max(0,p36-.10)*.75,transition=Math.max(0,pt-.10)*1.35,recency=gap===0?.11:gap===1?.06:gap===2?.03:0,repeat=streak*.075;
  let risk=.40*p12+.23*p36+.17*p120+.20*pt+hot+transition+recency+repeat;
  rows.push({d,risk,p12,p36,pt,gap});
 }
 rows.sort((a,b)=>a.risk-b.risk);
 let pool=rows.filter(x=>x.d!==lastPick),q=pool[0],second=pool[1];if(!q||!second)return null;
 let spread=second.risk-q.risk,near=pnl>=target()*.75,need=near?.05:.03,maxRisk=near?.085:.105;
 let safe=q.risk<maxRisk&&spread>=need&&q.p12<=.10&&q.pt<=.13;
 return{q,spread,H,near,safe};
}
function showSignal(s){
 lastSignal=s;
 if(!s){$('decision').textContent='OBSERVANDO';$('reason').textContent='Aún no existe suficiente historial.';$('buy').textContent='COMPRAR AHORA · CALIBRANDO';return}
 $('risk').textContent=(s.q.risk*100).toFixed(1)+'%';
 $('buy').textContent='COMPRAR AHORA · D'+s.q.d+' · RIESGO '+(s.q.risk*100).toFixed(1)+'%';
 $('spread').textContent=(s.spread*100).toFixed(1);$('entropy').textContent=s.H.toFixed(2);$('phase').textContent=s.near?'MODO META':'ANÁLISIS';$('meter').style.width=Math.min(100,s.spread*1000)+'%';
 if(!s.safe){$('decision').textContent='NO OPERAR';$('reason').textContent='La separación estadística no supera el filtro de seguridad.'}
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
$('buy').onclick=()=>{if(!running){$('status').textContent='PULSA REINICIAR SESIÓN';return}if(pending){$('status').textContent='OPERACIÓN EN CURSO';return}let s=analyse();if(!s){$('status').textContent='AÚN CALIBRANDO';return}enter(s)};
window.demoSettlement=p=>finish(p,'DERIV DEMO');
window.demoTradeError=tradeError;
stake=baseStake();$('status').textContent='ANÁLISIS ACTIVO';ui();connect();