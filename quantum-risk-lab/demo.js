(()=>{let socket=null,ready=false,currency='USD',accountId=null,busy=false,proposalReq=1000,buyReq=2000,pingReq=5000;
let reconnectTimer=null,keepAliveTimer=null,watchdogTimer=null,reconnectAttempt=0,autoReconnect=false,everConnected=false,lastMessageAt=0;
const $=x=>document.getElementById(x);
const set=s=>{$('authStatus').textContent=s};
const errMsg=(j,fallback)=>j?.errors?.[0]?.message||j?.error?.message||fallback;
const hasCreds=()=>!!($('pat').value.trim()&&$('app').value.trim());

async function api(path,opts={}){
 const pat=$('pat').value.trim(),app=$('app').value.trim();
 const r=await fetch('https://api.derivws.com'+path,{...opts,headers:{'Authorization':'Bearer '+pat,'Deriv-App-ID':app,'Content-Type':'application/json',...(opts.headers||{})}});
 let j={};try{j=await r.json()}catch(_){}
 if(!r.ok)throw new Error(errMsg(j,'HTTP '+r.status));
 return j;
}
function clearTimers(){
 clearInterval(keepAliveTimer);keepAliveTimer=null;
 clearInterval(watchdogTimer);watchdogTimer=null;
}
function safeSend(obj){
 if(socket&&socket.readyState===WebSocket.OPEN){
  try{socket.send(JSON.stringify(obj));return true}catch(_){}
 }
 return false;
}
function startKeepAlive(){
 clearTimers();
 lastMessageAt=Date.now();
 keepAliveTimer=setInterval(()=>{
  if(socket&&socket.readyState===WebSocket.OPEN){
   safeSend({ping:1,req_id:++pingReq});
  }
 },30000);
 watchdogTimer=setInterval(()=>{
  if(!socket||socket.readyState!==WebSocket.OPEN)return;
  if(Date.now()-lastMessageAt>90000){
   set('DEMO · RECUPERANDO CONEXIÓN…');
   try{socket.close()}catch(_){}
  }
 },10000);
}
function scheduleReconnect(){
 if(!autoReconnect||!hasCreds())return;
 if(reconnectTimer)return;
 const steps=[1500,3000,6000,12000,20000,30000];
 const delay=steps[Math.min(reconnectAttempt,steps.length-1)];
 reconnectAttempt++;
 set('DEMO DESCONECTADO · RECONECTA EN '+Math.ceil(delay/1000)+'s');
 reconnectTimer=setTimeout(()=>{
  reconnectTimer=null;
  connectDemo(true);
 },delay);
}

async function connectDemo(isAuto=false){
 if(busy)return;
 if(!hasCreds()){
  ready=false;window.demoReady=false;
  set('Falta App ID o PAT');
  return;
 }
 autoReconnect=true;
 busy=true;ready=false;window.demoReady=false;
 if(!isAuto)set('CONECTANDO…');else set('RECONECTANDO…');
 try{
  const a=await api('/trading/v1/options/accounts',{method:'GET'});
  const rows=Array.isArray(a.data)?a.data:(a.data?[a.data]:[]);
  const demo=rows.find(x=>String(x.account_type||'').toLowerCase()==='demo'&&String(x.status||'active').toLowerCase()==='active');
  if(!demo)throw new Error('No encontré una cuenta Options DEMO activa');
  accountId=demo.account_id;currency=demo.currency||'USD';

  // Cada reconexión solicita un OTP nuevo: los OTP son de un solo uso.
  const o=await api('/trading/v1/options/accounts/'+encodeURIComponent(accountId)+'/otp',{method:'POST'});
  const url=o?.data?.url;
  if(!url)throw new Error('Deriv no devolvió URL OTP');
  if(!/\/ws\/demo\?otp=/i.test(url))throw new Error('BLOQUEADO: la cuenta seleccionada no es DEMO');

  const previous=socket;
  const ws=new WebSocket(url);
  socket=ws;
  if(previous&&previous!==ws)try{previous.close()}catch(_){}

  ws.onopen=()=>{
   if(socket!==ws)return;
   busy=false;ready=true;window.demoReady=true;everConnected=true;reconnectAttempt=0;
   clearTimeout(reconnectTimer);reconnectTimer=null;
   set('DEMO CONECTADO · '+accountId+' · AUTO-RECONEXIÓN ON');
   startKeepAlive();

   // Si la conexión cayó durante un contrato, recuperamos su seguimiento.
   if(activeContract){
    safeSend({proposal_open_contract:1,contract_id:activeContract,subscribe:1,req_id:3001});
   }
  };
  ws.onerror=()=>{
   if(socket!==ws)return;
   ready=false;window.demoReady=false;
   set('DEMO · ERROR DE CONEXIÓN');
  };
  ws.onclose=()=>{
   if(socket!==ws)return;
   busy=false;ready=false;window.demoReady=false;clearTimers();
   if(pendingProposal){
    const rej=pendingProposal.reject;
    pendingProposal=null;
    try{rej(new Error('Conexión interrumpida; Deriv se está reconectando'))}catch(_){}
   }
   scheduleReconnect();
  };
  ws.onmessage=ev=>{
   if(socket!==ws)return;
   lastMessageAt=Date.now();
   onMessage(ev);
  };
 }catch(e){
  busy=false;ready=false;window.demoReady=false;clearTimers();
  set('ERROR · '+e.message);
  // Solo reintentamos automáticamente si la sesión ya había conectado antes.
  if(isAuto||everConnected)scheduleReconnect();
 }
}

let pendingProposal=null,activeContract=null,settled=new Set();
function onMessage(ev){
 let m;try{m=JSON.parse(ev.data)}catch(_){return}
 if(m.error){
  if(pendingProposal){
   const rej=pendingProposal.reject;pendingProposal=null;
   rej(new Error(m.error.message||'Error Deriv'));
  }else{
   window.demoTradeError?.(new Error(m.error.message||'Error Deriv'));
  }
  return;
 }
 if(m.msg_type==='proposal'&&pendingProposal){
  const p=pendingProposal;pendingProposal=null;
  const id=m.proposal?.id,ask=Number(m.proposal?.ask_price);
  if(!id||!Number.isFinite(ask)){p.reject(new Error('Propuesta inválida'));return}
  safeSend({buy:id,price:ask,req_id:++buyReq});
  p.resolve();
 }
 if(m.msg_type==='buy'&&m.buy?.contract_id){
  activeContract=m.buy.contract_id;
  set('DEMO CONECTADO · '+accountId+' · CONTRATO '+activeContract);
  safeSend({proposal_open_contract:1,contract_id:activeContract,subscribe:1,req_id:3001});
 }
 if(m.msg_type==='proposal_open_contract'&&m.proposal_open_contract){
  const c=m.proposal_open_contract,id=c.contract_id;
  if(c.is_sold&&!settled.has(id)){
   settled.add(id);activeContract=null;
   const profit=Number(c.profit);
   window.demoSettlement?.(profit);
   set('DEMO CONECTADO · '+accountId+' · AUTO-RECONEXIÓN ON');
   if(m.subscription?.id)safeSend({forget:m.subscription.id});
  }
 }
}

window.sendDemoTrade=(digit,stake)=>new Promise((resolve,reject)=>{
 if(!ready||!socket||socket.readyState!==WebSocket.OPEN){
  reject(new Error('Demo Deriv reconectando; espera conexión'));
  return;
 }
 if(activeContract||pendingProposal){
  reject(new Error('Ya existe una operación en curso'));
  return;
 }
 digit=Number(digit);stake=Number(stake);
 if(!Number.isInteger(digit)||digit<0||digit>9||!Number.isFinite(stake)||stake<=0){
  reject(new Error('Parámetros inválidos'));
  return;
 }
 pendingProposal={resolve,reject};
 safeSend({proposal:1,amount:Number(stake.toFixed(2)),basis:'stake',contract_type:'DIGITDIFF',currency,barrier:String(digit),duration:1,duration_unit:'t',underlying_symbol:'R_75',req_id:++proposalReq});
});

$('auth').onclick=()=>connectDemo(false);
window.addEventListener('online',()=>{
 if(autoReconnect&&!ready&&!busy)connectDemo(true);
});
window.addEventListener('offline',()=>{
 ready=false;window.demoReady=false;
 set('SIN INTERNET · ESPERANDO RED');
});
document.addEventListener('visibilitychange',()=>{
 if(document.visibilityState==='visible'&&autoReconnect&&!ready&&!busy){
  connectDemo(true);
 }else if(document.visibilityState==='visible'&&ready){
  safeSend({ping:1,req_id:++pingReq});
 }
});
window.demoReady=false;
})();