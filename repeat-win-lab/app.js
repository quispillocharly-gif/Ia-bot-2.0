const $=x=>document.getElementById(x);
let seq=[],run=false,pending=null,lastBought=null,pnl=0,stake=1,w=0,l=0,ops=0,watch=new Set(),priority=null,repeatBlocked=null;
function log(x){$('log').textContent=x+'\n'+$('log').textContent}
function entropy(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);return c.reduce((h,x)=>{if(!x)return h;let p=x/a.length;return h-p*Math.log2(p)},0)}
function repeatBlock(){let a=seq.slice(-20),c=Array(10).fill(0);a.forEach(x=>c[x]++);let m=Math.max(...c),q=c.map((v,d)=>({d,v})).filter(x=>x.v===m);return m>=3&&q.length===1?q[0].d:null}
function analyze(){if(seq.length<120)return null;let s=Array(10).fill(0),e=Array.from({length:10},()=>[]);
for(const n of [10,20,40,80]){let a=seq.slice(-n),sd=Math.sqrt(a.length*.09);for(let d=0;d<10;d++){let z=(a.filter(x=>x===d).length-a.length*.1)/sd;if(z<0){s[d]+=Math.min(2.2,-z);e[d].push('Z'+n+'='+z.toFixed(2))}}}
let r=seq.slice(-200),last=r.at(-1),tr=Array.from({length:10},()=>Array(10).fill(1));for(let i=1;i<r.length;i++)tr[r[i-1]][r[i]]++;let total=tr[last].reduce((a,b)=>a+b,0);for(let d=0;d<10;d++){let p=tr[last][d]/total;if(p<.1){s[d]+=(.1-p)*15;e[d].push('M1='+(p*100).toFixed(1)+'%')}}
repeatBlocked=repeatBlock();let pool=s.map((v,d)=>({d,s:v,e:e[d]})).filter(x=>x.d!==lastBought&&x.d!==repeatBlocked);if(!pool.length)return null;pool.sort((a,b)=>b.s-a.s);return pool[0]}
function ui(){$('pnl').textContent=(pnl>=0?'+':'')+'$'+pnl.toFixed(2);$('stake').textContent='$'+stake.toFixed(2);$('win').textContent=w;$('match').textContent=l;$('ops').textContent=ops}
function decide(){if(!run||pending)return;let q=analyze();if(!q){$('status').textContent='CALIBRANDO '+seq.length+'/120';return}let p=priority!==null?{d:priority,e:['RECOMPRA POST-WIN']}:q;if(priority!==null){log('PRIORIDAD D'+priority+' · REAPARECIÓ DESPUÉS DE WIN');priority=null}lastBought=p.d;ops++;pending={d:p.d,left:1,stake};$('decision').textContent='D'+p.d+' · 1T';$('evidence').textContent=p.e.join(' · ')+' | H='+entropy(seq.slice(-100)).toFixed(3);$('status').textContent='SIM DIFFER D'+p.d;ui()}
function settle(d){if(!pending)return;let p=pending;if(--p.left>0)return;if(d!==p.d){let x=p.stake*.1;pnl+=x;stake=p.stake+x;w++;watch.add(p.d);log('WIN D'+p.d+' · VIGILANDO D'+p.d)}else{pnl-=p.stake;stake=1;l++;log('MATCH D'+p.d)}pending=null;if(pnl>=3){run=false;$('status').textContent='META +$3 · STOP'}ui()}
function tick(d){if(watch.has(d)){watch.delete(d);priority=d;log('D'+d+' REAPARECIÓ · PREPARANDO RECOMPRA D'+d)}seq.push(d);if(seq.length>2500)seq.shift();settle(d);if(run&&!pending)decide()}
let ws=null,lastEpoch=0,retry=null;
function connect(){ws=new WebSocket('wss://api.derivws.com/trading/v1/options/ws/public');ws.onopen=()=>ws.send(JSON.stringify({ticks_history:'R_75',count:700,end:'latest',style:'ticks'}));ws.onmessage=e=>{let m=JSON.parse(e.data);if(m.history&&m.history.prices){let p=Number(m.pip_size||4);seq=m.history.prices.map(x=>Number(Number(x).toFixed(p).slice(-1)));ws.send(JSON.stringify({ticks:'R_75',subscribe:1}));$('status').textContent='TICKS EN VIVO'}if(m.tick){let ep=+m.tick.epoch;if(ep===lastEpoch)return;lastEpoch=ep;let p=Number(m.tick.pip_size||4);tick(Number(Number(m.tick.quote).toFixed(p).slice(-1)))}};ws.onclose=()=>retry=setTimeout(connect,2500)}
$('start').onclick=()=>{pnl=0;stake=1;w=l=ops=0;pending=null;run=true;ui();log('REPEAT WIN LAB INICIADO');decide()};
$('stop').onclick=()=>{run=false;pending=null;$('status').textContent='STOP MANUAL'};
$('mode').innerHTML='<option>SIM</option>';$('auth').disabled=true;$('auths').textContent='Proyecto SIM independiente';
ui();connect();