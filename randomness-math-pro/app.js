const $=x=>document.getElementById(x);let seq=[],run=false,pending=null,lastBought=null,pnl=0,stake=1,w=0,l=0,ops=0;
function log(x){$('log').textContent=x+'\n'+$('log').textContent}
function entropy(a){let c=Array(10).fill(0);a.forEach(x=>c[x]++);return c.reduce((h,x)=>{if(!x)return h;let p=x/a.length;return h-p*Math.log2(p)},0)}
function analyze(){if(seq.length<120)return null;let s=Array(10).fill(0),e=Array.from({length:10},()=>[]);
for(const n of [10,20,40,80]){let c=Array(10).fill(0);seq.slice(-n).forEach(x=>c[x]++);let sd=Math.sqrt(n*.09);for(let d=0;d<10;d++){let z=(c[d]-n*.1)/sd;if(z<0){s[d]+=Math.min(2.2,-z);e[d].push('Z'+n+'='+z.toFixed(2))}}}
let c=Array(10).fill(0);seq.slice(-80).forEach(x=>c[x]++);let chi=c.reduce((q,x)=>q+(x-8)*(x-8)/8,0),g=c.reduce((q,x)=>x?q+2*x*Math.log(x/8):q,0);for(let d=0;d<10;d++)if(c[d]<8){if(chi>=8){s[d]+=.35;e[d].push('CHI='+chi.toFixed(1))}if(g>=8){s[d]+=.3;e[d].push('G='+g.toFixed(1))}}
let r=seq.slice(-200),last=r.at(-1),tr=Array.from({length:10},()=>Array(10).fill(1));for(let i=1;i<r.length;i++)tr[r[i-1]][r[i]]++;let row=tr[last],rs=row.reduce((q,x)=>q+x,0);for(let d=0;d<10;d++){let p=row[d]/rs;if(p<.1){s[d]+=(.1-p)*15;e[d].push('M1='+(p*100).toFixed(1)+'%')}}
let ranked=s.map((v,d)=>({d,s:v,e:e[d]})).filter(x=>x.d%2===1&&x.d!==lastBought).sort((a,b)=>b.s-a.s),best=ranked[0];if(!best)return null;let band=ranked.filter(x=>x.s>=2.8&&x.e.length>=3&&x.s>=best.s-.8);return {ok:band.length>0,p:band.length?band[Math.floor(Math.random()*band.length)]:best,H:entropy(seq.slice(-100)),chi,g}}
function ui(){$('pnl').textContent=(pnl>=0?'+':'')+'$'+pnl.toFixed(2);$('stake').textContent='$'+stake.toFixed(2);$('win').textContent=w;$('match').textContent=l;$('ops').textContent=ops}
