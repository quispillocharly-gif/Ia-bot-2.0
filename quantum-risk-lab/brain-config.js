window.QUANTUM_CONFIG={baseStake:1,target:3};
(function(){
 const bs=document.getElementById('baseStake'),ti=document.getElementById('targetInput'),tv=document.getElementById('targetView');
 function read(){
  let b=Number(bs.value),t=Number(ti.value);
  window.QUANTUM_CONFIG.baseStake=Number.isFinite(b)&&b>0?b:1;
  window.QUANTUM_CONFIG.target=Number.isFinite(t)&&t>0?t:3;
  bs.value=window.QUANTUM_CONFIG.baseStake.toFixed(2);
  ti.value=window.QUANTUM_CONFIG.target.toFixed(2);
  tv.textContent='+$'+window.QUANTUM_CONFIG.target.toFixed(2);
 }
 bs.addEventListener('change',read);ti.addEventListener('change',read);read();
})();