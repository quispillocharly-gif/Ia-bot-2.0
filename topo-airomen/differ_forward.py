#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil, time
import numpy as np
import joblib
from scipy.stats import binomtest, chisquare

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
CAND=M/"differ_candidate.joblib"; CANDMETA=M/"differ_candidate.json"
FROZEN=M/"differ_frozen.joblib"; FMETA=M/"differ_frozen.json"
STATE=M/"differ_forward_state.json"; OUT=M/"differ_forward_latest.json"; PAYOUT=M/"differ_payout_snapshot.json"
BASE=.90; PAYOUT_MAX_AGE=7200

VARIANTS=[
 {"id":"neural_argmin","type":"neural","max_p":1.0},
 {"id":"neural_p09","type":"neural","max_p":.09},
 {"id":"neural_p08","type":"neural","max_p":.08},
 {"id":"neural_p07","type":"neural","max_p":.07},
 {"id":"cold_consensus","type":"consensus","max_p":.09},
 {"id":"strict_consensus","type":"strict","max_p":.08},
]
TESTS=len(VARIANTS)

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def cold(h,n):
    a=h[-n:]; c=[a.count(d) for d in range(10)]
    return min(range(10),key=lambda d:(c[d],d))

def trans_cold(h,o):
    a=h[-1000:]
    if len(a)<=o:return cold(a,80)
    ctx=tuple(a[-o:]); c=[0]*10
    for i in range(o,len(a)):
        if tuple(a[i-o:i])==ctx:c[a[i]]+=1
    return min(range(10),key=lambda d:(c[d],d)) if sum(c) else cold(a,80)

def payout_info():
    try:p=json.loads(PAYOUT.read_text())
    except Exception:return {},None,None,False
    rows={}
    for x in p.get("per_digit",[]):
        if x.get("ask_price") is not None and x.get("payout") is not None:
            rows[int(x["digit"])]=(float(x["ask_price"]),float(x["payout"]))
    be=p.get("max_break_even") if p.get("status")=="OK" else None
    ts=p.get("timestamp"); age=max(0,int(time.time())-int(ts)) if ts is not None else None
    fresh=bool(age is not None and age<=PAYOUT_MAX_AGE and len(rows)==10 and be is not None)
    return rows,float(be) if be is not None else None,age,fresh

def freeze(ticks):
    if not CAND.exists() or not CANDMETA.exists(): raise RuntimeError("No DIFFER candidate")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=json.loads(FMETA.read_text()); last=int(ticks[-1]["epoch"])
    st={"model_id":meta["model_id"],"start_epoch":last,"last_epoch":last,"runs":0,
        "history":[int(x["digit"]) for x in ticks[-1500:]],"opportunities":0,
        "digit_counts":[0]*10,"forward_ticks":0,
        "variants":{v["id"]:{"n":0,"w":0,"pnl":0.0,"staked":0.0,"outcomes":[]} for v in VARIANTS}}
    STATE.write_text(json.dumps(st,indent=2)); return st

def blocks(outcomes):
    if len(outcomes)<600:return []
    arr=np.asarray(outcomes,dtype=np.int8)
    return [{"n":len(p),"wins":int(p.sum()),"hit_rate":float(p.mean()),"wilson_lower":wilson(int(p.sum()),len(p))}
            for p in np.array_split(arr,3)]

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    if not ticks: raise RuntimeError("No ticks")
    pmap,be,payout_age,payout_fresh=payout_info()
    if STATE.exists() and FROZEN.exists() and FMETA.exists(): st=json.loads(STATE.read_text())
    else: st=freeze(ticks)
    bundle=joblib.load(FROZEN); clf=bundle["clf"]; enc=bundle["encoder"]; window=int(bundle["window"])
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st["history"]))
    for x in fresh:
        d=int(x["digit"]); st["forward_ticks"]=int(st.get("forward_ticks",0))+1
        counts=list(st.get("digit_counts",[0]*10)); counts[d]+=1; st["digit_counts"]=counts
        if len(hist)>=max(100,window):
            st["opportunities"]+=1
            X=enc.transform(np.asarray([hist[-window:]],dtype=np.int16)); proba=clf.predict_proba(X)[0]
            full=np.ones(10,dtype=float)
            for j,c in enumerate(clf.classes_): full[int(c)]=float(proba[j])
            barrier=int(np.argmin(full)); minp=float(full[barrier])
            c20,c40,t1=cold(hist,20),cold(hist,40),trans_cold(hist,1)
            agree=sum(int(z==barrier) for z in (c20,c40,t1))
            for cfg in VARIANTS:
                if cfg["type"]=="neural": emit=minp<=cfg["max_p"]
                elif cfg["type"]=="consensus": emit=minp<=cfg["max_p"] and agree>=1
                else: emit=minp<=cfg["max_p"] and agree>=2
                if not emit: continue
                hit=int(d!=barrier); s=st["variants"][cfg["id"]]
                s["n"]+=1; s["w"]+=hit; s["outcomes"].append(hit)
                if len(s["outcomes"])>5000:s["outcomes"]=s["outcomes"][-5000:]
                if barrier in pmap:
                    ask,payout=pmap[barrier]; delta=(payout-ask) if hit else -ask
                    s["pnl"]+=delta; s["staked"]+=ask
        hist.append(d)
        if len(hist)>1500:hist=hist[-1500:]
        st["last_epoch"]=int(x["epoch"])
    st["history"]=hist; st["runs"]=int(st["runs"])+1; STATE.write_text(json.dumps(st,indent=2))
    threshold=be if be is not None else BASE; rows=[]
    for cfg in VARIANTS:
        s=st["variants"][cfg["id"]]; n=int(s["n"]); w=int(s["w"]); rate=w/n if n else None
        raw=float(binomtest(w,n,p=threshold,alternative="greater").pvalue) if n else 1.0
        adj=min(1.0,raw*TESTS); pnl=float(s["pnl"]); staked=float(s["staked"]); bs=blocks(s["outcomes"])
        stable=bool(len(bs)==3 and all(b["hit_rate"]>threshold for b in bs))
        confirmed=bool(n>=2000 and payout_fresh and wilson(w,n)>threshold and pnl>0 and adj<.05 and stable)
        rows.append({"id":cfg["id"],"signals":n,"wins":w,"losses":n-w,"hit_rate":rate,
                     "wilson_lower":wilson(w,n),"raw_economic_p":raw,"bonferroni_p":adj,
                     "shadow_pnl":pnl,"shadow_roi":pnl/staked if staked else None,
                     "temporal_blocks":bs,"block_stable":stable,
                     "status":"ECONOMIC_CANDIDATE" if confirmed else ("COLLECTING" if n<2000 else "NOT_CONFIRMED")})
    rows.sort(key=lambda x:(x["status"]=="ECONOMIC_CANDIDATE",x["wilson_lower"],x["signals"]),reverse=True)
    counts=np.asarray(st.get("digit_counts",[0]*10),dtype=float); total=int(counts.sum())
    if total>=100:
        chi=chisquare(counts,f_exp=np.full(10,total/10)); chi_p=float(chi.pvalue); maxdev=float(np.max(np.abs(counts/total-.1)))
    else: chi_p=None; maxdev=None
    out={"version":"1.0-differ-forward-lab","timestamp":int(time.time()),"runs":st["runs"],
         "model_id":st["model_id"],"start_epoch":st["start_epoch"],"new_ticks_this_run":len(fresh),
         "opportunities":int(st["opportunities"]),"baseline_random":BASE,"tests":TESTS,
         "break_even_rate":be,"payout_fresh":payout_fresh,"payout_age_seconds":payout_age,
         "data_quality":{"forward_ticks":int(st.get("forward_ticks",0)),"digit_counts":[int(x) for x in counts],
                         "uniformity_chi_square_p":chi_p,"max_digit_share_deviation":maxdev},
         "leader":rows[0] if rows else None,"variants":rows,
         "confirmed_variants":[x["id"] for x in rows if x["status"]=="ECONOMIC_CANDIDATE"],
         "status":"EDGE_CANDIDATE" if any(x["status"]=="ECONOMIC_CANDIDATE" for x in rows) else "RESEARCHING",
         "note":"Forward-only DIFFER shadow lab. 100% is not assumed or guaranteed. Promotion requires economic break-even, Wilson lower bound, Bonferroni significance, positive shadow P&L and stability across three time blocks."}
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
