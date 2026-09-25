#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil, time
import numpy as np
import joblib
from scipy.stats import binomtest

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
CAND=M/"neural_v5_candidate.joblib"; CANDMETA=M/"neural_v5_candidate.json"
FROZEN=M/"v8_frozen_neural.joblib"; FMETA=M/"v8_frozen_neural.json"
STATE=M/"v8_challenger_state.json"; OUT=M/"v8_challenger_latest.json"
PAYOUT=M/"payout_snapshot.json"
BASE=.10; PAYOUT_MAX_AGE=7200

VARIANTS=[
 {"id":"agree3_all","min_agree":3,"min_conf":0.0,"regimes":None},
 {"id":"agree4_all","min_agree":4,"min_conf":0.0,"regimes":None},
 {"id":"conf14_agree3","min_agree":3,"min_conf":0.14,"regimes":None},
 {"id":"conf18_agree3","min_agree":3,"min_conf":0.18,"regimes":None},
 {"id":"conf22_agree2","min_agree":2,"min_conf":0.22,"regimes":None},
 {"id":"stable_regimes","min_agree":3,"min_conf":0.14,"regimes":["UNIFORM","CONCENTRATED"]},
]
TESTS=len(VARIANTS)

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def entropy_norm(h,n=200):
    a=h[-n:]
    if not a:return 1.0
    c=np.bincount(np.asarray(a,dtype=np.int16),minlength=10).astype(float)
    p=c/c.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/math.log(10))

def repeat_rate(h,n=200):
    a=h[-n:]
    return 0.0 if len(a)<2 else float(sum(a[i]==a[i-1] for i in range(1,len(a)))/(len(a)-1))

def max_share(h,n=200):
    a=h[-n:]
    if not a:return .1
    c=np.bincount(np.asarray(a,dtype=np.int16),minlength=10)
    return float(c.max()/len(a))

def transition_peak(h,n=500):
    a=h[-n:]
    if len(a)<20:return .1
    mat=np.zeros((10,10),dtype=np.int32)
    for x,y in zip(a[:-1],a[1:]):mat[x,y]+=1
    vals=[]
    for r in mat:
        if r.sum():vals.append(float(r.max()/r.sum()))
    return float(np.mean(vals)) if vals else .1

def regime(h):
    en=entropy_norm(h); rep=repeat_rate(h); mx=max_share(h); tp=transition_peak(h)
    if en<.955 or mx>.155:name="CONCENTRATED"
    elif rep>.125:name="REPEAT_HEAVY"
    elif tp>.18:name="TRANSITIONAL"
    else:name="UNIFORM"
    return name

def hot(h,n):
    a=h[-n:]; c=[a.count(d) for d in range(10)]
    return max(range(10),key=lambda d:(c[d],-d))
def cold(h,n):
    a=h[-n:]; c=[a.count(d) for d in range(10)]
    return min(range(10),key=lambda d:(c[d],d))
def gap(h):
    g=[]
    for d in range(10):
        try:v=len(h)-1-max(i for i,x in enumerate(h) if x==d)
        except ValueError:v=len(h)+1
        g.append(v)
    return max(range(10),key=lambda d:(g[d],-d))
def trans(h,o):
    a=h[-1000:]
    if len(a)<=o:return hot(a,80)
    ctx=tuple(a[-o:]); c=[0]*10
    for i in range(o,len(a)):
        if tuple(a[i-o:i])==ctx:c[a[i]]+=1
    return max(range(10),key=lambda d:(c[d],-d)) if sum(c) else hot(a,80)

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
    if not CAND.exists() or not CANDMETA.exists():raise RuntimeError("No neural candidate")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=json.loads(FMETA.read_text()); latest=int(ticks[-1]["epoch"])
    st={"version":"8.0","model_id":meta["model_id"],"start_epoch":latest,"last_epoch":latest,
        "runs":0,"history":[int(x["digit"]) for x in ticks[-1500:]],"opportunities":0,
        "variants":{v["id"]:{"n":0,"w":0,"pnl":0.0,"staked":0.0,"outcomes":[]} for v in VARIANTS}}
    STATE.write_text(json.dumps(st,indent=2)); return st

def temporal_blocks(outcomes):
    n=len(outcomes)
    if n<300:return []
    parts=np.array_split(np.asarray(outcomes,dtype=np.int8),3)
    rows=[]
    for p in parts:
        m=len(p); w=int(p.sum())
        rows.append({"n":m,"wins":w,"hit_rate":w/m if m else None,"wilson_lower":wilson(w,m)})
    return rows

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    if not ticks:raise RuntimeError("No ticks")
    pmap,be,payout_age,payout_fresh=payout_info()
    if STATE.exists() and FROZEN.exists() and FMETA.exists():st=json.loads(STATE.read_text())
    else:st=freeze(ticks)

    bundle=joblib.load(FROZEN); meta=json.loads(FMETA.read_text())
    window=int(bundle["window"]); clf=bundle["clf"]; enc=bundle["encoder"]
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[])))

    for x in fresh:
        d=int(x["digit"])
        if len(hist)>=max(200,window):
            st["opportunities"]+=1
            X=enc.transform(np.asarray([hist[-window:]],dtype=np.int16))
            proba=clf.predict_proba(X)[0]; j=int(np.argmax(proba))
            pred=int(clf.classes_[j]); conf=float(proba[j]); hit=int(pred==d)
            reg=regime(hist)
            experts=[hot(hist,20),cold(hist,20),hot(hist,40),cold(hist,40),gap(hist),trans(hist,1),trans(hist,2),trans(hist,3)]
            agree=sum(int(p==pred) for p in experts)
            for cfg in VARIANTS:
                if cfg["regimes"] is not None and reg not in cfg["regimes"]:continue
                if agree<cfg["min_agree"] or conf<cfg["min_conf"]:continue
                s=st["variants"][cfg["id"]]; s["n"]+=1; s["w"]+=hit
                s["outcomes"].append(hit)
                if len(s["outcomes"])>3000:s["outcomes"]=s["outcomes"][-3000:]
                if pred in pmap:
                    ask,payout=pmap[pred]; delta=(payout-ask) if hit else -ask
                    s["pnl"]+=delta; s["staked"]+=ask
        hist.append(d)
        if len(hist)>1500:hist=hist[-1500:]
        st["last_epoch"]=int(x["epoch"])

    st["history"]=hist; st["runs"]=int(st.get("runs",0))+1
    STATE.write_text(json.dumps(st,indent=2))

    rows=[]
    threshold=be if be is not None else BASE
    for cfg in VARIANTS:
        s=st["variants"][cfg["id"]]; n=int(s["n"]); w=int(s["w"])
        rate=w/n if n else None; low=wilson(w,n)
        raw_p=float(binomtest(w,n,p=threshold,alternative="greater").pvalue) if n else 1.0
        adj=min(1.0,raw_p*TESTS)
        pnl=float(s["pnl"]); staked=float(s["staked"])
        blocks=temporal_blocks(s["outcomes"])
        stable=bool(len(blocks)==3 and sum(1 for b in blocks if b["hit_rate"] is not None and b["hit_rate"]>threshold)>=2 and min(b["wilson_lower"] for b in blocks)>BASE*.7)
        confirmed=bool(n>=1000 and payout_fresh and low>threshold and pnl>0 and adj<.05 and stable)
        rows.append({
          "id":cfg["id"],"min_agree":cfg["min_agree"],"min_conf":cfg["min_conf"],"regimes":cfg["regimes"],
          "signals":n,"wins":w,"hit_rate":rate,"wilson_lower":low,
          "raw_economic_p":raw_p,"bonferroni_p":adj,
          "shadow_pnl":pnl,"shadow_roi":pnl/staked if staked else None,
          "temporal_blocks":blocks,"block_stable":stable,
          "status":"ECONOMIC_CANDIDATE" if confirmed else ("COLLECTING" if n<1000 else "NOT_CONFIRMED")
        })
    rows.sort(key=lambda x:(x["status"]=="ECONOMIC_CANDIDATE",x["wilson_lower"],x["signals"]),reverse=True)
    confirmed=[x["id"] for x in rows if x["status"]=="ECONOMIC_CANDIDATE"]
    out={
      "version":"8.0-prospective-challenger-lab","timestamp":int(time.time()),"runs":st["runs"],
      "model_id":st["model_id"],"start_epoch":st["start_epoch"],"new_ticks_this_run":len(fresh),
      "opportunities":int(st["opportunities"]),"tests":TESTS,
      "break_even_rate":be,"payout_fresh":payout_fresh,"payout_age_seconds":payout_age,
      "confirmed_variants":confirmed,"status":"EDGE_CANDIDATE" if confirmed else "RESEARCHING",
      "leader":rows[0] if rows else None,"variants":rows,
      "note":"Prospective-only challenger tournament. Six variants were predeclared at launch and all economic significance is Bonferroni-corrected. No real trades are placed."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":main()
