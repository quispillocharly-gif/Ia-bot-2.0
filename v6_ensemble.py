#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil
import numpy as np
import joblib

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
CAND=M/"neural_v5_candidate.joblib"; CANDMETA=M/"neural_v5_candidate.json"
FROZEN=M/"v6_frozen_neural.joblib"; FMETA=M/"v6_frozen_neural.json"
STATE=M/"v6_ensemble_state.json"; OUT=M/"v6_ensemble_latest.json"
BASE=.10; MIN_AGREE=2

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

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

def current_status(st):
    n=int(st.get("n",0)); w=int(st.get("w",0))
    if n<1000:return "COLLECTING"
    return "CONFIRMED_EDGE" if wilson(w,n)>BASE else "NOT_CONFIRMED"

def freeze(ticks):
    if not CAND.exists() or not CANDMETA.exists():raise RuntimeError("No neural candidate")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=json.loads(FMETA.read_text()); latest=int(ticks[-1]["epoch"])
    st={"version":"6.1","model_id":meta["model_id"],"start_epoch":latest,"last_epoch":latest,
        "history":[int(x["digit"]) for x in ticks[-1200:]],"n":0,"w":0,"runs":0,
        "opportunities":0,"skipped_confidence":0,"skipped_agreement":0,
        "confidence_sum":0.0,"agreement_sum":0.0,"recent_results":[]}
    STATE.write_text(json.dumps(st,indent=2)); return st

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    if not ticks:raise RuntimeError("No ticks")
    candmeta=json.loads(CANDMETA.read_text()) if CANDMETA.exists() else None
    if STATE.exists() and FROZEN.exists() and FMETA.exists():
        st=json.loads(STATE.read_text())
        if current_status(st)=="NOT_CONFIRMED" and candmeta and candmeta.get("model_id")!=st.get("model_id"):
            st=freeze(ticks)
    else:st=freeze(ticks)

    bundle=joblib.load(FROZEN); meta=json.loads(FMETA.read_text())
    window=int(bundle["window"]); clf=bundle["clf"]; enc=bundle["encoder"]
    threshold=float(bundle.get("confidence_threshold",meta.get("confidence_threshold",0.0) or 0.0))
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[]))); recent=list(map(int,st.get("recent_results",[])))

    for x in fresh:
        d=int(x["digit"])
        if len(hist)>=max(80,window):
            st["opportunities"]+=1
            X=enc.transform(np.asarray([hist[-window:]],dtype=np.int16))
            proba=clf.predict_proba(X)[0]; j=int(np.argmax(proba))
            pred=int(clf.classes_[j]); conf=float(proba[j])
            experts=[hot(hist,20),cold(hist,20),hot(hist,40),cold(hist,40),gap(hist),trans(hist,1),trans(hist,2),trans(hist,3)]
            agree=sum(int(p==pred) for p in experts)
            if conf<threshold:st["skipped_confidence"]+=1
            elif agree<MIN_AGREE:st["skipped_agreement"]+=1
            else:
                ok=int(pred==d); st["n"]+=1; st["w"]+=ok
                st["confidence_sum"]+=conf; st["agreement_sum"]+=agree
                recent.append(ok)
                if len(recent)>300:recent=recent[-300:]
        hist.append(d)
        if len(hist)>1200:hist=hist[-1200:]
        st["last_epoch"]=int(x["epoch"])

    st["history"]=hist; st["recent_results"]=recent; st["runs"]=int(st.get("runs",0))+1
    STATE.write_text(json.dumps(st,indent=2))
    n=int(st["n"]); w=int(st["w"]); opp=int(st["opportunities"])
    rate=w/n if n else None; low=wilson(w,n); coverage=n/opp if opp else None
    recent_rate=sum(recent)/len(recent) if recent else None
    drift="COLLECTING"
    if len(recent)>=150:
        if recent_rate<.085:drift="DRIFT_ALERT"
        elif recent_rate<BASE:drift="DRIFT_WATCH"
        else:drift="STABLE"
    out={"version":"6.1-forward-ensemble","model_id":st["model_id"],"search_version":meta.get("search_version"),
         "window":window,"hidden":meta.get("hidden"),"confidence_threshold":threshold,
         "min_expert_agreement":MIN_AGREE,"runs":st["runs"],"new_ticks_this_run":len(fresh),
         "opportunities":opp,"predictions":n,"wins":w,"losses":n-w,"hit_rate":rate,
         "wilson_lower":low,"coverage":coverage,
         "skipped_confidence":int(st["skipped_confidence"]),"skipped_agreement":int(st["skipped_agreement"]),
         "mean_confidence":st["confidence_sum"]/n if n else None,
         "mean_expert_agreement":st["agreement_sum"]/n if n else None,
         "recent_predictions":len(recent),"recent_hit_rate":recent_rate,"drift_status":drift,
         "baseline":BASE,"status":current_status(st),"start_epoch":st["start_epoch"],
         "note":"Forward-only ensemble: frozen neural candidate plus independent statistical expert agreement. No historical outcomes are counted."}
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":main()
