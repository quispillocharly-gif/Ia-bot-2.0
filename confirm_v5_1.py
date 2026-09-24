#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil
import numpy as np
import joblib

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
CAND=M/"neural_v5_candidate.joblib"
CANDMETA=M/"neural_v5_candidate.json"
FROZEN=M/"neural_v5_frozen.joblib"
FMETA=M/"neural_v5_frozen.json"
STATE=M/"v5_1_confirmation_state.json"
OUT=M/"v5_1_confirmation_latest.json"
BASE=.10

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def freeze_candidate(ticks):
    if not CAND.exists() or not CANDMETA.exists():
        raise RuntimeError("Neural candidate not available yet")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=json.loads(FMETA.read_text())
    latest=int(ticks[-1]["epoch"]) if ticks else 0
    st={
        "version":"5.2","model_id":meta["model_id"],
        "start_epoch":latest,"last_epoch":latest,
        "history":[int(x["digit"]) for x in ticks[-600:]],
        "n":0,"w":0,"runs":0,"opportunities":0,"skipped":0,"confidence_sum":0.0
    }
    STATE.write_text(json.dumps(st,indent=2))
    return st

def current_status(st):
    n=int(st.get("n",0)); w=int(st.get("w",0))
    if n<1000:return "COLLECTING"
    return "CONFIRMED_EDGE" if wilson(w,n)>BASE else "NOT_CONFIRMED"

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    if not ticks: raise RuntimeError("No ticks")
    candmeta=json.loads(CANDMETA.read_text()) if CANDMETA.exists() else None

    if STATE.exists() and FROZEN.exists() and FMETA.exists():
        st=json.loads(STATE.read_text())
        if current_status(st)=="NOT_CONFIRMED" and candmeta and candmeta.get("model_id")!=st.get("model_id"):
            st=freeze_candidate(ticks)
    else:
        st=freeze_candidate(ticks)

    bundle=joblib.load(FROZEN)
    meta=json.loads(FMETA.read_text())
    window=int(bundle["window"])
    clf=bundle["clf"]; enc=bundle["encoder"]
    threshold=float(bundle.get("confidence_threshold",meta.get("confidence_threshold",0.0) or 0.0))

    st.setdefault("opportunities",int(st.get("n",0)))
    st.setdefault("skipped",0)
    st.setdefault("confidence_sum",0.0)

    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[])))

    for x in fresh:
        d=int(x["digit"])
        if len(hist)>=window:
            st["opportunities"]+=1
            raw=np.asarray([hist[-window:]],dtype=np.int16)
            X=enc.transform(raw)
            proba=clf.predict_proba(X)[0]
            j=int(np.argmax(proba))
            pred=int(clf.classes_[j])
            conf=float(proba[j])
            if conf>=threshold:
                st["n"]+=1
                st["w"]+=int(pred==d)
                st["confidence_sum"]+=conf
            else:
                st["skipped"]+=1
        hist.append(d)
        if len(hist)>600:hist=hist[-600:]
        st["last_epoch"]=int(x["epoch"])

    st["history"]=hist
    st["runs"]=int(st.get("runs",0))+1
    STATE.write_text(json.dumps(st,indent=2))

    n=int(st["n"]); w=int(st["w"])
    opp=int(st["opportunities"]); skipped=int(st["skipped"])
    rate=w/n if n else None
    low=wilson(w,n)
    coverage=n/opp if opp else None
    mean_conf=st["confidence_sum"]/n if n else None
    status=current_status(st)

    out={
        "version":"5.2-neural-forward-confirmation",
        "model_id":st["model_id"],"search_version":meta.get("search_version"),
        "window":window,"hidden":meta.get("hidden"),
        "confidence_threshold":threshold,
        "runs":st["runs"],"new_ticks_this_run":len(fresh),
        "opportunities":opp,"predictions":n,"skipped":skipped,"coverage":coverage,
        "wins":w,"losses":n-w,"hit_rate":rate,"wilson_lower":low,
        "mean_confidence":mean_conf,"baseline":BASE,"status":status,
        "start_epoch":st["start_epoch"],
        "note":"Frozen neural candidate. Only signals above its preselected confidence threshold are counted. Evaluation uses only future ticks."
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
