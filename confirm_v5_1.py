#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil, time
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
    st={"version":"5.1","model_id":meta["model_id"],"start_epoch":latest,"last_epoch":latest,
        "history":[int(x["digit"]) for x in ticks[-500:]],"n":0,"w":0,"runs":0}
    STATE.write_text(json.dumps(st,indent=2))
    return st

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    if not ticks: raise RuntimeError("No ticks")
    candmeta=json.loads(CANDMETA.read_text()) if CANDMETA.exists() else None
    if STATE.exists() and FROZEN.exists() and FMETA.exists():
        st=json.loads(STATE.read_text())
        current_status="COLLECTING"
        if st["n"]>=1000:
            current_status="CONFIRMED_EDGE" if wilson(st["w"],st["n"])>BASE else "NOT_CONFIRMED"
        if current_status=="NOT_CONFIRMED" and candmeta and candmeta.get("model_id")!=st.get("model_id"):
            st=freeze_candidate(ticks)
    else:
        st=freeze_candidate(ticks)
    bundle=joblib.load(FROZEN)
    window=int(bundle["window"]); clf=bundle["clf"]; enc=bundle["encoder"]
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[])))
    for x in fresh:
        d=int(x["digit"])
        if len(hist)>=window:
            raw=np.asarray([hist[-window:]],dtype=np.int16)
            X=enc.transform(raw)
            p=int(clf.predict(X)[0])
            st["n"]+=1; st["w"]+=int(p==d)
        hist.append(d)
        if len(hist)>500: hist=hist[-500:]
        st["last_epoch"]=int(x["epoch"])
    st["history"]=hist; st["runs"]+=1
    STATE.write_text(json.dumps(st,indent=2))
    n=st["n"]; w=st["w"]; rate=w/n if n else None; low=wilson(w,n)
    status="COLLECTING"
    if n>=1000: status="CONFIRMED_EDGE" if low>BASE else "NOT_CONFIRMED"
    meta=json.loads(FMETA.read_text())
    out={"version":"5.1-neural-confirmation","model_id":st["model_id"],"window":window,
         "hidden":meta.get("hidden"),"runs":st["runs"],"new_ticks_this_run":len(fresh),
         "predictions":n,"wins":w,"losses":n-w,"hit_rate":rate,"wilson_lower":low,
         "baseline":BASE,"status":status,"start_epoch":st["start_epoch"],
         "note":"Frozen neural model. Confirmation uses only ticks that arrived after the model was frozen."}
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
