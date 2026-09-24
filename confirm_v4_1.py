#!/usr/bin/env python3
from pathlib import Path
import json,math,time
R=Path(__file__).resolve().parent; M=R/"memory"; M.mkdir(exist_ok=True)
S=M/"v4_1_confirmation_state.json"; O=M/"v4_1_confirmation_latest.json"

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def transition3(hist):
    if len(hist)<4:return 0
    ctx=tuple(hist[-3:]); c=[0]*10
    for i in range(3,len(hist)):
        if tuple(hist[i-3:i])==ctx:c[hist[i]]+=1
    if sum(c): return max(range(10),key=lambda d:(c[d],-d))
    a=hist[-80:]; return max(range(10),key=lambda d:(a.count(d),-d))

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    if S.exists(): st=json.loads(S.read_text())
    else:
        # Freeze now: future ticks only. No historical results are counted.
        st={"version":"4.1","frozen_model":"transition_3","wait_ticks":3,
            "start_epoch":int(ticks[-1]["epoch"]) if ticks else 0,
            "last_epoch":int(ticks[-1]["epoch"]) if ticks else 0,
            "history":[int(x["digit"]) for x in ticks[-500:]],"pending":[],"n":0,"w":0,"runs":0}
    fresh=[x for x in ticks if int(x["epoch"])>st["last_epoch"]]
    hist=list(map(int,st["history"])); pending=st.get("pending",[])
    for x in fresh:
        d=int(x["digit"])
        # Resolve predictions whose 3-tick horizon has arrived.
        for q in pending: q["left"]-=1
        done=[q for q in pending if q["left"]<=0]
        pending=[q for q in pending if q["left"]>0]
        for q in done:
            st["n"]+=1; st["w"]+=int(q["pred"]==d)
        # Make a new frozen-model prediction from information available BEFORE this tick is appended.
        if len(hist)>=80:
            pending.append({"pred":transition3(hist),"left":3})
        hist.append(d)
        if len(hist)>500: hist=hist[-500:]
        st["last_epoch"]=int(x["epoch"])
    st["history"]=hist; st["pending"]=pending; st["runs"]+=1
    n=st["n"]; w=st["w"]; rate=w/n if n else None; low=wilson(w,n)
    status="COLLECTING"
    if n>=1000: status="CONFIRMED_EDGE" if low>.10 else "NOT_CONFIRMED"
    out={"version":"4.1-confirmation","frozen_model":"transition_3","wait_ticks":3,
         "start_epoch":st["start_epoch"],"runs":st["runs"],"new_ticks_this_run":len(fresh),
         "predictions":n,"wins":w,"losses":n-w,"hit_rate":rate,"wilson_lower":low,
         "baseline":.10,"status":status,
         "note":"Forward-only confirmation. The frozen model is not changed using confirmation outcomes."}
    S.write_text(json.dumps(st,indent=2)); O.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
main()
