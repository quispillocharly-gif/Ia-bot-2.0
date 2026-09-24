#!/usr/bin/env python3
from pathlib import Path
import json, math, time
R=Path(__file__).resolve().parent; M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"research_v4_latest.json"; STATE=M/"research_v4_state.json"
BASE=.10

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def pred_freq(h,w):
    a=h[-w:]; c=[a.count(d) for d in range(10)]
    return max(range(10),key=lambda d:(c[d],-d))
def pred_cold(h,w):
    a=h[-w:]; c=[a.count(d) for d in range(10)]
    return min(range(10),key=lambda d:(c[d],d))
def pred_gap(h):
    gaps=[]
    for d in range(10):
        try:g=len(h)-1-max(i for i,x in enumerate(h) if x==d)
        except:g=len(h)+1
        gaps.append(g)
    return max(range(10),key=lambda d:(gaps[d],-d))
def pred_trans(h,order):
    if len(h)<=order:return pred_freq(h,80)
    ctx=tuple(h[-order:]); c=[0]*10
    for i in range(order,len(h)):
        if tuple(h[i-order:i])==ctx:c[h[i]]+=1
    return max(range(10),key=lambda d:(c[d],-d)) if sum(c) else pred_freq(h,80)

MODELS={}
for w in (10,20,40,80,120,200):
    MODELS[f"hot_{w}"]=lambda h,w=w:pred_freq(h,w)
    MODELS[f"cold_{w}"]=lambda h,w=w:pred_cold(h,w)
MODELS["gap"]=pred_gap
for o in (1,2,3): MODELS[f"transition_{o}"]=lambda h,o=o:pred_trans(h,o)

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    ds=[int(x["digit"]) for x in ticks]
    # Strict chronological holdout: first 60% context/training, last 40% validation.
    cut=max(250,int(len(ds)*.60)); rows=[]
    for name,fn in MODELS.items():
      for wait in (0,1,2,3):
        n=w=0
        for i in range(cut,len(ds)-wait):
            h=ds[:i]
            p=fn(h); actual=ds[i+wait]
            n+=1; w+=int(p==actual)
        rate=w/n if n else 0; low=wilson(w,n)
        rows.append({"model":name,"wait_ticks":wait,"predictions":n,"wins":w,
                     "hit_rate":rate,"wilson_lower":low,"edge_vs_10pct":rate-BASE})
    rows.sort(key=lambda x:(x["wilson_lower"],x["predictions"]),reverse=True)
    best=rows[0] if rows else None
    status="NO_EDGE_DEMONSTRATED"
    if best and best["predictions"]>=1000 and best["wilson_lower"]>BASE: status="EDGE_CANDIDATE"
    old=json.loads(STATE.read_text()) if STATE.exists() else {"runs":0}
    old["runs"]+=1; old["last_epoch"]=int(ticks[-1]["epoch"]) if ticks else 0
    STATE.write_text(json.dumps(old,indent=2))
    out={"version":"4.0-research-lab","timestamp":int(time.time()),"runs":old["runs"],
         "ticks":len(ds),"holdout_start":cut,"baseline":BASE,"status":status,
         "leader":best,"top_models":rows[:20],
         "note":"Models are ranked on chronological holdout data. No accuracy target is guaranteed."}
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
main()
