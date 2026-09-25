#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil, time
import numpy as np
import joblib

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
CAND=M/"differ_ensemble_candidate.joblib"; CANDMETA=M/"differ_ensemble_candidate.json"
FROZEN=M/"differ_v3_frozen.joblib"; FMETA=M/"differ_v3_frozen.json"
STATE=M/"differ_v3_state.json"; OUT=M/"differ_v3_latest.json"
PAYOUT=M/"differ_payout_snapshot.json"
PAYOUT_MAX_AGE=7200

# One DIFFER barrier per tick. Goal: maximize usable ticks while avoiding the
# next digit MATCH and remaining above the real economic break-even.
VARIANTS=[
 {"id":"every_tick_best","margin":-1.0,"sd":1.0,"agree":0,"gap":0.0,"recent":1.0,"trans":1.0},
 {"id":"be_only","margin":0.0,"sd":1.0,"agree":0,"gap":0.0,"recent":1.0,"trans":1.0},
 {"id":"margin002_sd025","margin":.002,"sd":.025,"agree":0,"gap":0.0,"recent":1.0,"trans":1.0},
 {"id":"margin005_sd020","margin":.005,"sd":.020,"agree":0,"gap":0.0,"recent":1.0,"trans":1.0},
 {"id":"agree2_margin002","margin":.002,"sd":.025,"agree":2,"gap":0.0,"recent":1.0,"trans":1.0},
 {"id":"agree2_gap001","margin":.002,"sd":.025,"agree":2,"gap":.001,"recent":1.0,"trans":1.0},
 {"id":"agree2_gap001_r14","margin":.002,"sd":.025,"agree":2,"gap":.001,"recent":.14,"trans":1.0},
 {"id":"safety_full","margin":.002,"sd":.025,"agree":2,"gap":.001,"recent":.14,"trans":.16},
 {"id":"agree3_margin002","margin":.002,"sd":.025,"agree":3,"gap":0.0,"recent":1.0,"trans":1.0},
]

def loadj(path,default=None):
    try:return json.loads(path.read_text())
    except Exception:return {} if default is None else default

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def payout_info():
    p=loadj(PAYOUT,{})
    rows={}
    for x in p.get("per_digit",[]):
        if x.get("ask_price") is not None and x.get("payout") is not None:
            rows[int(x["digit"])]=(float(x["ask_price"]),float(x["payout"]))
    be=p.get("max_break_even") if p.get("status")=="OK" else None
    ts=p.get("timestamp")
    age=max(0,int(time.time())-int(ts)) if ts is not None else None
    fresh=bool(be is not None and age is not None and age<=PAYOUT_MAX_AGE and len(rows)==10)
    return rows,float(be) if be is not None else None,age,fresh

def probs(model,hist):
    w=int(model["window"])
    X=model["encoder"].transform(np.asarray([hist[-w:]],dtype=np.int16))
    p=model["clf"].predict_proba(X)[0]
    full=np.ones(10,dtype=float)
    for j,c in enumerate(model["clf"].classes_):full[int(c)]=float(p[j])
    return full

def recent_share(hist,digit,n=80):
    a=hist[-n:]
    return (sum(1 for x in a if x==digit)/len(a)) if a else .1

def transition_risk(hist,digit,n=500):
    a=hist[-n:]
    if len(a)<2:return .1
    last=a[-1]; total=0; hits=0
    for x,y in zip(a[:-1],a[1:]):
        if x==last:
            total+=1
            if y==digit:hits+=1
    # Shrink to 10% prior to avoid tiny transition samples dominating.
    return float((hits+2)/(total+20))

def freeze(ticks):
    if not CAND.exists() or not CANDMETA.exists():raise RuntimeError("No ensemble candidate")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=loadj(FMETA,{})
    last=int(ticks[-1]["epoch"])
    st={
      "ensemble_id":meta.get("ensemble_id"),"start_epoch":last,"last_epoch":last,
      "runs":0,"forward_ticks":0,
      "history":[int(x["digit"]) for x in ticks[-1500:]],
      "variants":{v["id"]:{
        "signals":0,"wins":0,"pnl":0.0,"staked":0.0,"outcomes":[],
        "equity":0.0,"peak":0.0,"max_drawdown":0.0,
        "sum_predicted_win":0.0,"sum_agreement":0.0
      } for v in VARIANTS}
    }
    STATE.write_text(json.dumps(st,indent=2));return st

def blocks(outcomes):
    if len(outcomes)<600:return []
    a=np.asarray(outcomes,dtype=np.int8)
    out=[]
    for x in np.array_split(a,3):
        n=len(x);w=int(x.sum())
        out.append({"n":n,"wins":w,"hit_rate":float(w/n),"wilson_lower":wilson(w,n)})
    return out

def main():
    ticks=sorted(loadj(R/"ticks.json",{}).get("ticks",[]),key=lambda x:int(x["epoch"]))
    if not ticks:raise RuntimeError("No ticks")
    pmap,be,age,payout_fresh=payout_info()
    if STATE.exists() and FROZEN.exists() and FMETA.exists():
        st=loadj(STATE,{})
        # Reset if state belongs to old multi-contract V3 schema.
        if "signals" not in next(iter(st.get("variants",{}).values()),{}):
            st=freeze(ticks)
    else:st=freeze(ticks)

    # Add newly declared research variants without resetting accumulated history.
    template={
      "signals":0,"wins":0,"pnl":0.0,"staked":0.0,"outcomes":[],
      "equity":0.0,"peak":0.0,"max_drawdown":0.0,
      "sum_predicted_win":0.0,"sum_agreement":0.0
    }
    for cfg in VARIANTS:
        if cfg["id"] not in st["variants"]:
            st["variants"][cfg["id"]]=dict(template)

    bundle=joblib.load(FROZEN);models=bundle["models"]
    maxw=max(int(m["window"]) for m in models)
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[])))
    threshold=be if be is not None else .9174311926605504

    for x in fresh:
        actual=int(x["digit"])
        st["forward_ticks"]=int(st.get("forward_ticks",0))+1
        if len(hist)>=max(500,maxw):
            arr=np.vstack([probs(m,hist) for m in models])
            avg=arr.mean(axis=0);sd=arr.std(axis=0)
            individual=np.argmin(arr,axis=1)
            barrier=int(np.argmin(avg))
            predicted_loss=float(avg[barrier]);predicted_win=1-predicted_loss
            agreement=int(np.sum(individual==barrier))
            ordered=np.argsort(avg)
            second=int(ordered[1]) if len(ordered)>1 else barrier
            gap=float(avg[second]-avg[barrier])
            rshare=recent_share(hist,barrier,80)
            trisk=transition_risk(hist,barrier,500)

            for cfg in VARIANTS:
                emit=True
                if cfg["margin"]>=0:
                    emit=predicted_win>=threshold+cfg["margin"]
                emit=emit and float(sd[barrier])<=cfg["sd"] and agreement>=cfg["agree"]
                emit=emit and gap>=cfg.get("gap",0.0)
                emit=emit and rshare<=cfg.get("recent",1.0)
                emit=emit and trisk<=cfg.get("trans",1.0)
                if not emit:continue

                s=st["variants"][cfg["id"]]
                hit=int(actual!=barrier)  # 1 = avoided the next-digit MATCH
                s["signals"]+=1;s["wins"]+=hit;s["outcomes"].append(hit)
                s["sum_predicted_win"]+=predicted_win;s["sum_agreement"]+=agreement
                if len(s["outcomes"])>10000:s["outcomes"]=s["outcomes"][-10000:]
                if barrier in pmap:
                    ask,payout=pmap[barrier]
                    delta=(payout-ask) if hit else -ask
                    s["pnl"]+=delta;s["staked"]+=ask
                    s["equity"]+=delta;s["peak"]=max(float(s["peak"]),float(s["equity"]))
                    s["max_drawdown"]=max(float(s["max_drawdown"]),float(s["peak"])-float(s["equity"]))

        hist.append(actual)
        if len(hist)>1500:hist=hist[-1500:]
        st["last_epoch"]=int(x["epoch"])

    st["history"]=hist;st["runs"]=int(st.get("runs",0))+1
    STATE.write_text(json.dumps(st,indent=2))

    rows=[];ft=int(st.get("forward_ticks",0))
    for cfg in VARIANTS:
        s=st["variants"][cfg["id"]];n=int(s["signals"]);w=int(s["wins"])
        rate=w/n if n else None;bs=blocks(s["outcomes"])
        stable=bool(len(bs)==3 and all(b["hit_rate"]>threshold for b in bs))
        confirmed=bool(
          n>=2000 and payout_fresh and wilson(w,n)>threshold and
          float(s["pnl"])>0 and stable
        )
        coverage=n/ft if ft else 0.0
        rows.append({
          "id":cfg["id"],"signals":n,"wins":w,"matches":n-w,
          "hit_rate":rate,"wilson_lower":wilson(w,n),
          "coverage":coverage,"signals_per_1000_ticks":coverage*1000,
          "matches_avoided_per_1000_ticks":(w/ft*1000) if ft else 0.0,
          "avg_predicted_win":float(s["sum_predicted_win"]/n) if n else None,
          "avg_model_agreement":float(s["sum_agreement"]/n) if n else None,
          "safety_gate":{"min_gap":cfg.get("gap",0.0),"max_recent_share":cfg.get("recent",1.0),"max_transition_risk":cfg.get("trans",1.0)},
          "shadow_pnl":float(s["pnl"]),
          "shadow_roi":float(s["pnl"])/float(s["staked"]) if s["staked"] else None,
          "max_drawdown":float(s["max_drawdown"]),
          "temporal_blocks":bs,"block_stable":stable,
          "status":"AVOID_MATCH_CANDIDATE" if confirmed else ("COLLECTING" if n<2000 else "NOT_CONFIRMED")
        })

    # Throughput matters only after economics. Never promote a high-frequency negative rule.
    rows.sort(
      key=lambda r:(
        r["status"]=="AVOID_MATCH_CANDIDATE",
        r["wilson_lower"]>threshold and (r["shadow_pnl"] or 0)>0,
        r["matches_avoided_per_1000_ticks"],
        r["wilson_lower"],
        r["shadow_pnl"]
      ),reverse=True
    )
    confirmed=[r["id"] for r in rows if r["status"]=="AVOID_MATCH_CANDIDATE"]
    out={
      "version":"3.2-next-digit-avoid-match-safety","timestamp":int(time.time()),"runs":st["runs"],
      "ensemble_id":st.get("ensemble_id"),"new_ticks_this_run":len(fresh),
      "forward_ticks":ft,"break_even_rate":be,"payout_fresh":payout_fresh,
      "payout_age_seconds":age,"leader":rows[0] if rows else None,
      "variants":rows,"confirmed_variants":confirmed,
      "status":"AVOID_MATCH_CANDIDATE" if confirmed else "RESEARCHING",
      "objective":"Choose exactly one DIFFER digit per usable tick: the digit with the lowest predicted probability of being the next digit. Maximize avoided MATCH events per 1000 ticks, subject to positive forward economics.",
      "note":"every_tick_best is a frequency benchmark only. Live promotion requires Wilson above break-even, positive P&L and temporal stability."
    }
    OUT.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=="__main__":main()
