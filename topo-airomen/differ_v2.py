#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil, time
import numpy as np
import joblib
from scipy.stats import binomtest

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
CAND=M/"differ_ensemble_candidate.joblib"; CANDMETA=M/"differ_ensemble_candidate.json"
FROZEN=M/"differ_v2_frozen.joblib"; FMETA=M/"differ_v2_frozen.json"
STATE=M/"differ_v2_state.json"; OUT=M/"differ_v2_latest.json"
PAYOUT=M/"differ_payout_snapshot.json"; GRAVE=M/"differ_v2_graveyard.json"
PAYOUT_MAX_AGE=7200; BASE=.90
VARIANTS=[
 {"id":"vote2_margin005","agree":2,"margin":.005,"cal":False,"stable":False},
 {"id":"vote2_margin010","agree":2,"margin":.010,"cal":False,"stable":False},
 {"id":"vote3_margin005","agree":3,"margin":.005,"cal":False,"stable":False},
 {"id":"vote2_calibrated","agree":2,"margin":.002,"cal":True,"stable":False},
 {"id":"vote3_calibrated","agree":3,"margin":.002,"cal":True,"stable":False},
 {"id":"stable_vote2_calibrated","agree":2,"margin":.002,"cal":True,"stable":True},
]
TESTS=len(VARIANTS)

def loadj(path,default=None):
    try:return json.loads(path.read_text())
    except Exception:return {} if default is None else default

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def cal_upper(rows,p):
    for r in rows or []:
        if float(r["lo"])<=p<float(r["hi"]):
            return float(r.get("upper95_loss",1.0))
    return 1.0

def payout_info():
    p=loadj(PAYOUT,{})
    rows={}
    for x in p.get("per_digit",[]):
        if x.get("ask_price") is not None and x.get("payout") is not None:
            rows[int(x["digit"])]=(float(x["ask_price"]),float(x["payout"]))
    be=p.get("max_break_even") if p.get("status")=="OK" else None
    ts=p.get("timestamp"); age=max(0,int(time.time())-int(ts)) if ts is not None else None
    fresh=bool(be is not None and age is not None and age<=PAYOUT_MAX_AGE and len(rows)==10)
    return rows,float(be) if be is not None else None,age,fresh

def entropy_norm(h,n=200):
    a=h[-n:]
    if not a:return 1.0
    c=np.bincount(np.asarray(a,dtype=np.int16),minlength=10).astype(float)
    p=c/c.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/math.log(10))

def repeat_rate(h,n=200):
    a=h[-n:]
    return 0.0 if len(a)<2 else float(sum(a[i]==a[i-1] for i in range(1,len(a)))/(len(a)-1))

def transition_peak(h,n=500):
    a=h[-n:]
    if len(a)<20:return .1
    mat=np.zeros((10,10),dtype=np.int32)
    for x,y in zip(a[:-1],a[1:]):mat[x,y]+=1
    vals=[float(r.max()/r.sum()) for r in mat if r.sum()]
    return float(np.mean(vals)) if vals else .1

def regime(h):
    en=entropy_norm(h); rep=repeat_rate(h); tp=transition_peak(h)
    return "CHAOTIC" if rep>.13 or tp>.18 or en<.955 else "STABLE"

def freeze(ticks):
    if not CAND.exists() or not CANDMETA.exists():raise RuntimeError("No V2 ensemble candidate")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=loadj(FMETA,{})
    last=int(ticks[-1]["epoch"])
    st={
      "ensemble_id":meta.get("ensemble_id"),"start_epoch":last,"last_epoch":last,"runs":0,
      "history":[int(x["digit"]) for x in ticks[-1500:]],"forward_ticks":0,
      "variants":{v["id"]:{
        "n":0,"w":0,"pnl":0.0,"staked":0.0,"outcomes":[],"equity":0.0,"peak":0.0,"max_drawdown":0.0
      } for v in VARIANTS}
    }
    STATE.write_text(json.dumps(st,indent=2)); return st

def full_probs(model,hist):
    w=int(model["window"])
    X=model["encoder"].transform(np.asarray([hist[-w:]],dtype=np.int16))
    p=model["clf"].predict_proba(X)[0]
    full=np.ones(10,dtype=float)
    for j,c in enumerate(model["clf"].classes_):full[int(c)]=float(p[j])
    return full

def blocks(outcomes):
    if len(outcomes)<600:return []
    arr=np.asarray(outcomes,dtype=np.int8)
    out=[]
    for p in np.array_split(arr,3):
        n=len(p); w=int(p.sum())
        out.append({"n":n,"wins":w,"hit_rate":float(w/n),"wilson_lower":wilson(w,n)})
    return out

def main():
    ticks=sorted(loadj(R/"ticks.json",{}).get("ticks",[]),key=lambda x:int(x["epoch"]))
    if not ticks:raise RuntimeError("No ticks")
    pmap,be,payout_age,payout_fresh=payout_info()
    candmeta=loadj(CANDMETA,{})
    prev=loadj(OUT,{})
    if STATE.exists() and FROZEN.exists() and FMETA.exists():
        st=loadj(STATE,{})
        prev_leader=prev.get("leader") or {}
        failed=bool(prev_leader.get("status")=="NOT_CONFIRMED" and not prev.get("confirmed_variants"))
        new_candidate=bool(candmeta.get("ensemble_id") and candmeta.get("ensemble_id")!=st.get("ensemble_id"))
        if failed and new_candidate:
            g=loadj(GRAVE,{"entries":[]})
            g.setdefault("entries",[]).append({
              "ensemble_id":st.get("ensemble_id"),"failed_at":int(time.time()),
              "leader_id":prev_leader.get("id"),"signals":prev_leader.get("signals"),
              "hit_rate":prev_leader.get("hit_rate"),"break_even_rate":prev.get("break_even_rate"),
              "reason":"V2_FORWARD_NOT_CONFIRMED"
            })
            g["entries"]=g["entries"][-60:]; GRAVE.write_text(json.dumps(g,indent=2))
            st=freeze(ticks)
    else: st=freeze(ticks)

    bundle=joblib.load(FROZEN); models=bundle["models"]
    maxw=max(int(m["window"]) for m in models)
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[])))

    for x in fresh:
        d=int(x["digit"]); st["forward_ticks"]=int(st.get("forward_ticks",0))+1
        if len(hist)>=max(500,maxw):
            fulls=[]; votes=[]; cups=[]
            for m in models:
                f=full_probs(m,hist); b=int(np.argmin(f)); pm=float(f[b])
                fulls.append(f); votes.append(b); cups.append(cal_upper(m.get("calibration",[]),pm))
            vc=np.bincount(np.asarray(votes,dtype=np.int16),minlength=10)
            top=int(np.argmax(vc)); agree=int(vc[top])
            avg_loss=float(np.mean([f[top] for f in fulls]))
            agree_cals=[cups[i] for i,b in enumerate(votes) if b==top]
            conservative_loss=max(avg_loss,max(agree_cals) if agree_cals else 1.0)
            predicted_win=1-avg_loss; conservative_win=1-conservative_loss
            reg=regime(hist); threshold=be if be is not None else BASE
            for cfg in VARIANTS:
                score=conservative_win if cfg["cal"] else predicted_win
                emit=agree>=cfg["agree"] and score>=threshold+cfg["margin"]
                if cfg["stable"]:emit=emit and reg=="STABLE"
                if not emit:continue
                hit=int(d!=top); s=st["variants"][cfg["id"]]
                s["n"]+=1; s["w"]+=hit; s["outcomes"].append(hit)
                if len(s["outcomes"])>5000:s["outcomes"]=s["outcomes"][-5000:]
                if top in pmap:
                    ask,payout=pmap[top]; delta=(payout-ask) if hit else -ask
                    s["pnl"]+=delta; s["staked"]+=ask
                    s["equity"]+=delta; s["peak"]=max(float(s["peak"]),float(s["equity"]))
                    s["max_drawdown"]=max(float(s["max_drawdown"]),float(s["peak"])-float(s["equity"]))
        hist.append(d)
        if len(hist)>1500:hist=hist[-1500:]
        st["last_epoch"]=int(x["epoch"])

    st["history"]=hist; st["runs"]=int(st.get("runs",0))+1
    STATE.write_text(json.dumps(st,indent=2))
    threshold=be if be is not None else BASE
    rows=[]
    for cfg in VARIANTS:
        s=st["variants"][cfg["id"]]; n=int(s["n"]); w=int(s["w"]); rate=w/n if n else None
        raw=float(binomtest(w,n,p=threshold,alternative="greater").pvalue) if n else 1.0
        adj=min(1.0,raw*TESTS); bs=blocks(s["outcomes"])
        stable=bool(len(bs)==3 and all(b["hit_rate"]>threshold for b in bs))
        confirmed=bool(n>=2000 and payout_fresh and wilson(w,n)>threshold and float(s["pnl"])>0 and adj<.05 and stable)
        rows.append({
          "id":cfg["id"],"signals":n,"wins":w,"losses":n-w,"hit_rate":rate,
          "wilson_lower":wilson(w,n),"raw_economic_p":raw,"bonferroni_p":adj,
          "shadow_pnl":float(s["pnl"]),"shadow_roi":float(s["pnl"])/float(s["staked"]) if s["staked"] else None,
          "max_drawdown_flat_stake":float(s["max_drawdown"]),"temporal_blocks":bs,"block_stable":stable,
          "status":"ECONOMIC_CANDIDATE" if confirmed else ("COLLECTING" if n<2000 else "NOT_CONFIRMED")
        })
    rows.sort(key=lambda r:(r["status"]=="ECONOMIC_CANDIDATE",r["wilson_lower"],r["shadow_pnl"],r["signals"]),reverse=True)
    confirmed=[r["id"] for r in rows if r["status"]=="ECONOMIC_CANDIDATE"]
    out={
      "version":"2.0-economic-ensemble-forward","timestamp":int(time.time()),"runs":st["runs"],
      "ensemble_id":st.get("ensemble_id"),"new_ticks_this_run":len(fresh),"forward_ticks":st.get("forward_ticks",0),
      "tests":TESTS,"break_even_rate":be,"payout_fresh":payout_fresh,"payout_age_seconds":payout_age,
      "leader":rows[0] if rows else None,"variants":rows,"confirmed_variants":confirmed,
      "graveyard_size":len(loadj(GRAVE,{"entries":[]}).get("entries",[])),
      "status":"EDGE_CANDIDATE" if confirmed else "RESEARCHING",
      "note":"Prospective-only V2 ensemble. Uses model agreement, payout-aware margins, conservative validation calibration, Bonferroni correction and temporal stability."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":main()
