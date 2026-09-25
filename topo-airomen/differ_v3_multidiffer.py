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

VARIANTS=[
 {"id":"top1_m005_sd020","max_k":1,"margin":.005,"sd":.020},
 {"id":"top2_m005_sd020","max_k":2,"margin":.005,"sd":.020},
 {"id":"top3_m005_sd020","max_k":3,"margin":.005,"sd":.020},
 {"id":"top3_m010_sd015","max_k":3,"margin":.010,"sd":.015},
 {"id":"max5_m005_sd015","max_k":5,"margin":.005,"sd":.015},
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
    ts=p.get("timestamp"); age=max(0,int(time.time())-int(ts)) if ts is not None else None
    fresh=bool(be is not None and age is not None and age<=PAYOUT_MAX_AGE and len(rows)==10)
    return rows,float(be) if be is not None else None,age,fresh

def probs(model,hist):
    w=int(model["window"])
    X=model["encoder"].transform(np.asarray([hist[-w:]],dtype=np.int16))
    p=model["clf"].predict_proba(X)[0]
    full=np.ones(10,dtype=float)
    for j,c in enumerate(model["clf"].classes_):full[int(c)]=float(p[j])
    return full

def freeze(ticks):
    if not CAND.exists() or not CANDMETA.exists():raise RuntimeError("No ensemble candidate")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=loadj(FMETA,{})
    last=int(ticks[-1]["epoch"])
    st={
      "ensemble_id":meta.get("ensemble_id"),"start_epoch":last,"last_epoch":last,"runs":0,
      "history":[int(x["digit"]) for x in ticks[-1500:]],"forward_ticks":0,
      "variants":{v["id"]:{
        "traded_ticks":0,"contracts":0,"wins":0,"pnl":0.0,"staked":0.0,
        "tick_pnls":[],"contract_outcomes":[],"equity":0.0,"peak":0.0,"max_drawdown":0.0,
        "selected_sum":0,"selected_max":0
      } for v in VARIANTS}
    }
    STATE.write_text(json.dumps(st,indent=2));return st

def block_stats(vals):
    if len(vals)<300:return []
    a=np.asarray(vals,dtype=float)
    return [
      {"n":len(x),"pnl":float(x.sum()),"mean_tick_pnl":float(x.mean())}
      for x in np.array_split(a,3)
    ]

def main():
    ticks=sorted(loadj(R/"ticks.json",{}).get("ticks",[]),key=lambda x:int(x["epoch"]))
    if not ticks:raise RuntimeError("No ticks")
    pmap,be,age,fresh_payout=payout_info()
    if STATE.exists() and FROZEN.exists() and FMETA.exists():st=loadj(STATE,{})
    else:st=freeze(ticks)

    bundle=joblib.load(FROZEN); models=bundle["models"]; maxw=max(int(m["window"]) for m in models)
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[])))
    threshold=be if be is not None else .9174311926605504

    for x in fresh:
        actual=int(x["digit"]);st["forward_ticks"]=int(st.get("forward_ticks",0))+1
        if len(hist)>=max(500,maxw):
            arr=np.vstack([probs(m,hist) for m in models])
            avg=arr.mean(axis=0); sd=arr.std(axis=0)
            for cfg in VARIANTS:
                loss_limit=max(0.0,1.0-(threshold+cfg["margin"]))
                eligible=[d for d in range(10) if avg[d]<=loss_limit and sd[d]<=cfg["sd"]]
                eligible.sort(key=lambda d:(avg[d],sd[d],d))
                selected=eligible[:cfg["max_k"]]
                if not selected:continue
                s=st["variants"][cfg["id"]]
                s["traded_ticks"]+=1;s["selected_sum"]+=len(selected);s["selected_max"]=max(int(s["selected_max"]),len(selected))
                tick_pnl=0.0
                for d in selected:
                    hit=int(actual!=d);s["contracts"]+=1;s["wins"]+=hit;s["contract_outcomes"].append(hit)
                    if d in pmap:
                        ask,payout=pmap[d];delta=(payout-ask) if hit else -ask
                        tick_pnl+=delta;s["pnl"]+=delta;s["staked"]+=ask
                s["tick_pnls"].append(tick_pnl)
                if len(s["tick_pnls"])>5000:s["tick_pnls"]=s["tick_pnls"][-5000:]
                if len(s["contract_outcomes"])>20000:s["contract_outcomes"]=s["contract_outcomes"][-20000:]
                s["equity"]+=tick_pnl;s["peak"]=max(float(s["peak"]),float(s["equity"]))
                s["max_drawdown"]=max(float(s["max_drawdown"]),float(s["peak"])-float(s["equity"]))
        hist.append(actual)
        if len(hist)>1500:hist=hist[-1500:]
        st["last_epoch"]=int(x["epoch"])

    st["history"]=hist;st["runs"]=int(st.get("runs",0))+1
    STATE.write_text(json.dumps(st,indent=2))

    rows=[]
    for cfg in VARIANTS:
        s=st["variants"][cfg["id"]];c=int(s["contracts"]);w=int(s["wins"]);tt=int(s["traded_ticks"])
        bs=block_stats(s["tick_pnls"])
        stable=bool(len(bs)==3 and all(b["pnl"]>0 for b in bs))
        candidate=bool(tt>=1000 and fresh_payout and c>=1500 and wilson(w,c)>threshold and float(s["pnl"])>0 and stable)
        rows.append({
          "id":cfg["id"],"traded_ticks":tt,"contracts":c,"wins":w,"losses":c-w,
          "contract_hit_rate":float(w/c) if c else None,"wilson_lower":wilson(w,c),
          "avg_differs_per_traded_tick":float(s["selected_sum"]/tt) if tt else None,
          "max_differs_in_one_tick":int(s["selected_max"]),
          "shadow_pnl":float(s["pnl"]),"shadow_roi":float(s["pnl"])/float(s["staked"]) if s["staked"] else None,
          "max_drawdown":float(s["max_drawdown"]),"blocks":bs,"block_stable":stable,
          "status":"PORTFOLIO_CANDIDATE" if candidate else "COLLECTING"
        })
    rows.sort(key=lambda r:(r["status"]=="PORTFOLIO_CANDIDATE",r["shadow_pnl"],r["avg_differs_per_traded_tick"] or 0,r["traded_ticks"]),reverse=True)
    confirmed=[r["id"] for r in rows if r["status"]=="PORTFOLIO_CANDIDATE"]
    out={
      "version":"3.0-multi-differ-portfolio","timestamp":int(time.time()),"runs":st["runs"],
      "ensemble_id":st.get("ensemble_id"),"new_ticks_this_run":len(fresh),"forward_ticks":st.get("forward_ticks",0),
      "break_even_rate":be,"payout_fresh":fresh_payout,"payout_age_seconds":age,
      "leader":rows[0] if rows else None,"variants":rows,"confirmed_variants":confirmed,
      "status":"PORTFOLIO_CANDIDATE" if confirmed else "RESEARCHING",
      "note":"Prospective multi-DIFFER portfolio lab. It scans all 10 digits per tick and selects as many as pass payout-aware probability and model-dispersion gates. No live multi-buy promotion before forward portfolio evidence."
    }
    OUT.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=="__main__":main()
