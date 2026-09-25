#!/usr/bin/env python3
from pathlib import Path
import json, math, time
import numpy as np

R=Path(__file__).resolve().parent
TICKS=R/"ticks.json"
OUT=R/"memory"/"differ_campaign_21_latest.json"
CAMPAIGN_WINS=21
MIN_HISTORY=220

def load_ticks():
    j=json.loads(TICKS.read_text())
    return np.asarray([int(x["digit"]) for x in sorted(j["ticks"],key=lambda z:int(z["epoch"]))],dtype=np.int16)

def probs_recent(hist,n):
    a=hist[-n:]
    c=np.full(10,1.5,dtype=float)
    for d in a:c[int(d)]+=1
    return c/c.sum()

def probs_trans(hist,order,lookback):
    a=hist[-lookback:]
    c=np.full(10,1.2,dtype=float)
    if len(a)<=order:return c/c.sum()
    ctx=tuple(int(x) for x in a[-order:])
    support=0
    for i in range(order,len(a)):
        if tuple(int(x) for x in a[i-order:i])==ctx:
            c[int(a[i])]+=1;support+=1
    p=c/c.sum()
    if order==3 and support<12:
        p2=probs_trans(hist,2,min(1200,lookback))
        w=max(.08,min(.65,support/12))
        p=w*p+(1-w)*p2
    return p

def features(digits):
    rows=[]
    for i in range(MIN_HISTORY,len(digits)):
        hist=digits[:i]
        recent20=probs_recent(hist,20)
        recent40=probs_recent(hist,40)
        recent80=probs_recent(hist,80)
        t1=probs_trans(hist,1,1000)
        t2=probs_trans(hist,2,1200)
        t3=probs_trans(hist,3,1500)
        tail20=[int(x) for x in hist[-20:]]
        repeat20=sum(1 for k in range(1,len(tail20)) if tail20[k]==tail20[k-1])/(len(tail20)-1)
        cc=np.bincount(np.asarray(tail20,dtype=np.int16),minlength=10)
        maxshare20=float(cc.max()/len(tail20))
        ages=[]
        for d in range(10):
            age=61
            for back in range(1,min(60,len(hist))+1):
                if int(hist[-back])==d:
                    age=back;break
            ages.append(age)
        rows.append({
            "target":int(digits[i]),
            "recent20":recent20,"recent40":recent40,"recent80":recent80,
            "t1":t1,"t2":t2,"t3":t3,
            "repeat20":repeat20,"maxshare20":maxshare20,
            "ages":ages,
            "last3":[int(x) for x in hist[-3:]]
        })
    return rows

def candidate_grid():
    out=[]
    weights=[
      ("balanced",(0.10,0.15,0.20,0.25,0.30)),
      ("t23",(0.05,0.10,0.10,0.30,0.45)),
      ("recent",(0.25,0.20,0.15,0.18,0.22)),
      ("t3",(0.05,0.08,0.10,0.22,0.55)),
    ]
    triggers=["any","uniform","very_uniform","no_repeat","gap3","gap5","balanced_gap"]
    for name,w in weights:
      for bottom_need in [2,3,4]:
        for recent_penalty in [0.0,.0015,.0030]:
          for age_bonus in [0.0,.00025,.0005]:
            for trigger in triggers:
              out.append({
                "id":f"{name}_b{bottom_need}_rp{recent_penalty:.4f}_ab{age_bonus:.5f}_trg-{trigger}",
                "weights":w,"bottom_need":bottom_need,
                "recent_penalty":recent_penalty,"age_bonus":age_bonus,"trigger":trigger
              })
    return out

def trigger_ok(ex,mode):
    rep=float(ex["repeat20"]);mx=float(ex["maxshare20"])
    last=int(ex["last3"][-1]);age=int(ex["ages"][last])
    if mode=="any":return True
    if mode=="uniform":return rep<=.10 and mx<=.20
    if mode=="very_uniform":return rep<=.05 and mx<=.20
    if mode=="no_repeat":return rep<=.05
    if mode=="gap3":return age>=3
    if mode=="gap5":return age>=5
    if mode=="balanced_gap":return rep<=.10 and mx<=.20 and age>=3
    return True

def choose(ex,cfg):
    src=[ex["recent40"],ex["recent80"],ex["t1"],ex["t2"],ex["t3"]]
    bottoms=[set(np.argsort(v)[:3].tolist()) for v in src]
    w=cfg["weights"]
    score=w[0]*src[0]+w[1]*src[1]+w[2]*src[2]+w[3]*src[3]+w[4]*src[4]
    vals=[]
    last3=set(ex["last3"])
    for d in range(10):
        votes=sum(int(d in z) for z in bottoms)
        if votes<cfg["bottom_need"]:continue
        s=float(score[d])
        if d in last3:s+=float(cfg["recent_penalty"])
        s-=min(40,int(ex["ages"][d]))*float(cfg["age_bonus"])
        vals.append((s,-votes,-int(ex["ages"][d]),d))
    if not vals:return None
    vals.sort()
    return int(vals[0][3])

def evaluate(cfg,examples):
    campaigns=0;completed=0
    current=0
    signals=0;wins=0
    streak=0;best_streak=0
    starts_wait=[];wait=0
    outcomes=[]
    for ex in examples:
        wait+=1
        if current==0 and not trigger_ok(ex,cfg["trigger"]):
            continue
        d=choose(ex,cfg)
        if d is None:
            continue
        if current==0:
            campaigns+=1
            starts_wait.append(wait);wait=0
        ok=int(ex["target"]!=d)
        outcomes.append(ok)
        signals+=1;wins+=ok
        if ok:
            current+=1;streak+=1;best_streak=max(best_streak,streak)
            if current>=CAMPAIGN_WINS:
                completed+=1;current=0;streak=0
        else:
            current=0;streak=0
    rate=wins/signals if signals else 0.0
    completion=completed/campaigns if campaigns else 0.0
    return {
      "campaigns":campaigns,"completed_21":completed,"completion_rate":completion,
      "signals":signals,"wins":wins,"losses":signals-wins,"hit_rate":rate,
      "longest_streak":best_streak,
      "avg_start_wait":float(np.mean(starts_wait)) if starts_wait else None,
      "max_start_wait":int(max(starts_wait)) if starts_wait else None
    }

def main():
    digits=load_ticks()
    if len(digits)<1000:raise RuntimeError("Not enough ticks")
    ex=features(digits)
    n=len(ex);a=int(n*.50);b=int(n*.75)
    discovery=ex[:a];validation=ex[a:b];holdout=ex[b:]
    grid=candidate_grid()

    ranked=[]
    for cfg in grid:
        m=evaluate(cfg,discovery)
        if m["campaigns"]<20:continue
        ranked.append((m["completion_rate"],m["completed_21"],m["longest_streak"],m["hit_rate"],cfg,m))
    ranked.sort(reverse=True,key=lambda x:(x[0],x[1],x[2],x[3]))
    top=[x[4] for x in ranked[:80]]

    val=[]
    for cfg in top:
        md=evaluate(cfg,discovery);mv=evaluate(cfg,validation)
        if mv["campaigns"]<8:continue
        stable=min(md["completion_rate"],mv["completion_rate"])
        val.append((stable,min(md["completed_21"],mv["completed_21"]),min(md["longest_streak"],mv["longest_streak"]),cfg,md,mv))
    val.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))

    results=[]
    for _,_,_,cfg,md,mv in val[:30]:
        mh=evaluate(cfg,holdout)
        results.append({
          "id":cfg["id"],"config":cfg,
          "discovery":md,"validation":mv,"holdout":mh,
          "min_completion_rate":min(mv["completion_rate"],mh["completion_rate"]),
          "min_longest_streak":min(mv["longest_streak"],mh["longest_streak"]),
          "meets_21_in_both":bool(mv["longest_streak"]>=21 and mh["longest_streak"]>=21),
          "repeated_21_both":bool(mv["completed_21"]>=2 and mh["completed_21"]>=2)
        })
    results.sort(key=lambda x:(
      x["repeated_21_both"],x["meets_21_in_both"],x["min_completion_rate"],
      min(x["validation"]["completed_21"],x["holdout"]["completed_21"]),
      x["min_longest_streak"]
    ),reverse=True)

    out={
      "version":"1.0-campaign-21-pattern-search",
      "timestamp":int(time.time()),
      "ticks":len(digits),"examples":len(ex),
      "split":{"discovery":len(discovery),"validation":len(validation),"holdout":len(holdout)},
      "campaign_wins_required":CAMPAIGN_WINS,
      "payout_assumption":1.09,
      "start_stake":1.0,
      "target_net_profit":5.0,
      "strategies_tested":len(grid),
      "leader":results[0] if results else None,
      "finalists":results,
      "status":"CANDIDATE_FOUND" if results and results[0]["repeated_21_both"] else "SEARCHING",
      "note":"Historical chronological discovery/validation/holdout search. A 21+ streak is not a guarantee; forward-only challenge remains required."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
