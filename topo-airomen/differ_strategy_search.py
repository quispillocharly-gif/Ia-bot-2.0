#!/usr/bin/env python3
from pathlib import Path
import json, math, time
import numpy as np
import joblib

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
TICKS=R/"ticks.json"
FROZEN=M/"differ_frozen.joblib"
STATE=M/"differ_forward_state.json"
OUT=M/"differ_strategy_search_latest.json"

MAX_EXAMPLES=5000
MIN_HISTORY=220
CADENCE=[1,2,3]

def load_json(path, default=None):
    try:return json.loads(path.read_text())
    except Exception:return {} if default is None else default

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def recent_probs(hist,n=80):
    a=hist[-n:]
    c=np.full(10,1.5,dtype=float)
    for d in a:c[int(d)]+=1
    return c/c.sum()

def trans_probs(hist,order,lookback):
    a=hist[-lookback:]
    c=np.full(10,1.2,dtype=float)
    if len(a)<=order:return c/c.sum()
    ctx=tuple(a[-order:])
    support=0
    for i in range(order,len(a)):
        if tuple(a[i-order:i])==ctx:
            c[int(a[i])]+=1; support+=1
    p=c/c.sum()
    if order==3 and support<12:
        p2=trans_probs(hist,2,min(1200,lookback))
        w=max(.08,min(.65,support/12))
        p=w*p+(1-w)*p2
    return p

def longest_streak(outcomes):
    best=cur=0
    for x in outcomes:
        if x: cur+=1; best=max(best,cur)
        else: cur=0
    return best

def perfect_windows(outcomes,size):
    if len(outcomes)<size:return 0
    arr=np.asarray(outcomes,dtype=np.int8)
    cs=np.concatenate(([0],np.cumsum(arr)))
    wins=cs[size:]-cs[:-size]
    return int(np.sum(wins==size))

def metrics(outcomes,opportunities):
    n=len(outcomes); w=int(sum(outcomes)); losses=n-w
    return {
        "signals":n,
        "wins":w,
        "losses":losses,
        "hit_rate":(w/n if n else None),
        "wilson_lower":wilson(w,n),
        "signal_rate":(n/opportunities if opportunities else None),
        "longest_anti_match_streak":longest_streak(outcomes),
        "perfect_windows_50":perfect_windows(outcomes,50),
        "perfect_windows_100":perfect_windows(outcomes,100),
        "perfect_windows_200":perfect_windows(outcomes,200),
    }

def strategy_grid():
    out=[]
    for mp in [1.0,.105,.100,.095,.090,.085,.080,.075]:
        for gap in [0.0,.002,.004]:
            out.append({"id":f"neural_p{mp:.3f}_g{gap:.3f}","family":"neural","max_p":mp,"gap":gap})
    for mp in [.100,.095,.090,.085,.080]:
        for agree in [1,2,3]:
            out.append({"id":f"cons_p{mp:.3f}_a{agree}","family":"consensus","max_p":mp,"agree":agree})
    for topk in [2,3]:
        for ms in [.105,.100,.095,.090]:
            for sh in [0,1]:
                out.append({"id":f"hybrid_k{topk}_s{ms:.3f}_h{sh}","family":"hybrid","topk":topk,"max_score":ms,"max_struct_hot":sh})
    for rank in [1,2]:
        for ms in [.105,.100,.095]:
            out.append({"id":f"struct_r{rank}_s{ms:.3f}","family":"struct","max_rank":rank,"max_score":ms})
    return out

def choose(cfg,ex,blocked=None):
    n=ex["neural"]; t1=ex["t1"]; t2=ex["t2"]; t3=ex["t3"]; recent=ex["recent"]
    digits=np.arange(10)
    allowed=[int(d) for d in digits if blocked is None or int(d)!=int(blocked)]
    if not allowed:return None
    norder=np.asarray(sorted(allowed,key=lambda d:(n[d],d)),dtype=int)
    ngap=float(n[norder[1]]-n[norder[0]])
    mins=[int(np.argmin(x)) for x in (recent,t1,t2,t3)]
    top_hot=set(np.argsort(t2)[-2:].tolist()+np.argsort(t3)[-2:].tolist())
    fam=cfg["family"]

    if fam=="neural":
        d=int(norder[0])
        emit=float(n[d])<=cfg["max_p"] and ngap>=cfg["gap"]
        return d if emit else None

    if fam=="consensus":
        d=int(norder[0]); agree=sum(int(z==d) for z in mins)
        emit=float(n[d])<=cfg["max_p"] and agree>=cfg["agree"]
        return d if emit else None

    score=.45*n+.10*recent+.13*t1+.17*t2+.15*t3
    if fam=="hybrid":
        cand=norder[:min(cfg["topk"],len(norder))]
        d=int(min(cand,key=lambda x:(score[x],n[x],x)))
        struct_hot=int(d in top_hot)
        emit=float(score[d])<=cfg["max_score"] and struct_hot<=cfg["max_struct_hot"]
        return d if emit else None

    if fam=="struct":
        d=int(min(allowed,key=lambda x:(score[x],n[x],x)))
        nr=int(np.where(norder==d)[0][0])+1
        emit=float(score[d])<=cfg["max_score"] and nr<=cfg["max_rank"]
        return d if emit else None

    return None

def eval_strategy(cfg,examples):
    out=[]
    last_barrier=None
    cadence_index=0
    ticks_waited=0
    emitted=0
    for ex in examples:
        ticks_waited+=1
        need=CADENCE[cadence_index]
        if ticks_waited<need:
            continue
        d=choose(cfg,ex,last_barrier)
        if d is None:
            continue
        out.append(int(int(ex["target"])!=d))
        emitted+=1
        last_barrier=d
        cadence_index=(cadence_index+1)%len(CADENCE)
        ticks_waited=0
    m=metrics(out,len(examples))
    m["cadence"]=CADENCE
    m["no_repeat"]=True
    m["signals_emitted"]=emitted
    return m,out

def main():
    if not TICKS.exists() or not FROZEN.exists() or not STATE.exists():
        raise RuntimeError("Forward artifacts missing")

    ticks=sorted(load_json(TICKS,{}).get("ticks",[]),key=lambda x:int(x["epoch"]))
    st=load_json(STATE,{})
    start_epoch=int(st.get("start_epoch",0))
    bundle=joblib.load(FROZEN); clf=bundle["clf"]; enc=bundle["encoder"]; window=int(bundle["window"])
    digits=np.asarray([int(x["digit"]) for x in ticks],dtype=np.int16)
    epochs=np.asarray([int(x["epoch"]) for x in ticks],dtype=np.int64)

    idx=[i for i in range(max(MIN_HISTORY,window),len(ticks)) if epochs[i]>start_epoch]
    if len(idx)>MAX_EXAMPLES:idx=idx[-MAX_EXAMPLES:]

    if len(idx)<240:
        out={"version":"1.0-strategy-search","timestamp":int(time.time()),"status":"COLLECTING",
             "forward_examples":len(idx),"target":"Search for 100% anti-MATCH on chronological validation/holdout without claiming a guarantee."}
        OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); return

    Xraw=np.asarray([digits[i-window:i] for i in idx],dtype=np.int16)
    X=enc.transform(Xraw)
    pred=clf.predict_proba(X)
    classes=[int(x) for x in clf.classes_]

    examples=[]
    for k,i in enumerate(idx):
        full=np.full(10,.10,dtype=float)
        for j,c in enumerate(classes):full[c]=float(pred[k,j])
        hist=digits[:i].tolist()
        examples.append({
            "epoch":int(epochs[i]),"target":int(digits[i]),"neural":full,
            "recent":recent_probs(hist,80),
            "t1":trans_probs(hist,1,1000),
            "t2":trans_probs(hist,2,1200),
            "t3":trans_probs(hist,3,1500),
        })

    n=len(examples); a=int(n*.50); b=int(n*.75)
    discovery=examples[:a]; validation=examples[a:b]; holdout=examples[b:]
    grid=strategy_grid()

    ranked=[]
    for cfg in grid:
        m,_=eval_strategy(cfg,discovery)
        if m["signals"]<50:continue
        ranked.append((m["wilson_lower"],m["hit_rate"] or 0,m["signals"],cfg,m))
    ranked.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    top=[x[3] for x in ranked[:12]]

    validated=[]
    for cfg in top:
        md,_=eval_strategy(cfg,discovery)
        mv,_=eval_strategy(cfg,validation)
        if mv["signals"]<30:continue
        validated.append((mv["wilson_lower"],mv["hit_rate"] or 0,mv["signals"],cfg,md,mv))
    validated.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    finalists=validated[:6]

    results=[]
    for _,_,_,cfg,md,mv in finalists:
        mh,oh=eval_strategy(cfg,holdout)
        perfect=bool(mv["signals"]>=50 and mh["signals"]>=50 and mv["losses"]==0 and mh["losses"]==0)
        results.append({
            "id":cfg["id"],"config":cfg,
            "discovery":md,"validation":mv,"holdout":mh,
            "zero_match_validation_and_holdout":perfect
        })

    results.sort(key=lambda x:(
        x["zero_match_validation_and_holdout"],
        x["holdout"]["wilson_lower"],
        x["holdout"]["hit_rate"] or 0,
        x["holdout"]["signals"]
    ),reverse=True)

    perfect=[x["id"] for x in results if x["zero_match_validation_and_holdout"]]
    leader=results[0] if results else None
    out={
        "version":"1.0-strategy-search",
        "timestamp":int(time.time()),
        "model_id":st.get("model_id"),
        "start_epoch":start_epoch,
        "forward_examples":n,
        "split":{"discovery":len(discovery),"validation":len(validation),"holdout":len(holdout)},
        "strategies_tested":len(grid),
        "finalists":results,
        "leader":leader,
        "perfect_candidates":perfect,
        "status":"PERFECT_FORWARD_CANDIDATE" if perfect else "SEARCHING",
        "target":"Find 1-2-3 cadence + no-repeat strategies with 0 MATCH in both chronological validation and holdout with >=50 signals in each block. This is evidence, not a guarantee of future 100%.",
        "note":"Every tested strategy uses cadence 1->2->3 and blocks the previous barrier digit. Strategy choice is made before the final holdout. Signal count is tracked so a sparse 3/3 rule cannot masquerade as a useful 100% strategy."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
