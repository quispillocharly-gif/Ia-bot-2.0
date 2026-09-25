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
MIN_SIGNAL_RATE=.30
MAX_IDLE_TICKS=18
CADENCE_MODES=['ctx3','ctx4','ctx3_nr','ctx4_nr','ctx5_nr']

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

def cadence_wait(mode,ex,step,blocked,prev_wait):
    tail=[int(x) for x in ex.get("tail",[])]
    max_wait=5 if mode.startswith("ctx5") else (4 if mode.startswith("ctx4") else 3)
    h=17+int(step)*31+(0 if blocked is None else int(blocked)*13)
    for i,d in enumerate(tail):
        h=(h*33+d*(i+3)+7)%9973
    wait=1+(h%max_wait)
    if mode.endswith("_nr") and prev_wait and wait==prev_wait:
        wait=1+(wait%max_wait)
    return int(wait)

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
    first_match=None
    runs=[]; cur=0
    for i,x in enumerate(outcomes):
        if x:
            cur+=1
        else:
            if first_match is None:first_match=i+1
            runs.append(cur); cur=0
    runs.append(cur)
    return {
        "signals":n,
        "wins":w,
        "losses":losses,
        "hit_rate":(w/n if n else None),
        "wilson_lower":wilson(w,n),
        "signal_rate":(n/opportunities if opportunities else None),
        "first_match_at":first_match,
        "median_anti_match_run":float(np.median(runs)) if runs else 0.0,
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
    for topk in [2,3,4]:
        for mw in [.115,.110,.105,.100]:
            for cons in [2,3,4]:
                out.append({
                    "id":f"veto_k{topk}_w{mw:.3f}_c{cons}",
                    "family":"veto",
                    "topk":topk,
                    "max_worst":mw,
                    "min_consensus":cons
                })
    expanded=[]
    for cfg in out:
        for mode in CADENCE_MODES:
            z=dict(cfg)
            z["cadence_mode"]=mode
            z["id"]=cfg["id"]+"_"+mode
            expanded.append(z)
    return expanded

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

    if fam=="veto":
        core=[n,t2,t3]
        hot=set()
        for src in core:
            hot.update(np.argsort(src)[-2:].tolist())
        bottom3=[set(np.argsort(src)[:3].tolist()) for src in (n,recent,t1,t2,t3)]
        cand=[int(d) for d in norder[:min(cfg["topk"],len(norder))] if int(d) not in hot]
        if not cand:
            return None
        def key(d):
            worst=max(float(n[d]),float(t2[d]),float(t3[d]))
            avg=.35*float(n[d])+.10*float(recent[d])+.15*float(t1[d])+.20*float(t2[d])+.20*float(t3[d])
            return (worst,avg,d)
        d=min(cand,key=key)
        consensus=sum(int(d in z) for z in bottom3)
        worst=max(float(n[d]),float(t2[d]),float(t3[d]))
        emit=worst<=cfg["max_worst"] and consensus>=cfg["min_consensus"]
        return d if emit else None

    return None

def eval_strategy(cfg,examples):
    out=[]
    last_barrier=None
    ticks_waited=0
    emitted=0
    mode=cfg.get("cadence_mode","ctx3_nr")
    prev_wait=0
    current_wait=cadence_wait(mode,examples[0],0,None,0) if examples else 1
    chain=[]
    idle=0
    max_idle=0
    for ex in examples:
        ticks_waited+=1
        idle+=1
        max_idle=max(max_idle,idle)
        if ticks_waited<current_wait:
            continue
        d=choose(cfg,ex,last_barrier)
        if d is None:
            continue
        out.append(int(int(ex["target"])!=d))
        emitted+=1
        last_barrier=d
        chain.append(current_wait)
        prev_wait=current_wait
        current_wait=cadence_wait(mode,ex,emitted,last_barrier,prev_wait)
        ticks_waited=0
        idle=0
    m=metrics(out,len(examples))
    m["cadence_mode"]=mode
    m["cadence_preview"]=chain[:30]
    m["no_repeat_digit"]=True
    m["signals_emitted"]=emitted
    m["max_idle_ticks"]=int(max_idle)
    m["frequency_ok"]=bool((m["signal_rate"] or 0)>=MIN_SIGNAL_RATE and max_idle<=MAX_IDLE_TICKS)
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
        out={"version":"2.0-zero-match-seek","timestamp":int(time.time()),"status":"COLLECTING",
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
            "tail":[int(x) for x in hist[-6:]],
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
        if m["signals"]<50 or not m.get("frequency_ok"):continue
        ranked.append((m["longest_anti_match_streak"],m["wilson_lower"],m["hit_rate"] or 0,m["signal_rate"] or 0,m["signals"],cfg,m))
    ranked.sort(reverse=True,key=lambda x:(x[0],x[1],x[2],x[3],x[4]))
    top=[x[5] for x in ranked[:36]]

    validated=[]
    for cfg in top:
        md,_=eval_strategy(cfg,discovery)
        mv,_=eval_strategy(cfg,validation)
        if mv["signals"]<40 or not mv.get("frequency_ok"):continue
        stable_streak=min(md["longest_anti_match_streak"],mv["longest_anti_match_streak"])
        validated.append((stable_streak,mv["wilson_lower"],mv["hit_rate"] or 0,mv["signal_rate"] or 0,mv["signals"],cfg,md,mv))
    validated.sort(reverse=True,key=lambda x:(x[0],x[1],x[2],x[3],x[4]))
    finalists=validated[:18]

    results=[]
    for _,_,_,_,_,cfg,md,mv in finalists:
        mh,oh=eval_strategy(cfg,holdout)
        perfect=bool(
            mv["signals"]>=50 and mh["signals"]>=50 and
            mv["losses"]==0 and mh["losses"]==0 and
            mv.get("frequency_ok") and mh.get("frequency_ok")
        )
        results.append({
            "id":cfg["id"],"config":cfg,
            "discovery":md,"validation":mv,"holdout":mh,
            "zero_match_validation_and_holdout":perfect
        })

    results.sort(key=lambda x:(
        x["zero_match_validation_and_holdout"],
        min(x["validation"]["longest_anti_match_streak"],x["holdout"]["longest_anti_match_streak"]),
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
        "target":"Maximize the minimum anti-MATCH streak across chronological validation and holdout while keeping signal_rate >= 30% and max idle <= 18 ticks; also search for 0-MATCH blocks.",
        "note":"ZERO-MATCH SEEK tests weighted, consensus, minimax-veto and dynamic cadence families (including 1-4 and 1-5 no-repeat). Sparse/frozen strategies are rejected; finite 0-MATCH results are not a future guarantee."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
