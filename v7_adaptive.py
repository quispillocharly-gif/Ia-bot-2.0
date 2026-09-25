#!/usr/bin/env python3
from pathlib import Path
import json, math, shutil, time
import numpy as np
import joblib

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
CAND=M/"neural_v5_candidate.joblib"
CANDMETA=M/"neural_v5_candidate.json"
FROZEN=M/"v7_frozen_neural.joblib"
FMETA=M/"v7_frozen_neural.json"
STATE=M/"v7_adaptive_state.json"
OUT=M/"v7_adaptive_latest.json"
PAYOUT=M/"payout_snapshot.json"
BASE=.10
PAYOUT_MAX_AGE=7200

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def entropy_norm(h,n=200):
    a=h[-n:]
    if not a:return 1.0
    c=np.bincount(np.asarray(a,dtype=np.int16),minlength=10).astype(float)
    p=c/c.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/math.log(10))

def repeat_rate(h,n=200):
    a=h[-n:]
    if len(a)<2:return 0.0
    return float(sum(int(a[i]==a[i-1]) for i in range(1,len(a)))/(len(a)-1))

def max_share(h,n=200):
    a=h[-n:]
    if not a:return .1
    c=np.bincount(np.asarray(a,dtype=np.int16),minlength=10)
    return float(c.max()/len(a))

def transition_peak(h,n=500):
    a=h[-n:]
    if len(a)<20:return .1
    mat=np.zeros((10,10),dtype=np.int32)
    for x,y in zip(a[:-1],a[1:]):mat[x,y]+=1
    vals=[]
    for r in mat:
        s=r.sum()
        if s: vals.append(float(r.max()/s))
    return float(np.mean(vals)) if vals else .1

def regime(h):
    en=entropy_norm(h); rep=repeat_rate(h); mx=max_share(h); tp=transition_peak(h)
    if en<.955 or mx>.155:
        name="CONCENTRATED"
    elif rep>.125:
        name="REPEAT_HEAVY"
    elif tp>.18:
        name="TRANSITIONAL"
    else:
        name="UNIFORM"
    return name,{"entropy":en,"repeat_rate":rep,"max_share":mx,"transition_peak":tp}

def hot(h,n):
    a=h[-n:]; c=[a.count(d) for d in range(10)]
    return max(range(10),key=lambda d:(c[d],-d))
def cold(h,n):
    a=h[-n:]; c=[a.count(d) for d in range(10)]
    return min(range(10),key=lambda d:(c[d],d))
def gap(h):
    g=[]
    for d in range(10):
        try:v=len(h)-1-max(i for i,x in enumerate(h) if x==d)
        except ValueError:v=len(h)+1
        g.append(v)
    return max(range(10),key=lambda d:(g[d],-d))
def trans(h,o):
    a=h[-1000:]
    if len(a)<=o:return hot(a,80)
    ctx=tuple(a[-o:]); c=[0]*10
    for i in range(o,len(a)):
        if tuple(a[i-o:i])==ctx:c[a[i]]+=1
    return max(range(10),key=lambda d:(c[d],-d)) if sum(c) else hot(a,80)

def calibration_key(conf):
    edges=[0,.12,.14,.16,.18,.22,1.01]
    for a,b in zip(edges[:-1],edges[1:]):
        if a<=conf<b:return f"{a:.2f}-{b:.2f}"
    return "0.22-1.01"

def payout_map():
    try:p=json.loads(PAYOUT.read_text())
    except Exception:return {},None,None,False
    rows={}
    for x in p.get("per_digit",[]):
        if x.get("ask_price") is not None and x.get("payout") is not None:
            rows[int(x["digit"])]=(float(x["ask_price"]),float(x["payout"]))
    be=p.get("max_break_even") if p.get("status")=="OK" else None
    ts=p.get("timestamp")
    age=max(0,int(time.time())-int(ts)) if ts is not None else None
    fresh=bool(age is not None and age<=PAYOUT_MAX_AGE and p.get("status")=="OK" and len(rows)==10)
    return rows,float(be) if be is not None else None,age,fresh

def status_for(st,break_even,payout_fresh=True):
    n=int(st.get("n",0)); w=int(st.get("w",0))
    if n<1000:return "COLLECTING"
    if not payout_fresh:return "PAYOUT_STALE"
    threshold=break_even if break_even is not None else BASE
    recent=st.get("recent_outcomes",[])
    pval=float(st.get("placebo_economic_p",1.0))
    pnl=float(st.get("shadow_pnl",0.0))
    if wilson(w,n)>threshold and pnl>0 and pval<.05 and len(recent)>=300:
        return "ECONOMICALLY_CONFIRMED"
    return "NOT_CONFIRMED"

def freeze_candidate(ticks):
    if not CAND.exists() or not CANDMETA.exists():raise RuntimeError("No neural candidate")
    shutil.copy2(CAND,FROZEN); shutil.copy2(CANDMETA,FMETA)
    meta=json.loads(FMETA.read_text()); latest=int(ticks[-1]["epoch"])
    st={
        "version":"7.0","model_id":meta["model_id"],"start_epoch":latest,"last_epoch":latest,
        "history":[int(x["digit"]) for x in ticks[-1500:]],"runs":0,
        "opportunities":0,"n":0,"w":0,"skipped_confidence":0,"skipped_agreement":0,
        "shadow_pnl":0.0,"shadow_staked":0.0,"confidence_sum":0.0,"agreement_sum":0.0,
        "brier_sum":0.0,"multiclass_brier_sum":0.0,"calibration_n":0,"calibration":{},"regimes":{},
        "recent_outcomes":[]
    }
    STATE.write_text(json.dumps(st,indent=2)); return st

def monte_carlo(st,break_even):
    recent=list(map(int,st.get("recent_outcomes",[])))
    n=len(recent); w=sum(recent)
    if n<30:
        return 1.0,1.0,[None,None]
    seed=(sum(ord(c) for c in str(st.get("model_id","v7")))+int(st.get("runs",0))*97+n)%2**32
    rng=np.random.default_rng(seed)
    draws_random=rng.binomial(n,BASE,size=5000)
    econ_p=break_even if break_even is not None else BASE
    draws_econ=rng.binomial(n,econ_p,size=5000)
    pr=float(np.mean(draws_random>=w))
    pe=float(np.mean(draws_econ>=w))
    rate=w/n
    boot=rng.binomial(n,rate,size=5000)/n
    ci=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
    return pr,pe,ci

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    if not ticks:raise RuntimeError("No ticks")
    pmap,break_even,payout_age,payout_fresh=payout_map()
    candmeta=json.loads(CANDMETA.read_text()) if CANDMETA.exists() else None

    if STATE.exists() and FROZEN.exists() and FMETA.exists():
        st=json.loads(STATE.read_text())
        if status_for(st,break_even,payout_fresh)=="NOT_CONFIRMED" and candmeta and candmeta.get("model_id")!=st.get("model_id"):
            st=freeze_candidate(ticks)
    else:
        st=freeze_candidate(ticks)

    bundle=joblib.load(FROZEN); meta=json.loads(FMETA.read_text())
    window=int(bundle["window"]); clf=bundle["clf"]; enc=bundle["encoder"]
    threshold=float(bundle.get("confidence_threshold",meta.get("confidence_threshold",0.0) or 0.0))
    fresh=[x for x in ticks if int(x["epoch"])>int(st["last_epoch"])]
    hist=list(map(int,st.get("history",[])))
    recent=list(map(int,st.get("recent_outcomes",[])))

    for x in fresh:
        d=int(x["digit"])
        if len(hist)>=max(200,window):
            st["opportunities"]+=1
            X=enc.transform(np.asarray([hist[-window:]],dtype=np.int16))
            proba=clf.predict_proba(X)[0]; j=int(np.argmax(proba))
            pred=int(clf.classes_[j]); conf=float(proba[j])
            hit=int(pred==d)

            st["calibration_n"]+=1
            st["brier_sum"]+=float((conf-hit)**2)
            full=np.zeros(10,dtype=float)
            for cls,pv in zip(clf.classes_,proba): full[int(cls)]=float(pv)
            target=np.zeros(10,dtype=float); target[d]=1.0
            st["multiclass_brier_sum"]=float(st.get("multiclass_brier_sum",0.0))+float(np.mean((full-target)**2))
            key=calibration_key(conf)
            cb=st["calibration"].setdefault(key,{"n":0,"w":0,"conf_sum":0.0})
            cb["n"]+=1; cb["w"]+=hit; cb["conf_sum"]+=conf

            reg,features=regime(hist)
            rs=st["regimes"].setdefault(reg,{"opportunities":0,"n":0,"w":0,"pnl":0.0})
            rs["opportunities"]+=1

            experts=[hot(hist,20),cold(hist,20),hot(hist,40),cold(hist,40),gap(hist),trans(hist,1),trans(hist,2),trans(hist,3)]
            agree=sum(int(p==pred) for p in experts)
            min_agree=3 if reg=="UNIFORM" else 2

            if conf<threshold:
                st["skipped_confidence"]+=1
            elif agree<min_agree:
                st["skipped_agreement"]+=1
            else:
                st["n"]+=1; st["w"]+=hit
                st["confidence_sum"]+=conf; st["agreement_sum"]+=agree
                rs["n"]+=1; rs["w"]+=hit
                recent.append(hit)
                if len(recent)>2000:recent=recent[-2000:]
                if pred in pmap:
                    ask,payout=pmap[pred]
                    delta=(payout-ask) if hit else -ask
                    st["shadow_pnl"]+=delta; st["shadow_staked"]+=ask; rs["pnl"]+=delta

        hist.append(d)
        if len(hist)>1500:hist=hist[-1500:]
        st["last_epoch"]=int(x["epoch"])

    st["history"]=hist; st["recent_outcomes"]=recent; st["runs"]=int(st.get("runs",0))+1
    placebo_random,placebo_econ,boot_ci=monte_carlo(st,break_even)
    st["placebo_random_p"]=placebo_random; st["placebo_economic_p"]=placebo_econ
    STATE.write_text(json.dumps(st,indent=2))

    n=int(st["n"]); w=int(st["w"]); opp=int(st["opportunities"])
    rate=w/n if n else None; low=wilson(w,n); coverage=n/opp if opp else None

    cal_rows=[]
    ece_num=0.0; ece_den=0
    for key,v in sorted(st["calibration"].items()):
        bn=int(v["n"]); bw=int(v["w"]); avg=float(v["conf_sum"]/bn) if bn else None
        br=float(bw/bn) if bn else None
        if bn:
            ece_num+=abs(br-avg)*bn; ece_den+=bn
        cal_rows.append({"bin":key,"n":bn,"wins":bw,"hit_rate":br,"avg_confidence":avg})
    ece=ece_num/ece_den if ece_den else None
    brier=st["brier_sum"]/st["calibration_n"] if st["calibration_n"] else None
    brier_multi=float(st.get("multiclass_brier_sum",0.0))/st["calibration_n"] if st["calibration_n"] else None

    current_reg,current_features=regime(hist)
    regime_rows=[]
    for name,v in st["regimes"].items():
        rn=int(v["n"]); rw=int(v["w"])
        regime_rows.append({"regime":name,"opportunities":int(v["opportunities"]),"signals":rn,
                            "wins":rw,"hit_rate":rw/rn if rn else None,"wilson_lower":wilson(rw,rn),
                            "shadow_pnl":float(v["pnl"])})
    regime_rows.sort(key=lambda x:(x["wilson_lower"],x["signals"]),reverse=True)

    pnl=float(st["shadow_pnl"]); staked=float(st["shadow_staked"])
    ev_per_signal=pnl/n if n else None
    roi=pnl/staked if staked else None
    status=status_for(st,break_even,payout_fresh)
    out={
        "version":"7.0-adaptive-shadow-lab","timestamp":int(time.time()),
        "model_id":st["model_id"],"search_version":meta.get("search_version"),
        "runs":st["runs"],"new_ticks_this_run":len(fresh),
        "window":window,"hidden":meta.get("hidden"),"confidence_threshold":threshold,
        "current_regime":current_reg,"regime_features":current_features,"regime_performance":regime_rows,
        "opportunities":opp,"predictions":n,"wins":w,"losses":n-w,
        "hit_rate":rate,"wilson_lower":low,"coverage":coverage,
        "skipped_confidence":int(st["skipped_confidence"]),"skipped_agreement":int(st["skipped_agreement"]),
        "mean_confidence":st["confidence_sum"]/n if n else None,
        "mean_expert_agreement":st["agreement_sum"]/n if n else None,
        "calibration":{"ece":ece,"brier_binary":brier,"brier_multiclass":brier_multi,
                       "bins":cal_rows,"samples":int(st["calibration_n"])},
        "placebo":{"random_10pct_p":placebo_random,"economic_break_even_p":placebo_econ,
                   "bootstrap_95_rate":boot_ci,"recent_samples":len(recent)},
        "economic":{"break_even_rate":break_even,"shadow_pnl":pnl,"shadow_staked":staked,
                    "shadow_roi":roi,"ev_per_signal":ev_per_signal,
                    "payout_snapshot_available":bool(pmap),"payout_snapshot_age_seconds":payout_age,
                    "payout_snapshot_fresh":payout_fresh,"payout_max_age_seconds":PAYOUT_MAX_AGE},
        "status":status,"start_epoch":st["start_epoch"],
        "note":"Forward-only shadow laboratory. Regime-aware agreement, binary and proper multiclass calibration diagnostics, placebo tests, payout freshness and economic EV are measured without placing trades."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":main()
