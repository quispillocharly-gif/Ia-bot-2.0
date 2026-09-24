#!/usr/bin/env python3
from pathlib import Path
import json, math, time, hashlib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"neural_v5_latest.json"
STATE=M/"neural_v5_state.json"
MODEL=M/"neural_v5_candidate.joblib"
META=M/"neural_v5_candidate.json"
CONFIRM=M/"v5_1_confirmation_latest.json"
GRAVE=M/"v6_graveyard.json"

BASE=.10
SEARCH_VERSION="5.2"
CONFIGS=[
    {"window":8, "hidden":(32,), "alpha":1e-4, "lr":.0015},
    {"window":10,"hidden":(48,), "alpha":1e-4, "lr":.0015},
    {"window":20,"hidden":(64,32), "alpha":2e-4, "lr":.0012},
    {"window":30,"hidden":(64,32), "alpha":4e-4, "lr":.0012},
    {"window":40,"hidden":(96,48), "alpha":5e-4, "lr":.0010},
    {"window":60,"hidden":(96,48), "alpha":8e-4, "lr":.0010},
]
THRESHOLDS=[0.0,.115,.120,.125,.130,.140,.150,.160,.180]
MIN_SELECT_SIGNALS=300
MIN_SELECT_COVERAGE=.15
MIN_PROMOTE_SIGNALS=400
MIN_PROMOTE_COVERAGE=.15
MIN_PROMOTE_WILSON=.085

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def raw_xy(ds,window,start,end):
    start=max(start,window)
    X=[ds[i-window:i] for i in range(start,end)]
    y=[ds[i] for i in range(start,end)]
    return np.asarray(X,dtype=np.int16),np.asarray(y,dtype=np.int16)

def signal_metrics(clf,X,y,threshold):
    proba=clf.predict_proba(X)
    conf=proba.max(axis=1)
    pred=clf.classes_[proba.argmax(axis=1)].astype(np.int16)
    mask=conf>=threshold
    n=int(mask.sum()); total=int(len(y))
    wins=int((pred[mask]==y[mask]).sum()) if n else 0
    rate=wins/n if n else 0.0
    return {
        "predictions":n,"wins":wins,"hit_rate":rate,"wilson_lower":wilson(wins,n),
        "edge_vs_10pct":rate-BASE,"coverage":n/total if total else 0.0,
        "mean_confidence":float(conf[mask].mean()) if n else 0.0,
        "threshold":float(threshold),"opportunities":total,
    }

def choose_threshold(clf,X,y):
    scored=[]
    for t in THRESHOLDS:
        m=signal_metrics(clf,X,y,t)
        scored.append(m)
    eligible=[m for m in scored if m["predictions"]>=MIN_SELECT_SIGNALS and m["coverage"]>=MIN_SELECT_COVERAGE]
    pool=eligible or [scored[0]]
    return max(pool,key=lambda m:(m["wilson_lower"],m["hit_rate"],m["predictions"]))

def model_id(cfg,threshold,last_epoch,seed):
    raw=json.dumps({"cfg":cfg,"threshold":threshold,"last_epoch":last_epoch,"seed":seed,"v":SEARCH_VERSION},sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]

def model_signature(cfg,threshold):
    hidden="x".join(str(x) for x in cfg["hidden"])
    return f'w{int(cfg["window"])}-h{hidden}-t{float(threshold):.3f}'

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    ds=[int(x["digit"]) for x in ticks]
    if len(ds)<4000:
        raise RuntimeError("Need at least 4000 ticks for V5.2 chronological train/select/test")
    last_epoch=int(ticks[-1]["epoch"])

    st=json.loads(STATE.read_text()) if STATE.exists() else {"runs":0}
    run_no=int(st.get("runs",0))+1

    train_end=max(2000,int(len(ds)*.55))
    select_end=max(train_end+800,int(len(ds)*.75))
    select_end=min(select_end,len(ds)-800)

    selection_rows=[]
    trained=[]
    for k,cfg in enumerate(CONFIGS,1):
        w=int(cfg["window"])
        Xtr_raw,ytr=raw_xy(ds,w,w,train_end)
        Xsel_raw,ysel=raw_xy(ds,w,train_end,select_end)
        enc=OneHotEncoder(categories=[list(range(10))]*w,handle_unknown="ignore",sparse_output=False,dtype=np.float32)
        Xtr=enc.fit_transform(Xtr_raw)
        Xsel=enc.transform(Xsel_raw)
        seed=1000+run_no*37+k*11
        clf=MLPClassifier(
            hidden_layer_sizes=cfg["hidden"],activation="relu",solver="adam",
            alpha=cfg["alpha"],batch_size=128,learning_rate_init=cfg["lr"],
            max_iter=55,early_stopping=True,validation_fraction=.15,
            n_iter_no_change=6,random_state=seed
        )
        clf.fit(Xtr,ytr)
        chosen=choose_threshold(clf,Xsel,ysel)
        row={
            "model":"mlp","window":w,"hidden":list(cfg["hidden"]),"alpha":cfg["alpha"],
            "learning_rate":cfg["lr"],"epochs":int(getattr(clf,"n_iter_",0)),
            "selection_predictions":chosen["predictions"],"selection_wins":chosen["wins"],
            "selection_hit_rate":chosen["hit_rate"],"selection_wilson_lower":chosen["wilson_lower"],
            "selection_coverage":chosen["coverage"],"confidence_threshold":chosen["threshold"],
            "selection_mean_confidence":chosen["mean_confidence"],"seed":seed
        }
        selection_rows.append(row)
        trained.append((row,clf,enc,cfg,seed))

    selection_rows.sort(key=lambda x:(x["selection_wilson_lower"],x["selection_hit_rate"],x["selection_predictions"]),reverse=True)
    selected=selection_rows[0]
    row,clf,enc,cfg,seed=next(t for t in trained if t[0] is selected)

    Xtest_raw,ytest=raw_xy(ds,int(cfg["window"]),select_end,len(ds))
    Xtest=enc.transform(Xtest_raw)
    chosen_threshold=float(selected["confidence_threshold"])
    test=signal_metrics(clf,Xtest,ytest,chosen_threshold)
    parts=np.array_split(np.arange(len(ytest)),3)
    test_blocks=[signal_metrics(clf,Xtest[ix],ytest[ix],chosen_threshold) for ix in parts if len(ix)]
    positive_blocks=sum(1 for m in test_blocks if m["predictions"]>=100 and m["hit_rate"]>BASE)
    eligible_block_rates=[m["hit_rate"] for m in test_blocks if m["predictions"]>=100]
    block_floor=min(eligible_block_rates) if eligible_block_rates else 0.0
    block_stable=positive_blocks>=2 and block_floor>=.085
    leader={
        "model":"mlp","window":int(cfg["window"]),"hidden":list(cfg["hidden"]),
        "confidence_threshold":float(selected["confidence_threshold"]),
        "predictions":test["predictions"],"wins":test["wins"],"hit_rate":test["hit_rate"],
        "wilson_lower":test["wilson_lower"],"edge_vs_10pct":test["edge_vs_10pct"],
        "coverage":test["coverage"],"mean_confidence":test["mean_confidence"],
        "opportunities":test["opportunities"],"epochs":int(getattr(clf,"n_iter_",0)),
        "selection_hit_rate":selected["selection_hit_rate"],
        "selection_wilson_lower":selected["selection_wilson_lower"],
        "selection_coverage":selected["selection_coverage"],
        "positive_test_blocks":positive_blocks,
        "test_block_floor":block_floor,
        "block_stable":block_stable,
        "test_blocks":test_blocks,
    }

    challenger_id=model_id(cfg,leader["confidence_threshold"],last_epoch,seed)
    challenger_signature=model_signature(cfg,leader["confidence_threshold"])
    grave=json.loads(GRAVE.read_text()) if GRAVE.exists() else {"entries":[]}
    blocked_by_graveyard=any(
        x.get("signature")==challenger_signature and
        run_no<=int(x.get("blocked_until_v5_run",0))
        for x in grave.get("entries",[])
    )
    base_promotable=(
        leader["predictions"]>=MIN_PROMOTE_SIGNALS and
        leader["coverage"]>=MIN_PROMOTE_COVERAGE and
        leader["hit_rate"]>BASE and
        leader["wilson_lower"]>=MIN_PROMOTE_WILSON
    )
    if base_promotable and block_stable:
        challenger_streak=(int(st.get("challenger_streak",0))+1) if st.get("challenger_signature")==challenger_signature else 1
    else:
        challenger_streak=0
    promotable=base_promotable and block_stable and challenger_streak>=2

    incumbent=json.loads(META.read_text()) if META.exists() else None
    confirm=json.loads(CONFIRM.read_text()) if CONFIRM.exists() else None
    incumbent_failed=bool(
        incumbent and confirm and confirm.get("status")=="NOT_CONFIRMED" and
        confirm.get("model_id")==incumbent.get("model_id")
    )
    legacy_incumbent=bool(incumbent and incumbent.get("search_version")!=SEARCH_VERSION)
    improves=bool(incumbent and leader["wilson_lower"]>float(incumbent.get("wilson_lower",0))+0.001)
    bootstrap=incumbent is None

    promote=bool(
        not blocked_by_graveyard and promotable and
        (bootstrap or legacy_incumbent or incumbent_failed or improves)
    )

    if blocked_by_graveyard:
        promotion="HELD_GRAVEYARD"
    elif not block_stable:
        promotion="HELD_BLOCK_STABILITY"
    elif challenger_streak<2:
        promotion="HELD_STREAK"
    else:
        promotion="HELD"
    if promote:
        bundle={
            "search_version":SEARCH_VERSION,"model_id":challenger_id,
            "window":int(cfg["window"]),"hidden":tuple(cfg["hidden"]),
            "confidence_threshold":float(leader["confidence_threshold"]),
            "clf":clf,"encoder":enc
        }
        joblib.dump(bundle,MODEL)
        meta={
            "search_version":SEARCH_VERSION,"model_id":challenger_id,
            "window":int(cfg["window"]),"hidden":list(cfg["hidden"]),
            "confidence_threshold":float(leader["confidence_threshold"]),
            "last_epoch":last_epoch,"created_at":int(time.time()),
            "hit_rate":leader["hit_rate"],"wilson_lower":leader["wilson_lower"],
            "coverage":leader["coverage"],"predictions":leader["predictions"]
        }
        META.write_text(json.dumps(meta,indent=2))
        incumbent=meta
        promotion="BOOTSTRAP_PROMOTED" if bootstrap else "PROMOTED"

    strict_edge=leader["predictions"]>=500 and leader["wilson_lower"]>BASE
    if strict_edge:
        status="NEURAL_EDGE_CANDIDATE"
    elif promotable:
        status="NEURAL_CHALLENGER"
    else:
        status="NO_NEURAL_EDGE_DEMONSTRATED"

    st["runs"]=run_no; st["last_epoch"]=last_epoch
    st["challenger_signature"]=challenger_signature
    st["challenger_streak"]=challenger_streak
    STATE.write_text(json.dumps(st,indent=2))
    out={
        "version":"5.2-neural-evolution-lab","timestamp":int(time.time()),"runs":run_no,
        "ticks":len(ds),"train_end":train_end,"selection_end":select_end,
        "baseline":BASE,"status":status,"promotion":promotion,
        "leader":leader,"challenger_model_id":challenger_id,
        "challenger_signature":challenger_signature,
        "blocked_by_graveyard":blocked_by_graveyard,
        "block_stable":block_stable,
        "challenger_streak":challenger_streak,
        "candidate_model_id":incumbent.get("model_id") if incumbent else None,
        "candidate_search_version":incumbent.get("search_version") if incumbent else None,
        "selection_ranking":selection_rows,
        "note":"Architecture and confidence threshold are selected chronologically, evaluated on a later untouched test split, checked across three test blocks, and must repeat across runs before promotion. Future confirmation remains separate."
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
