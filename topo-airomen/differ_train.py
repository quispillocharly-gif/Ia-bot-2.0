#!/usr/bin/env python3
from pathlib import Path
import json, time, hashlib, math
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"differ_train_latest.json"
MODEL=M/"differ_candidate.joblib"
META=M/"differ_candidate.json"
LIVE=M/"differ_live_model.json"
ENS_MODEL=M/"differ_ensemble_candidate.joblib"
ENS_META=M/"differ_ensemble_candidate.json"
ENS_LIVE=M/"differ_live_ensemble.json"

CONFIGS=[
 {"window":8,"hidden":(32,),"alpha":.0002,"lr":.0015},
 {"window":10,"hidden":(48,),"alpha":.0002,"lr":.0014},
 {"window":20,"hidden":(64,32),"alpha":.0004,"lr":.0012},
 {"window":40,"hidden":(96,48),"alpha":.0006,"lr":.0010},
]
CAL_EDGES=[0.0,.05,.06,.07,.08,.09,.10,.12,1.01]

def dataset(digits,w):
    X=[]; y=[]
    for i in range(w,len(digits)):
        X.append(digits[i-w:i]); y.append(digits[i])
    return np.asarray(X,dtype=np.int16),np.asarray(y,dtype=np.int16)

def full_probs(proba,classes):
    full=np.zeros((len(proba),10),dtype=float)
    for j,c in enumerate(classes):
        full[:,int(c)]=proba[:,j]
    return full

def wilson_upper(w,n,z=1.96):
    if not n:return 1.0
    p=w/n; q=1+z*z/n
    return min(1.0,(p+z*z/(2*n)+z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def metrics(proba,y,classes):
    full=full_probs(proba,classes); barriers=np.argmin(full,axis=1)
    wins=(barriers!=y); target=np.eye(10)[y]
    return {
      "signals":int(len(y)),"wins":int(wins.sum()),"hit_rate":float(wins.mean()),
      "mean_min_probability":float(full[np.arange(len(y)),barriers].mean()),
      "brier10":float(np.mean((full-target)**2)),
      "log_loss":float(log_loss(y,full,labels=list(range(10))))
    }

def calibration(proba,y,classes):
    full=full_probs(proba,classes); barriers=np.argmin(full,axis=1)
    pmin=full[np.arange(len(y)),barriers]; losses=(barriers==y)
    rows=[]
    for lo,hi in zip(CAL_EDGES[:-1],CAL_EDGES[1:]):
        mask=(pmin>=lo)&(pmin<hi); n=int(mask.sum()); loss=int(losses[mask].sum()) if n else 0
        rows.append({
          "lo":lo,"hi":hi,"n":n,"losses":loss,
          "empirical_loss":float(loss/n) if n else None,
          "upper95_loss":float(wilson_upper(loss,n)) if n else 1.0
        })
    return rows

def live_payload(row):
    clf=row["clf"]
    return {
      "name":row["name"],"window":int(row["cfg"]["window"]),
      "classes":[int(x) for x in clf.classes_],
      "activation":str(clf.activation),"out_activation":str(clf.out_activation_),
      "coefs":[np.asarray(x,dtype=float).tolist() for x in clf.coefs_],
      "intercepts":[np.asarray(x,dtype=float).tolist() for x in clf.intercepts_],
      "calibration":row["calibration"]
    }

def main():
    raw=json.loads((R/"ticks.json").read_text())
    digits=np.asarray([int(x["digit"]) for x in raw["ticks"]],dtype=np.int16)
    if len(digits)<5000: raise RuntimeError("Need at least 5000 ticks")
    rows=[]
    for k,cfg in enumerate(CONFIGS):
        X,y=dataset(digits,cfg["window"]); n=len(y); a=int(n*.55); b=int(n*.75)
        enc=OneHotEncoder(categories=[list(range(10))]*cfg["window"],handle_unknown="ignore",sparse_output=True)
        Xtr=enc.fit_transform(X[:a]); Xv=enc.transform(X[a:b]); Xt=enc.transform(X[b:])
        clf=MLPClassifier(
          hidden_layer_sizes=cfg["hidden"],alpha=cfg["alpha"],learning_rate_init=cfg["lr"],
          max_iter=40,random_state=1900+k,early_stopping=True,n_iter_no_change=5
        )
        clf.fit(Xtr,y[:a])
        pv=clf.predict_proba(Xv); pt=clf.predict_proba(Xt)
        rows.append({
          "name":"w"+str(cfg["window"])+"-h"+"x".join(map(str,cfg["hidden"])),
          "cfg":cfg,"val":metrics(pv,y[a:b],clf.classes_),"test":metrics(pt,y[b:],clf.classes_),
          "calibration":calibration(pv,y[a:b],clf.classes_),"clf":clf,"enc":enc
        })
    rows.sort(key=lambda r:(r["val"]["brier10"],r["val"]["log_loss"]))
    best=rows[0]
    sig=best["name"]
    model_id=hashlib.sha256((sig+str(int(time.time())//3600)).encode()).hexdigest()[:16]
    joblib.dump({"clf":best["clf"],"encoder":best["enc"],"window":best["cfg"]["window"],"model_id":model_id},MODEL)
    live=live_payload(best); live["model_id"]=model_id; live["created_at"]=int(time.time())
    LIVE.write_text(json.dumps(live))
    meta={
      "model_id":model_id,"created_at":int(time.time()),"last_epoch":int(raw["ticks"][-1]["epoch"]),
      "window":best["cfg"]["window"],"hidden":list(best["cfg"]["hidden"]),
      "validation":best["val"],"test":best["test"],"calibration":best["calibration"],
      "selection_metric":"lowest multiclass Brier, then log loss"
    }
    META.write_text(json.dumps(meta,indent=2))

    top=rows[:3]
    ens_sig="|".join(r["name"] for r in top)
    ensemble_id=hashlib.sha256((ens_sig+str(int(time.time())//3600)).encode()).hexdigest()[:16]
    joblib.dump({
      "ensemble_id":ensemble_id,
      "models":[
        {"name":r["name"],"clf":r["clf"],"encoder":r["enc"],"window":r["cfg"]["window"],"calibration":r["calibration"]}
        for r in top
      ]
    },ENS_MODEL)
    ens_meta={
      "ensemble_id":ensemble_id,"created_at":int(time.time()),"last_epoch":int(raw["ticks"][-1]["epoch"]),
      "members":[
        {"name":r["name"],"window":r["cfg"]["window"],"hidden":list(r["cfg"]["hidden"]),
         "validation":r["val"],"test":r["test"],"calibration":r["calibration"]} for r in top
      ],
      "selection":"top 3 by validation multiclass Brier then log loss"
    }
    ENS_META.write_text(json.dumps(ens_meta,indent=2))
    ENS_LIVE.write_text(json.dumps({
      "ensemble_id":ensemble_id,"created_at":int(time.time()),
      "models":[live_payload(r) for r in top]
    }))
    safe=[{
      "name":r["name"],"window":r["cfg"]["window"],"hidden":list(r["cfg"]["hidden"]),
      "validation":r["val"],"test":r["test"],"calibration":r["calibration"]
    } for r in rows]
    out={
      "version":"2.0-differ-neural-ensemble-research","timestamp":int(time.time()),"baseline":.90,
      "candidate":meta,"ensemble_candidate":ens_meta,"ranking":safe,
      "note":"Chronological train/validation/test. V2 exports a top-3 ensemble and calibration learned only from the validation slice."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
