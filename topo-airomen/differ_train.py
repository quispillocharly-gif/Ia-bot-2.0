#!/usr/bin/env python3
from pathlib import Path
import json, time, hashlib
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"differ_train_latest.json"; MODEL=M/"differ_candidate.joblib"; META=M/"differ_candidate.json"; LIVE=M/"differ_live_model.json"

CONFIGS=[
 {"window":8,"hidden":(32,),"alpha":.0002,"lr":.0015},
 {"window":10,"hidden":(48,),"alpha":.0002,"lr":.0014},
 {"window":20,"hidden":(64,32),"alpha":.0004,"lr":.0012},
 {"window":40,"hidden":(96,48),"alpha":.0006,"lr":.0010},
]

def dataset(digits,w):
    X=[]; y=[]
    for i in range(w,len(digits)): X.append(digits[i-w:i]); y.append(digits[i])
    return np.asarray(X,dtype=np.int16),np.asarray(y,dtype=np.int16)

def full_probs(proba,classes):
    full=np.zeros((len(proba),10),dtype=float)
    for j,c in enumerate(classes): full[:,int(c)]=proba[:,j]
    return full

def metrics(proba,y,classes):
    full=full_probs(proba,classes); barriers=np.argmin(full,axis=1)
    wins=(barriers!=y); target=np.eye(10)[y]
    return {"signals":int(len(y)),"wins":int(wins.sum()),"hit_rate":float(wins.mean()),
            "mean_min_probability":float(full[np.arange(len(y)),barriers].mean()),
            "brier10":float(np.mean((full-target)**2)),
            "log_loss":float(log_loss(y,full,labels=list(range(10))))}

def main():
    raw=json.loads((R/"ticks.json").read_text()); digits=np.asarray([int(x["digit"]) for x in raw["ticks"]],dtype=np.int16)
    if len(digits)<5000: raise RuntimeError("Need at least 5000 ticks")
    rows=[]
    for k,cfg in enumerate(CONFIGS):
        X,y=dataset(digits,cfg["window"]); n=len(y); a=int(n*.55); b=int(n*.75)
        enc=OneHotEncoder(categories=[list(range(10))]*cfg["window"],handle_unknown="ignore",sparse_output=True)
        Xtr=enc.fit_transform(X[:a]); Xv=enc.transform(X[a:b]); Xt=enc.transform(X[b:])
        clf=MLPClassifier(hidden_layer_sizes=cfg["hidden"],alpha=cfg["alpha"],learning_rate_init=cfg["lr"],
                          max_iter=40,random_state=1900+k,early_stopping=True,n_iter_no_change=5)
        clf.fit(Xtr,y[:a])
        val=metrics(clf.predict_proba(Xv),y[a:b],clf.classes_)
        test=metrics(clf.predict_proba(Xt),y[b:],clf.classes_)
        rows.append({"cfg":cfg,"val":val,"test":test,"clf":clf,"enc":enc})
    rows.sort(key=lambda r:(r["val"]["brier10"],r["val"]["log_loss"]))
    best=rows[0]; sig=f'w{best["cfg"]["window"]}-h{"x".join(map(str,best["cfg"]["hidden"]))}'
    model_id=hashlib.sha256((sig+str(int(time.time())//3600)).encode()).hexdigest()[:16]
    joblib.dump({"clf":best["clf"],"encoder":best["enc"],"window":best["cfg"]["window"],"model_id":model_id},MODEL)
    live={
      "model_id":model_id,
      "window":int(best["cfg"]["window"]),
      "classes":[int(x) for x in best["clf"].classes_],
      "activation":str(best["clf"].activation),
      "out_activation":str(best["clf"].out_activation_),
      "coefs":[np.asarray(x,dtype=float).tolist() for x in best["clf"].coefs_],
      "intercepts":[np.asarray(x,dtype=float).tolist() for x in best["clf"].intercepts_],
      "created_at":int(time.time())
    }
    LIVE.write_text(json.dumps(live))
    meta={"model_id":model_id,"created_at":int(time.time()),"last_epoch":int(raw["ticks"][-1]["epoch"]),
          "window":best["cfg"]["window"],"hidden":list(best["cfg"]["hidden"]),
          "validation":best["val"],"test":best["test"],"selection_metric":"lowest multiclass Brier, then log loss"}
    META.write_text(json.dumps(meta,indent=2))
    safe=[{"window":r["cfg"]["window"],"hidden":list(r["cfg"]["hidden"]),"validation":r["val"],"test":r["test"]} for r in rows]
    out={"version":"1.0-differ-neural-research","timestamp":int(time.time()),"baseline":.90,
         "candidate":meta,"ranking":safe,
         "note":"Chronological train/validation/test. Candidate selection uses probability quality, not DIFFER hit rate alone."}
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
