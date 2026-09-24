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
BASE=.10
CANDIDATES=[
    {"window":10,"hidden":(32,)},
    {"window":20,"hidden":(48,)},
    {"window":40,"hidden":(64,32)},
]

def wilson(w,n,z=1.96):
    if not n:return 0.0
    p=w/n; q=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/q)

def build_xy(ds,window,start,end):
    start=max(start,window)
    X=[]; y=[]
    for i in range(start,end):
        X.append(ds[i-window:i]); y.append(ds[i])
    X=np.asarray(X,dtype=np.int16)
    y=np.asarray(y,dtype=np.int16)
    enc=OneHotEncoder(categories=[list(range(10))]*window,handle_unknown="ignore",sparse_output=False,dtype=np.float32)
    X=enc.fit_transform(X)
    return X,y,enc

def model_id(cfg,last_epoch):
    raw=json.dumps({"cfg":cfg,"last_epoch":last_epoch},sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]

def main():
    ticks=sorted(json.loads((R/"ticks.json").read_text())["ticks"],key=lambda x:int(x["epoch"]))
    ds=[int(x["digit"]) for x in ticks]
    last_epoch=int(ticks[-1]["epoch"]) if ticks else 0
    cut=max(1000,int(len(ds)*.70))
    rows=[]; trained=[]
    for k,cfg in enumerate(CANDIDATES,1):
        w=cfg["window"]
        Xtr,ytr,enc=build_xy(ds,w,w,cut)
        Xva,yva,_=build_xy(ds,w,cut,len(ds))
        clf=MLPClassifier(hidden_layer_sizes=cfg["hidden"],activation="relu",solver="adam",
            alpha=1e-4,batch_size=128,learning_rate_init=.0015,max_iter=45,
            early_stopping=True,validation_fraction=.15,n_iter_no_change=5,random_state=100+k)
        clf.fit(Xtr,ytr)
        pred=clf.predict(Xva)
        wins=int((pred==yva).sum()); n=int(len(yva)); rate=wins/n if n else 0.0
        low=wilson(wins,n)
        row={"model":"mlp","window":w,"hidden":list(cfg["hidden"]),"predictions":n,
             "wins":wins,"hit_rate":rate,"wilson_lower":low,"edge_vs_10pct":rate-BASE,
             "epochs":int(getattr(clf,"n_iter_",0))}
        rows.append(row); trained.append((row,clf,enc,cfg))
    rows.sort(key=lambda x:(x["wilson_lower"],x["hit_rate"]),reverse=True)
    best=rows[0]
    winner=next(t for t in trained if t[0]["window"]==best["window"] and t[0]["hidden"]==best["hidden"])
    _,clf,enc,cfg=winner
    mid=model_id(cfg,last_epoch)
    bundle={"model_id":mid,"window":cfg["window"],"hidden":cfg["hidden"],"clf":clf,"encoder":enc}
    joblib.dump(bundle,MODEL)
    meta={"model_id":mid,"window":cfg["window"],"hidden":list(cfg["hidden"]),"last_epoch":last_epoch,
          "created_at":int(time.time()),"hit_rate":best["hit_rate"],"wilson_lower":best["wilson_lower"]}
    META.write_text(json.dumps(meta,indent=2))
    st=json.loads(STATE.read_text()) if STATE.exists() else {"runs":0}
    st["runs"]+=1; st["last_epoch"]=last_epoch; STATE.write_text(json.dumps(st,indent=2))
    status="NEURAL_CANDIDATE" if best["predictions"]>=1000 and best["wilson_lower"]>BASE else "NO_NEURAL_EDGE_DEMONSTRATED"
    out={"version":"5.0-neural-lab","timestamp":int(time.time()),"runs":st["runs"],"ticks":len(ds),
         "train_end":cut,"baseline":BASE,"status":status,"leader":best,"candidate_model_id":mid,
         "top_models":rows,"note":"Neural candidates are trained only on earlier data and ranked on chronological holdout data."}
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
