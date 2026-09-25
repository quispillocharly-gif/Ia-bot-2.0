#!/usr/bin/env python3
from pathlib import Path
import json, os, time

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
TICKS=R/"ticks.json"
STATE=M/"differ_offline_state.json"
OUT=M/"differ_offline_stats.json"

DECAY=float(os.getenv("OFFLINE_DECAY","0.985"))
SYMBOL=os.getenv("SYMBOL","R_75")

def empty_state():
    return {
        "version":"1.0",
        "symbol":SYMBOL,
        "last_epoch":0,
        "tail":[],
        "runs":0,
        "total_new_ticks":0,
        "global":[0.0]*10,
        "t1":{},
        "t2":{},
        "t3":{}
    }

def decay_list(a):
    return [float(x)*DECAY for x in a]

def decay_state(s):
    s["global"]=decay_list(s.get("global",[0.0]*10))
    for name in ("t1","t2","t3"):
        tbl=s.get(name,{})
        for k,v in list(tbl.items()):
            tbl[k]=decay_list(v)
        s[name]=tbl

def bump(tbl,key,target):
    row=tbl.setdefault(key,[0.0]*10)
    row[int(target)]+=1.0

def probs(row,prior=0.7):
    vals=[max(0.0,float(x))+prior for x in row]
    z=sum(vals) or 1.0
    return [round(x/z,6) for x in vals]

def pack_table(tbl,min_support=0.0):
    out={}
    for k,row in tbl.items():
        n=float(sum(row))
        if n<min_support:
            continue
        out[k]={"n":round(n,3),"p":probs(row)}
    return out

def main():
    raw=json.loads(TICKS.read_text())
    ticks=raw.get("ticks") or []
    if not ticks:
        raise RuntimeError("ticks.json has no ticks")

    if STATE.exists():
        try:
            state=json.loads(STATE.read_text())
        except Exception:
            state=empty_state()
    else:
        state=empty_state()

    last_epoch=int(state.get("last_epoch") or 0)
    unseen=[x for x in ticks if int(x.get("epoch",0))>last_epoch]

    if unseen:
        decay_state(state)
        tail=[int(x) for x in state.get("tail",[])][-3:]
        digits=tail+[int(x["digit"]) for x in unseen]
        start=len(tail)

        for i in range(start,len(digits)):
            y=digits[i]
            state["global"][y]+=1.0
            if i>=1:
                bump(state["t1"],str(digits[i-1]),y)
            if i>=2:
                bump(state["t2"],f"{digits[i-2]},{digits[i-1]}",y)
            if i>=3:
                bump(state["t3"],f"{digits[i-3]},{digits[i-2]},{digits[i-1]}",y)

        state["last_epoch"]=max(int(x["epoch"]) for x in unseen)
        state["tail"]=digits[-3:]
        state["runs"]=int(state.get("runs",0))+1
        state["total_new_ticks"]=int(state.get("total_new_ticks",0))+len(unseen)

    now=int(time.time())
    global_counts=state.get("global",[0.0]*10)
    out={
        "version":"1.0-offline-stat-memory",
        "symbol":SYMBOL,
        "created_at":now,
        "last_epoch":int(state.get("last_epoch") or 0),
        "runs":int(state.get("runs",0)),
        "new_ticks_this_run":len(unseen),
        "total_new_ticks":int(state.get("total_new_ticks",0)),
        "decay":DECAY,
        "global":{"n":round(sum(global_counts),3),"p":probs(global_counts)},
        "t1":pack_table(state.get("t1",{}),1.0),
        "t2":pack_table(state.get("t2",{}),2.0),
        "t3":pack_table(state.get("t3",{}),2.0),
        "note":"Persistent decayed statistical memory. Only unseen ticks are added; older evidence decays each successful update."
    }

    STATE.write_text(json.dumps(state,indent=2,sort_keys=True))
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True))
    print(json.dumps({
        "status":"ok",
        "new_ticks":len(unseen),
        "last_epoch":out["last_epoch"],
        "runs":out["runs"],
        "t1_contexts":len(out["t1"]),
        "t2_contexts":len(out["t2"]),
        "t3_contexts":len(out["t3"])
    },indent=2))

if __name__=="__main__":
    main()
