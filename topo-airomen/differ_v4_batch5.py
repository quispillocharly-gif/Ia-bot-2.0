#!/usr/bin/env python3
from pathlib import Path
import json, time

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
V3=M/"differ_v3_state.json"
PAYOUT=M/"differ_payout_snapshot.json"
STATE=M/"differ_v4_batch5_state.json"
OUT=M/"differ_v4_batch5_latest.json"

TARGET=20.0
STAGE=5.0
STOP=5.0
BASE=1.0
MAX_STAKE=5.0
BATCH=5

POLICIES=[
 {"id":"flat_b5_p0","compound":0.0,"pause":0},
 {"id":"c25_b5_p0","compound":0.25,"pause":0},
 {"id":"c25_b5_p1","compound":0.25,"pause":1},
 {"id":"c50_b5_p0","compound":0.50,"pause":0},
 {"id":"c50_b5_p1","compound":0.50,"pause":1},
 {"id":"c50_b5_p2","compound":0.50,"pause":2},
 {"id":"c100_b5_p1","compound":1.00,"pause":1},
]

def loadj(p, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {} if default is None else default

def payout_ratio():
    p=loadj(PAYOUT,{})
    vals=[]
    for x in p.get("per_digit",[]):
        a=x.get("ask_price"); po=x.get("payout")
        if a and po is not None:
            vals.append((float(po)-float(a))/float(a))
    return sum(vals)/len(vals) if vals else .09

def session():
    return {"pnl":0.0,"stage":1,"stake":BASE,"batch_trades":0,"batch_wins":0,
            "pause_left":0,"trades":0,"signals_seen":0}

def policy_state():
    return {"sessions_completed":0,"targets":0,"stops":0,"total_trades":0,
            "total_signals_seen":0,"target_trade_counts":[],"equity":0.0,
            "peak":0.0,"max_drawdown":0.0,"session":session()}

def med(xs):
    if not xs: return None
    a=sorted(xs); n=len(a)
    return float(a[n//2]) if n%2 else float((a[n//2-1]+a[n//2])/2)

def reset(p):
    p["session"]=session()

def step(p,cfg,hit,wr):
    s=p["session"]
    s["signals_seen"]+=1; p["total_signals_seen"]+=1
    if s["pause_left"]>0:
        s["pause_left"]-=1
        return

    stake=float(s["stake"])
    profit=stake*wr if hit else -stake
    s["pnl"]+=profit; s["trades"]+=1; p["total_trades"]+=1
    s["batch_trades"]+=1

    if hit:
        s["batch_wins"]+=1
        s["stake"]=min(MAX_STAKE,max(BASE,stake+profit*float(cfg["compound"])))
    else:
        s["stake"]=BASE
        s["pause_left"]=int(cfg["pause"])

    p["equity"]+=profit
    p["peak"]=max(float(p["peak"]),float(p["equity"]))
    p["max_drawdown"]=max(float(p["max_drawdown"]),float(p["peak"])-float(p["equity"]))

    if s["batch_trades"]>=BATCH:
        if s["batch_wins"]<BATCH:
            s["stake"]=BASE
        s["batch_trades"]=0; s["batch_wins"]=0

    if s["stage"]<4 and s["pnl"]>=float(s["stage"])*STAGE:
        s["stage"]+=1
        s["stake"]=BASE
        s["batch_trades"]=0; s["batch_wins"]=0; s["pause_left"]=0

    if s["pnl"]>=TARGET:
        p["sessions_completed"]+=1; p["targets"]+=1
        p["target_trade_counts"].append(int(s["trades"]))
        reset(p)
    elif s["pnl"]<=-STOP:
        p["sessions_completed"]+=1; p["stops"]+=1
        reset(p)

def main():
    v3=loadj(V3,{})
    variants=v3.get("variants") or {}
    if not variants:
        raise RuntimeError("No V3 state")

    st=loadj(STATE,{})
    if not st:
        st={
          "version":"1.0-batch5-prospective-state",
          "created_at":int(time.time()),
          "source_counts":{vid:int(s.get("signals",0)) for vid,s in variants.items()},
          "sources":{vid:{"policies":{c["id"]:policy_state() for c in POLICIES}} for vid in variants},
          "runs":0
        }

    wr=payout_ratio()
    processed=0
    for vid,src in variants.items():
        if vid not in st["sources"]:
            st["sources"][vid]={"policies":{c["id"]:policy_state() for c in POLICIES}}
            st["source_counts"][vid]=int(src.get("signals",0))
            continue

        old=int(st["source_counts"].get(vid,0))
        new=int(src.get("signals",0))
        delta=max(0,new-old)
        outcomes=[int(x) for x in src.get("outcomes",[])]
        if delta>len(outcomes):
            st["source_counts"][vid]=new
            continue
        fresh=outcomes[-delta:] if delta else []
        processed+=len(fresh)
        for hit in fresh:
            for cfg in POLICIES:
                step(st["sources"][vid]["policies"][cfg["id"]],cfg,bool(hit),wr)
        st["source_counts"][vid]=new

    st["runs"]=int(st.get("runs",0))+1
    STATE.write_text(json.dumps(st,indent=2))

    rows=[]
    for vid,src in st["sources"].items():
        for cfg in POLICIES:
            p=src["policies"][cfg["id"]]
            comp=int(p["sessions_completed"])
            s=p["session"]
            rows.append({
              "source":vid,"policy":cfg["id"],"compound_fraction":cfg["compound"],
              "pause_after_match":cfg["pause"],"batch_size":BATCH,"stage_size":STAGE,
              "sessions_completed":comp,"targets":int(p["targets"]),"stops":int(p["stops"]),
              "target_rate":float(p["targets"]/comp) if comp else None,
              "total_trades":int(p["total_trades"]),"total_signals_seen":int(p["total_signals_seen"]),
              "median_trades_to_target":med(p["target_trade_counts"]),
              "max_drawdown":float(p["max_drawdown"]),
              "open_session":{"pnl":float(s["pnl"]),"stage":int(s["stage"]),"trades":int(s["trades"]),
                              "signals_seen":int(s["signals_seen"]),"stake":float(s["stake"]),
                              "batch_trades":int(s["batch_trades"]),"batch_wins":int(s["batch_wins"]),
                              "pause_left":int(s["pause_left"])},
              "status":"MEASURED" if comp>=20 else "COLLECTING"
            })

    measured=[
      r for r in rows
      if r["sessions_completed"]>=20
      and r["target_rate"] is not None
      and r["target_rate"]>=0.30
    ]
    measured.sort(key=lambda r:(r["target_rate"],-r["max_drawdown"],-(r["median_trades_to_target"] or 99999)),reverse=True)
    leader=measured[0] if measured else None

    out={
      "version":"1.0-batch5-stage-lab","timestamp":int(time.time()),"runs":st["runs"],
      "new_v3_outcomes_processed":processed,"target":TARGET,"stage_size":STAGE,
      "stages":4,"batch_size":BATCH,"stop_loss":STOP,"base_stake":BASE,
      "max_stake":MAX_STAKE,"payout_win_profit_ratio":wr,"leader":leader,
      "policies":rows,"status":"POLICY_CANDIDATE" if leader else "COLLECTING",
      "promotion_rule":"At least 20 completed prospective sessions and target_rate >= 30% before money-management promotion.",
      "note":"Prospective only. Four +$5 stages, checkpoints every 5 trades, optional compounding and pauses after MATCH. Live AUTO-DEMO is unchanged until enough completed sessions exist."
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
