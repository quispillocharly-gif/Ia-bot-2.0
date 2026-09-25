#!/usr/bin/env python3
from pathlib import Path
import json, time
import numpy as np

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"differ_session_eval_latest.json"
TARGET=20.0
STOP=5.0
BASE_STAKE=1.0
MAX_STAKE=5.0
BLOCK=20
SIMS=1500
MAX_TRADES=2000
SEED=20260925

def load(name,default=None):
    try:return json.loads((M/name).read_text())
    except Exception:return {} if default is None else default

def payout_ratio():
    p=load("differ_payout_snapshot.json",{})
    vals=[]
    for x in p.get("per_digit",[]):
        a=x.get("ask_price"); po=x.get("payout")
        if a and po is not None:
            vals.append((float(po)-float(a))/float(a))
    return float(np.mean(vals)) if vals else .09

def simulate(outcomes,win_ratio,rng):
    n=len(outcomes)
    pnl=0.0; stake=BASE_STAKE; trades=0
    while trades<MAX_TRADES and pnl<TARGET and pnl>-STOP:
        start=int(rng.integers(0,max(1,n-BLOCK+1)))
        block=outcomes[start:start+BLOCK]
        if not block:block=outcomes
        for hit in block:
            if hit:
                profit=stake*win_ratio
                pnl+=profit
                stake=min(MAX_STAKE,max(BASE_STAKE,stake+profit))
            else:
                pnl-=stake
                stake=BASE_STAKE
            trades+=1
            if pnl>=TARGET or pnl<=-STOP or trades>=MAX_TRADES:break
    return pnl,trades,("TARGET" if pnl>=TARGET else "STOP" if pnl<=-STOP else "TIMEOUT")

def eval_variant(source,vid,s):
    outcomes=[int(x) for x in s.get("outcomes",[])]
    n=len(outcomes)
    if n<100:
        return {"source":source,"id":vid,"signals":n,"status":"INSUFFICIENT"}
    rng=np.random.default_rng(SEED+sum(ord(c) for c in (source+vid)))
    wr=payout_ratio()
    results=[simulate(outcomes,wr,rng) for _ in range(SIMS)]
    pnl=np.asarray([x[0] for x in results],dtype=float)
    trades=np.asarray([x[1] for x in results],dtype=int)
    states=[x[2] for x in results]
    targets=np.asarray([st=="TARGET" for st in states])
    stops=np.asarray([st=="STOP" for st in states])
    target_trades=trades[targets]
    return {
      "source":source,"id":vid,"signals":n,
      "empirical_hit_rate":float(np.mean(outcomes)),
      "target":TARGET,"stop_loss":STOP,"base_stake":BASE_STAKE,"max_stake":MAX_STAKE,
      "simulations":SIMS,"block_length":BLOCK,"max_trades":MAX_TRADES,
      "target_rate":float(targets.mean()),"stop_rate":float(stops.mean()),
      "timeout_rate":float(1-targets.mean()-stops.mean()),
      "median_trades_to_target":float(np.median(target_trades)) if len(target_trades) else None,
      "mean_final_pnl":float(pnl.mean()),
      "p10_final_pnl":float(np.quantile(pnl,.10)),
      "p50_final_pnl":float(np.quantile(pnl,.50)),
      "p90_final_pnl":float(np.quantile(pnl,.90)),
      "status":"SIMULATED"
    }

def main():
    rows=[]
    v1=load("differ_forward_state.json",{})
    for vid,s in (v1.get("variants") or {}).items():
        rows.append(eval_variant("v1",vid,s))
    v2=load("differ_v2_state.json",{})
    for vid,s in (v2.get("variants") or {}).items():
        rows.append(eval_variant("v2",vid,s))
    eligible=[r for r in rows if r.get("status")=="SIMULATED" and int(r.get("signals",0))>=300]
    eligible.sort(key=lambda r:(r["target_rate"],-r["stop_rate"],r["mean_final_pnl"],r["signals"]),reverse=True)
    out={
      "version":"1.0-session-target-lab","timestamp":int(time.time()),
      "target":TARGET,"stop_loss":STOP,"base_stake":BASE_STAKE,"max_stake":MAX_STAKE,
      "payout_win_profit_ratio":payout_ratio(),
      "best":eligible[0] if eligible else None,
      "variants":rows,
      "note":"Diagnostic block-bootstrap over forward outcomes. It estimates session behavior under observed sequences; it does not guarantee future target attainment."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":main()
