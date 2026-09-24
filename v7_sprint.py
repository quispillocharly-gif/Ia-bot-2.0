#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, time

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"v7_sprint_latest.json"
START=datetime(2026,9,24,0,0,0,tzinfo=timezone(timedelta(hours=-5)))
DEADLINE=datetime(2026,10,8,23,59,59,tzinfo=timezone(timedelta(hours=-5)))

def load(name):
    try:return json.loads((M/name).read_text())
    except Exception:return {}

def main():
    now=datetime.now(timezone.utc)
    deadline_utc=DEADLINE.astimezone(timezone.utc)
    remaining=max(0.0,(deadline_utc-now).total_seconds())
    days_left=remaining/86400.0

    v7=load("v7_adaptive_latest.json")
    v61=load("v6_ensemble_latest.json")
    payout=load("payout_snapshot.json")
    v5=load("neural_v5_latest.json")

    econ=v7.get("economic") or {}
    placebo=v7.get("placebo") or {}
    calibration=v7.get("calibration") or {}
    n=int(v7.get("predictions",0) or 0)
    hit=v7.get("hit_rate")
    wilson=float(v7.get("wilson_lower",0) or 0)
    break_even=payout.get("max_break_even") if payout.get("status")=="OK" else econ.get("break_even_rate")
    break_even=float(break_even) if break_even is not None else None
    pnl=float(econ.get("shadow_pnl",0) or 0)
    ev=econ.get("ev_per_signal")
    placebo_econ=float(placebo.get("economic_break_even_p",1.0) or 1.0)
    ece=calibration.get("ece")

    regimes=v7.get("regime_performance") or []
    regime_samples=sum(1 for x in regimes if int(x.get("signals",0) or 0)>=100)
    regime_positive=sum(1 for x in regimes if int(x.get("signals",0) or 0)>=100 and float(x.get("shadow_pnl",0) or 0)>0)

    elapsed=max(1.0,time.time()-float(v7.get("start_epoch",time.time()) or time.time()))
    per_day=n/(elapsed/86400.0) if n else 0.0
    projected=n+per_day*days_left if per_day>0 else n

    gates={
        "forward_sample":{"pass":n>=1000,"value":n,"target":1000},
        "economic_hit_rate":{"pass":bool(hit is not None and break_even is not None and hit>break_even),"value":hit,"target":break_even},
        "economic_wilson":{"pass":bool(break_even is not None and wilson>break_even),"value":wilson,"target":break_even},
        "positive_shadow_ev":{"pass":bool(ev is not None and ev>0 and pnl>0),"value":ev,"pnl":pnl,"target":0.0},
        "economic_placebo":{"pass":placebo_econ<0.05,"value":placebo_econ,"target":0.05},
        "calibration":{"pass":bool(ece is not None and ece<=0.03),"value":ece,"target":0.03},
        "regime_coverage":{"pass":regime_samples>=2,"value":regime_samples,"positive_regimes":regime_positive,"target":2}
    }
    core=["forward_sample","economic_wilson","positive_shadow_ev","economic_placebo"]
    core_pass=sum(1 for k in core if gates[k]["pass"])

    if core_pass==len(core):
        status="ECONOMIC_CANDIDATE"
    elif now>=deadline_utc:
        status="NO_CONFIRMED_EDGE_AT_DEADLINE"
    elif days_left<=3 and projected<1000:
        status="DATA_RISK"
    elif n<1000:
        status="COLLECTING"
    else:
        status="RESEARCHING_EDGE"

    priorities=[]
    if n<1000: priorities.append("Accumulate at least 1000 forward V7 signals before any conclusion.")
    if break_even is not None and (hit is None or hit<=break_even): priorities.append("Raise forward hit rate above current payout break-even without reducing validation quality.")
    if break_even is not None and wilson<=break_even: priorities.append("Wilson lower bound is still below economic break-even; more stable evidence is required.")
    if ev is None or ev<=0 or pnl<=0: priorities.append("Shadow EV must turn positive under current payout.")
    if placebo_econ>=.05: priorities.append("Economic placebo probability must fall below 5%.")
    if regime_samples<2: priorities.append("Collect meaningful samples in at least two detected regimes.")
    if ece is None or ece>.03: priorities.append("Improve or verify probability calibration before trusting confidence thresholds.")

    out={
        "version":"7.1-two-week-evidence-sprint",
        "timestamp":int(time.time()),
        "start":"2026-09-24T00:00:00-05:00",
        "deadline":"2026-10-08T23:59:59-05:00",
        "days_left":days_left,
        "hours_left":remaining/3600.0,
        "status":status,
        "core_gates_passed":core_pass,
        "core_gates_total":len(core),
        "signals_per_day":per_day,
        "projected_signals_at_deadline":projected,
        "break_even_rate":break_even,
        "v7_status":v7.get("status","NOT_STARTED"),
        "v61_status":v61.get("status","NOT_STARTED"),
        "v5_promotion":v5.get("promotion"),
        "gates":gates,
        "priorities":priorities,
        "note":"Sprint success means evidence of an economically plausible edge, not a guaranteed profitable system. Real-money execution remains disabled."
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
