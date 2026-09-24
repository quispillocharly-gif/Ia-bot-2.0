#!/usr/bin/env python3
# Hourly final coordinator: evaluates the freshest outputs from all research stages.
from pathlib import Path
import json, math, time, hashlib

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"v6_director_latest.json"
REG=M/"v6_registry.json"
GRAVE=M/"v6_graveyard.json"
BASE=.10

def load(path,default=None):
    try:return json.loads(path.read_text())
    except Exception:return default if default is not None else {}

def one_sided_p(w,n,p=BASE):
    if not n:return 1.0
    mu=n*p; sd=math.sqrt(n*p*(1-p))
    if sd<=0:return 1.0
    z=(w-0.5-mu)/sd
    return max(0.0,min(1.0,0.5*math.erfc(z/math.sqrt(2))))

def adjusted_p(w,n,tests):
    return min(1.0,one_sided_p(w,n)*max(1,int(tests)))

def signature(window,hidden,threshold):
    h="x".join(str(x) for x in (hidden or [])) or "none"
    return f"w{window}-h{h}-t{float(threshold or 0):.3f}"

def main():
    v4=load(M/"research_v4_latest.json")
    v41=load(M/"v4_1_confirmation_latest.json")
    v5=load(M/"neural_v5_latest.json")
    v51=load(M/"v5_1_confirmation_latest.json")
    cand=load(M/"neural_v5_candidate.json")
    ens=load(M/"v6_ensemble_latest.json")
    payout=load(M/"payout_snapshot.json")
    adaptive=load(M/"v7_adaptive_latest.json")

    v4l=v4.get("leader") or {}
    v5l=v5.get("leader") or {}
    tests_v4=64
    tests_v5=max(1,len(v5.get("selection_ranking") or [])*9)

    v4_adj=adjusted_p(int(v4l.get("wins",0)),int(v4l.get("predictions",0)),tests_v4)
    v5_adj=adjusted_p(int(v5l.get("wins",0)),int(v5l.get("predictions",0)),tests_v5)
    v41_p=one_sided_p(int(v41.get("wins",0)),int(v41.get("predictions",0)))
    v51_p=one_sided_p(int(v51.get("wins",0)),int(v51.get("predictions",0)))
    ens_p=one_sided_p(int(ens.get("wins",0)),int(ens.get("predictions",0))) if ens else 1.0

    hist_rate=cand.get("hit_rate")
    fut_rate=v51.get("hit_rate")
    fut_n=int(v51.get("predictions",0) or 0)
    gap=None if hist_rate is None or fut_rate is None else float(fut_rate)-float(hist_rate)
    drift="COLLECTING"
    if fut_n>=500:
        if gap is not None and gap<=-0.015:drift="DRIFT_ALERT"
        elif float(fut_rate or 0)<BASE:drift="DRIFT_WATCH"
        else:drift="STABLE"

    grave=load(GRAVE,{"entries":[]})
    entries=grave.setdefault("entries",[])
    if v51.get("status")=="NOT_CONFIRMED" and v51.get("model_id"):
        mid=v51.get("model_id")
        sig=signature(v51.get("window"),v51.get("hidden"),v51.get("confidence_threshold",0))
        if not any(x.get("model_id")==mid for x in entries):
            entries.append({
                "model_id":mid,"signature":sig,"reason":"FUTURE_NOT_CONFIRMED",
                "failed_at":int(time.time()),"future_predictions":int(v51.get("predictions",0)),
                "future_hit_rate":v51.get("hit_rate"),
                "blocked_until_v5_run":int(v5.get("runs",0))+3
            })
    grave["entries"]=entries[-60:]
    GRAVE.write_text(json.dumps(grave,indent=2))

    forward_confirmed=[]
    confirmed_objs=[]
    for name,obj in (("V4.1",v41),("V5.2",v51),("V6.1",ens)):
        if obj and obj.get("status")=="CONFIRMED_EDGE":
            forward_confirmed.append(name); confirmed_objs.append(obj)
    if adaptive.get("status")=="ECONOMICALLY_CONFIRMED":
        forward_confirmed.append("V7")
        confirmed_objs.append(adaptive)

    break_even=payout.get("max_break_even") if payout.get("status")=="OK" else None
    if break_even is None:
        economic_status="PAYOUT_UNAVAILABLE"
    elif not confirmed_objs:
        economic_status="PAYOUT_READY_FORWARD_NOT_CONFIRMED"
    elif any(float(x.get("wilson_lower",0) or 0)>float(break_even) for x in confirmed_objs):
        economic_status="ECONOMIC_EDGE_CANDIDATE"
    else:
        economic_status="STAT_EDGE_BELOW_BREAK_EVEN"

    if forward_confirmed:
        decision="FORWARD_EDGE_SIGNAL_DETECTED"
    elif v41.get("status")=="NOT_CONFIRMED" and v51.get("status")=="NOT_CONFIRMED":
        decision="NO_CONFIRMED_EDGE_CONTINUE_RESEARCH"
    else:
        decision="COLLECT_MORE_FORWARD_DATA"

    priorities=[]
    if v41.get("status")=="NOT_CONFIRMED":
        priorities.append("Do not reuse the rejected V4.1 transition_3 candidate without new independent evidence.")
    if v51.get("status")=="NOT_CONFIRMED":
        priorities.append("Rotate from the failed neural candidate after cooldown and test a materially different challenger.")
    elif int(v51.get("predictions",0) or 0)<1000:
        priorities.append("Continue frozen neural forward confirmation to at least 1000 emitted signals.")
    if v5_adj>=.05:
        priorities.append("Treat V5 historical gains as exploratory after multiple-testing correction.")
    if drift=="DRIFT_ALERT":
        priorities.append("Quarantine neural promotion while future performance is materially below historical performance.")
    if not ens:
        priorities.append("Start V6.1 forward ensemble validation.")
    if adaptive:
        if adaptive.get("status")=="COLLECTING":
            priorities.append("Continue V7 regime/calibration/shadow EV collection without changing live execution.")
        elif adaptive.get("status")=="NOT_CONFIRMED":
            priorities.append("V7 adaptive shadow strategy did not confirm economic edge; use its regime and calibration diagnostics only.")
    if break_even is None:
        priorities.append("Payout probe is unavailable; do not claim profitability.")
    elif not confirmed_objs:
        priorities.append(f"Current conservative MATCH break-even is {break_even:.4f}; wait for forward confirmation before economic evaluation.")
    elif economic_status!="ECONOMIC_EDGE_CANDIDATE":
        priorities.append(f"Forward evidence has not cleared the current conservative break-even rate {break_even:.4f}.")

    snapshot={
        "timestamp":int(time.time()),"decision":decision,"drift":drift,
        "v4":{"model":v4l.get("model"),"rate":v4l.get("hit_rate"),"wilson":v4l.get("wilson_lower"),"adjusted_p":v4_adj},
        "v41":{"predictions":v41.get("predictions"),"rate":v41.get("hit_rate"),"status":v41.get("status"),"p":v41_p},
        "v5":{"challenger":v5.get("challenger_model_id"),"candidate":v5.get("candidate_model_id"),"rate":v5l.get("hit_rate"),"wilson":v5l.get("wilson_lower"),"coverage":v5l.get("coverage"),"adjusted_p":v5_adj,"promotion":v5.get("promotion")},
        "v51":{"model_id":v51.get("model_id"),"predictions":v51.get("predictions"),"rate":v51.get("hit_rate"),"coverage":v51.get("coverage"),"status":v51.get("status"),"p":v51_p},
        "v61":{"predictions":ens.get("predictions") if ens else 0,"rate":ens.get("hit_rate") if ens else None,"status":ens.get("status") if ens else "NOT_STARTED","p":ens_p},
        "v7":{"predictions":adaptive.get("predictions") if adaptive else 0,"rate":adaptive.get("hit_rate") if adaptive else None,
              "wilson":adaptive.get("wilson_lower") if adaptive else None,"status":adaptive.get("status") if adaptive else "NOT_STARTED",
              "shadow_pnl":(adaptive.get("economic") or {}).get("shadow_pnl") if adaptive else None,
              "placebo_economic_p":(adaptive.get("placebo") or {}).get("economic_break_even_p") if adaptive else None}
    }
    snapshot["experiment_id"]=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()[:16]
    reg=load(REG,{"entries":[],"total_recorded":0})
    reg["entries"].append(snapshot); reg["entries"]=reg["entries"][-200:]
    reg["total_recorded"]=int(reg.get("total_recorded",0))+1
    REG.write_text(json.dumps(reg,indent=2))

    out={
        "version":"6.0-autonomous-research-director","timestamp":int(time.time()),
        "decision":decision,"forward_confirmed_systems":forward_confirmed,
        "economic_status":economic_status,"payout_status":payout.get("status","MISSING"),
        "break_even_rate":break_even,"drift_status":drift,
        "historical_to_future_gap":gap,
        "multiple_testing":{"v4":{"tests":tests_v4,"adjusted_p":v4_adj},"v5":{"tests":tests_v5,"adjusted_p":v5_adj}},
        "forward_evidence":{"v4_1_p":v41_p,"v5_2_p":v51_p,"v6_1_p":ens_p,
                            "v7_economic_placebo_p":(adaptive.get("placebo") or {}).get("economic_break_even_p") if adaptive else None},
        "adaptive_v7":{"status":adaptive.get("status","NOT_STARTED") if adaptive else "NOT_STARTED",
                       "current_regime":adaptive.get("current_regime") if adaptive else None,
                       "shadow_pnl":(adaptive.get("economic") or {}).get("shadow_pnl") if adaptive else None,
                       "ev_per_signal":(adaptive.get("economic") or {}).get("ev_per_signal") if adaptive else None,
                       "calibration_ece":(adaptive.get("calibration") or {}).get("ece") if adaptive else None},
        "registry_total":reg["total_recorded"],"graveyard_size":len(grave["entries"]),
        "priorities":priorities,
        "note":"V6 records evidence, corrects exploratory multiple testing, monitors drift and failed candidates, and does not equate hit rate with profitability."
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
