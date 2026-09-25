#!/usr/bin/env python3
from pathlib import Path
import json, time

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"differ_director_latest.json"
POLICY=M/"differ_live_policy.json"

def load(name):
    try:return json.loads((M/name).read_text())
    except Exception:return {}

def viable(row,break_even):
    if not row:return False
    rate=row.get("hit_rate"); pnl=row.get("shadow_pnl")
    return bool(rate is not None and break_even is not None and rate>break_even and pnl is not None and pnl>0)

def main():
    train=load("differ_train_latest.json")
    v1=load("differ_forward_latest.json")
    v2=load("differ_v2_latest.json")
    v4=load("differ_v4_batch5_latest.json")
    pay=load("differ_payout_snapshot.json")
    be=pay.get("max_break_even")
    l1=v1.get("leader") or {}
    l2=v2.get("leader") or {}
    c1=v1.get("confirmed_variants") or []
    c2=v2.get("confirmed_variants") or []

    if c2:
        decision="V2_FORWARD_ECONOMIC_CANDIDATE"
    elif c1:
        decision="V1_FORWARD_ECONOMIC_CANDIDATE"
    elif int(l2.get("signals",0) or 0)<2000:
        decision="COLLECT_V2_FORWARD_DATA"
    else:
        decision="NO_CONFIRMED_EDGE_CONTINUE_RESEARCH"

    n1=int(l1.get("signals",0) or 0); n2=int(l2.get("signals",0) or 0)
    use_v2=bool(c2)
    if not use_v2 and n2>=300 and viable(l2,be):
        if not viable(l1,be) or float(l2.get("wilson_lower",0) or 0)>=float(l1.get("wilson_lower",0) or 0):
            use_v2=True
    if not use_v2 and n2>=600 and not viable(l1,be):
        use_v2=float(l2.get("wilson_lower",0) or 0)>float(l1.get("wilson_lower",0) or 0)

    if use_v2:
        source="v2"; rule=l2.get("id"); leader=l2
        confirmed=rule in c2
        candidate_id=v2.get("ensemble_id")
    else:
        source="v1"; rule=l1.get("id","neural_argmin"); leader=l1
        confirmed=rule in c1
        candidate_id=v1.get("model_id")

    economic_ok=viable(leader,be)
    money=v4.get("leader") if v4.get("status")=="POLICY_CANDIDATE" else None
    if money:
        money={
          "source":money.get("source"),"policy":money.get("policy"),
          "compound_fraction":money.get("compound_fraction",0.0),
          "pause_after_match":money.get("pause_after_match",0),
          "batch_size":money.get("batch_size",5),
          "stage_size":money.get("stage_size",5.0),
          "sessions_completed":money.get("sessions_completed",0),
          "target_rate":money.get("target_rate"),
          "promoted":True
        }
    policy={
      "version":"2.0-live-policy","timestamp":int(time.time()),
      "source":source,"rule":rule,"candidate_id":candidate_id,
      "mode":"CONFIRMED_DEMO" if confirmed else "RESEARCH_DEMO",
      "confirmed":confirmed,"economic_qualified":economic_ok,
      "break_even_rate":be,
      "leader_signals":leader.get("signals",0),
      "leader_hit_rate":leader.get("hit_rate"),
      "leader_wilson":leader.get("wilson_lower"),
      "leader_shadow_pnl":leader.get("shadow_pnl"),
      "money_policy":money,
      "note":"Demo-only policy. Research continues in parallel. The DIFFER digit is recalculated across 0-9 every tick; no digit is frozen. Money-management policy remains session-scoped."
    }
    POLICY.write_text(json.dumps(policy,indent=2))

    out={
      "version":"2.0-differ-director","timestamp":int(time.time()),"decision":decision,
      "baseline_random":.90,"payout_status":pay.get("status","MISSING"),"break_even_rate":be,
      "v1_status":v1.get("status","NOT_STARTED"),"v2_status":v2.get("status","NOT_STARTED"),"v4_status":v4.get("status","NOT_STARTED"),
      "v1_leader":l1,"v2_leader":l2,"v4_leader":v4.get("leader"),
      "v1_confirmed":c1,"v2_confirmed":c2,
      "candidate_model_id":(train.get("candidate") or {}).get("model_id"),
      "ensemble_id":(train.get("ensemble_candidate") or {}).get("ensemble_id"),
      "live_policy":policy,
      "note":"High DIFFER hit rate alone is not enough. Director prioritizes economic forward evidence and keeps live execution demo-only."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
