#!/usr/bin/env python3
from pathlib import Path
import json, time

R=Path(__file__).resolve().parent; M=R/"memory"; M.mkdir(exist_ok=True); OUT=M/"differ_director_latest.json"
def load(name):
    try:return json.loads((M/name).read_text())
    except Exception:return {}
def main():
    train=load("differ_train_latest.json"); fwd=load("differ_forward_latest.json"); pay=load("differ_payout_snapshot.json")
    L=fwd.get("leader") or {}; confirmed=fwd.get("confirmed_variants") or []
    if confirmed: decision="FORWARD_ECONOMIC_CANDIDATE"
    elif int(L.get("signals",0) or 0)<2000: decision="COLLECT_FORWARD_DATA"
    else: decision="NO_CONFIRMED_EDGE_CONTINUE_RESEARCH"
    out={"version":"1.0-differ-director","timestamp":int(time.time()),"decision":decision,"baseline_random":.90,
         "payout_status":pay.get("status","MISSING"),"break_even_rate":pay.get("max_break_even"),
         "forward_status":fwd.get("status","NOT_STARTED"),"leader":L,"confirmed_variants":confirmed,
         "candidate_model_id":(train.get("candidate") or {}).get("model_id"),
         "note":"High DIFFER hit rate alone is not enough. Decisions use economic break-even and forward evidence."}
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
