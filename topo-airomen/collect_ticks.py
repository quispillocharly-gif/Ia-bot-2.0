#!/usr/bin/env python3
import asyncio, json, os, time
from decimal import Decimal
from pathlib import Path
import websockets

R=Path(__file__).resolve().parent
URL="wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOL=os.getenv("SYMBOL","R_75")
COUNT=max(1000,min(10000,int(os.getenv("TICK_COUNT","10000"))))

async def main():
    rows=[]; end="latest"; pip_size=None
    async with websockets.connect(URL,open_timeout=30,close_timeout=5,ping_interval=20) as ws:
        while len(rows)<COUNT:
            req={"ticks_history":SYMBOL,"count":min(1000,COUNT-len(rows)),"end":end,"style":"ticks","req_id":1}
            await ws.send(json.dumps(req))
            data=json.loads(await ws.recv())
            if data.get("error"): raise RuntimeError(data["error"].get("message",str(data["error"])))
            h=data.get("history") or {}; times=h.get("times") or []; prices=h.get("prices") or []
            if not times or len(times)!=len(prices): raise RuntimeError("Invalid ticks_history response")
            if data.get("pip_size") is not None: pip_size=int(data["pip_size"])
            if pip_size is None: raise RuntimeError("pip_size missing")
            for epoch,price in zip(times,prices):
                quote=f"{Decimal(str(price)):.{pip_size}f}"
                rows.append({"epoch":int(epoch),"quote":quote,"digit":int(quote[-1])})
            end=min(map(int,times))-1
    unique={x["epoch"]:x for x in rows}
    ticks=[unique[e] for e in sorted(unique)][-COUNT:]
    counts=[0]*10
    for x in ticks: counts[int(x["digit"])]+=1
    epochs=[x["epoch"] for x in ticks]
    gaps=[b-a for a,b in zip(epochs[:-1],epochs[1:])]
    audit={"timestamp":int(time.time()),"symbol":SYMBOL,"pip_size":pip_size,"ticks":len(ticks),
           "duplicate_epochs":len(epochs)-len(set(epochs)),"digit_counts":counts,
           "min_epoch":min(epochs) if epochs else None,"max_epoch":max(epochs) if epochs else None,
           "max_gap_seconds":max(gaps) if gaps else None}
    (R/"ticks.json").write_text(json.dumps({"symbol":SYMBOL,"pip_size":pip_size,"ticks":ticks},indent=2))
    (R/"audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    if len(ticks)<COUNT: raise RuntimeError(f"Expected {COUNT} unique ticks, got {len(ticks)}")
    if any(n==0 for n in counts): raise RuntimeError("Digit audit failed: missing digit")

if __name__=="__main__": asyncio.run(main())
