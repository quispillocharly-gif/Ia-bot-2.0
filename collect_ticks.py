#!/usr/bin/env python3
import asyncio, json, os
from decimal import Decimal
from pathlib import Path
import websockets

URL="wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOL=os.getenv("SYMBOL","R_75")
COUNT=int(os.getenv("TICK_COUNT","5000"))

async def main():
    rows=[]; end="latest"; pip_size=None
    async with websockets.connect(URL,open_timeout=30) as ws:
        while len(rows)<COUNT:
            req={"ticks_history":SYMBOL,"count":min(1000,COUNT-len(rows)),
                 "end":end,"style":"ticks","req_id":1}
            await ws.send(json.dumps(req))
            data=json.loads(await ws.recv())
            if data.get("error"):
                raise RuntimeError(data["error"].get("message",str(data["error"])))
            h=data.get("history") or {}
            times=h.get("times") or []; prices=h.get("prices") or []
            if not times or len(times)!=len(prices):
                raise RuntimeError("Invalid ticks_history response")
            if data.get("pip_size") is not None:
                pip_size=int(data["pip_size"])
            if pip_size is None:
                raise RuntimeError("pip_size missing; refusing to lose trailing-zero precision")
            for epoch,price in zip(times,prices):
                quote=f"{Decimal(str(price)):.{pip_size}f}"
                rows.append({"epoch":int(epoch),"quote":quote,"digit":int(quote[-1])})
            end=min(map(int,times))-1
    unique={x["epoch"]:x for x in rows}
    ticks=[unique[e] for e in sorted(unique)][-COUNT:]
    counts={str(i):0 for i in range(10)}
    for x in ticks: counts[str(x["digit"])]+=1
    missing=[d for d,n in counts.items() if n==0]
    audit={"symbol":SYMBOL,"pip_size":pip_size,"ticks":len(ticks),
           "digit_counts":counts,"missing_digits":missing}
    Path("ticks.json").write_text(json.dumps({"symbol":SYMBOL,"pip_size":pip_size,"ticks":ticks},indent=2))
    Path("audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    if missing: raise RuntimeError(f"Digit audit failed: missing digits {missing}")

if __name__=="__main__":
    asyncio.run(main())
