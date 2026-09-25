#!/usr/bin/env python3
from pathlib import Path
import asyncio, json, os, time
import websockets

R=Path(__file__).resolve().parent
OUT=R/"ticks.json"
AUDIT=R/"audit.json"
URL="wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOL=os.getenv("SYMBOL","R_75")
COUNT=max(1000,min(10000,int(os.getenv("TICK_COUNT","10000"))))

def last_digit(price,pip):
    z=f"{float(price):.{int(pip)}f}"
    return int(z[-1])

async def main():
    req={"ticks_history":SYMBOL,"count":COUNT,"end":"latest","style":"ticks","req_id":1}
    async with websockets.connect(URL,open_timeout=20,close_timeout=5,ping_interval=20) as ws:
        await ws.send(json.dumps(req))
        while True:
            msg=json.loads(await ws.recv())
            if msg.get("error"): raise RuntimeError(msg["error"].get("message","ticks error"))
            if msg.get("history"):
                h=msg["history"]; prices=h.get("prices",[]); times=h.get("times",[])
                pip=int(msg.get("pip_size",4) or 4)
                ticks=[{"epoch":int(t),"quote":float(q),"digit":last_digit(q,pip)} for t,q in zip(times,prices)]
                ticks.sort(key=lambda x:x["epoch"])
                OUT.write_text(json.dumps({"symbol":SYMBOL,"timestamp":int(time.time()),"ticks":ticks},indent=2))
                epochs=[x["epoch"] for x in ticks]
                gaps=[b-a for a,b in zip(epochs[:-1],epochs[1:])]
                counts=[0]*10
                for x in ticks: counts[int(x["digit"])]+=1
                audit={
                    "timestamp":int(time.time()),"symbol":SYMBOL,"ticks":len(ticks),
                    "duplicate_epochs":len(epochs)-len(set(epochs)),
                    "min_epoch":min(epochs) if epochs else None,"max_epoch":max(epochs) if epochs else None,
                    "max_gap_seconds":max(gaps) if gaps else None,"digit_counts":counts
                }
                AUDIT.write_text(json.dumps(audit,indent=2))
                print(json.dumps(audit,indent=2)); return

if __name__=="__main__": asyncio.run(main())
