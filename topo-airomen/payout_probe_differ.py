#!/usr/bin/env python3
from pathlib import Path
import asyncio, json, time
import websockets

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"differ_payout_snapshot.json"
URL="wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOL="R_75"

async def one(ws,digit):
    req={"proposal":1,"amount":1,"basis":"stake","contract_type":"DIGITDIFF",
         "currency":"USD","duration":1,"duration_unit":"t","barrier":str(digit),
         "underlying_symbol":SYMBOL,"req_id":200+digit}
    await ws.send(json.dumps(req))
    while True:
        msg=json.loads(await ws.recv())
        if msg.get("req_id")!=200+digit: continue
        if msg.get("error"): return {"digit":digit,"error":msg["error"].get("message","proposal error")}
        p=msg.get("proposal") or {}
        try: ask=float(p.get("ask_price"))
        except Exception: ask=None
        try: payout=float(p.get("payout"))
        except Exception: payout=None
        be=(ask/payout) if ask is not None and payout and payout>0 else None
        return {"digit":digit,"ask_price":ask,"payout":payout,"break_even_rate":be,"proposal_id":p.get("id")}

async def main():
    rows=[]
    try:
        async with websockets.connect(URL,open_timeout=20,close_timeout=5,ping_interval=20) as ws:
            for d in range(10): rows.append(await one(ws,d))
    except Exception as e: rows.append({"error":str(e)})
    valid=[x for x in rows if x.get("break_even_rate") is not None]
    bes=[x["break_even_rate"] for x in valid]
    out={"version":"1.0-differ-payout","timestamp":int(time.time()),"symbol":SYMBOL,
         "contract_type":"DIGITDIFF","duration":1,"duration_unit":"t","stake":1.0,
         "status":"OK" if len(valid)==10 else ("PARTIAL" if valid else "ERROR"),
         "valid_digits":len(valid),"per_digit":rows,
         "avg_break_even":sum(bes)/len(bes) if bes else None,
         "min_break_even":min(bes) if bes else None,"max_break_even":max(bes) if bes else None,
         "note":"DIFFER wins when the next digit differs from the chosen barrier. Economic break-even is ask_price/payout and may be above the 90% random hit baseline."}
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__": asyncio.run(main())
