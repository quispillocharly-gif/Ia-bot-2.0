#!/usr/bin/env python3
from pathlib import Path
import asyncio, json, time
import websockets

R=Path(__file__).resolve().parent
M=R/"memory"; M.mkdir(exist_ok=True)
OUT=M/"payout_snapshot.json"
URL="wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOL="R_75"

async def one(ws,digit):
    req={
        "proposal":1,"amount":1,"basis":"stake","contract_type":"DIGITMATCH",
        "currency":"USD","duration":1,"duration_unit":"t","barrier":str(digit),
        "underlying_symbol":SYMBOL,"subscribe":0,"req_id":100+digit
    }
    await ws.send(json.dumps(req))
    while True:
        msg=json.loads(await ws.recv())
        if msg.get("req_id")!=100+digit:
            continue
        if msg.get("error"):
            first_error=msg["error"].get("message","proposal error")
            if "underlying_symbol" in first_error.lower() and ("not allowed" in first_error.lower() or "unknown" in first_error.lower()):
                old=dict(req); old.pop("underlying_symbol",None); old["symbol"]=SYMBOL
                await ws.send(json.dumps(old))
                while True:
                    msg=json.loads(await ws.recv())
                    if msg.get("req_id")==100+digit:break
            else:
                return {"digit":digit,"error":first_error,"request_schema":"new"}
        if msg.get("error"):
            return {"digit":digit,"error":msg["error"].get("message","proposal error"),"request_schema":"legacy-fallback"}
        p=msg.get("proposal") or {}
        ask=p.get("ask_price"); payout=p.get("payout")
        try:ask=float(ask)
        except Exception:ask=None
        try:payout=float(payout)
        except Exception:payout=None
        be=(ask/payout) if ask is not None and payout and payout>0 else None
        return {"digit":digit,"ask_price":ask,"payout":payout,"break_even_rate":be,"proposal_id":p.get("id")}

async def main():
    rows=[]
    try:
        async with websockets.connect(URL,open_timeout=20,close_timeout=5,ping_interval=20) as ws:
            for d in range(10):
                rows.append(await one(ws,d))
    except Exception as e:
        rows.append({"error":str(e)})
    valid=[x for x in rows if x.get("break_even_rate") is not None]
    status="OK" if len(valid)==10 else ("PARTIAL" if valid else "ERROR")
    bes=[x["break_even_rate"] for x in valid]
    out={
        "version":"1.0-match-payout-probe","timestamp":int(time.time()),"symbol":SYMBOL,
        "contract_type":"DIGITMATCH","duration":1,"duration_unit":"t","stake":1.0,
        "status":status,"valid_digits":len(valid),"per_digit":rows,
        "avg_break_even":sum(bes)/len(bes) if bes else None,
        "min_break_even":min(bes) if bes else None,
        "max_break_even":max(bes) if bes else None,
        "note":"Break-even rate is ask_price / payout for a 1-tick DIGITMATCH proposal. Payouts can change; this snapshot is informational and is refreshed periodically."
    }
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__":asyncio.run(main())
