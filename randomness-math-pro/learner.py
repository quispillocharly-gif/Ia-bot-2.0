import json, math, urllib.request, time, os
STATE="randomness-math-pro/cloud-state.json"
URL="https://api.derivws.com/trading/v1/options/ticks/history?ticks_history=R_75&count=5000&end=latest&style=ticks"
def fetch():
    with urllib.request.urlopen(URL,timeout=30) as r: data=json.load(r)
    h=data.get("history",{}); prices=h.get("prices",[]); times=h.get("times",[])
    digs=[int(str(x).replace(".","")[-1]) for x in prices]
    return list(zip(times,digs))
def repeat_block(hist):
    a=hist[-20:]
    if not a:return None
    c=[a.count(d) for d in range(10)];m=max(c);leaders=[d for d,v in enumerate(c) if v==m]
    return leaders[0] if m>=3 and len(leaders)==1 else None
def choose(hist,last_bought,discard,step):
    if len(hist)<120:return None
    s=[0.0]*10
    for n in (10,20,40,80):
        a=hist[-n:]; sd=math.sqrt(len(a)*.09)
        for d in range(10):
            z=(a.count(d)-len(a)*.1)/sd
            if z<0:s[d]+=min(2.2,-z)
    r=hist[-200:]; last=r[-1]; tr=[[1]*10 for _ in range(10)]
    for a,b in zip(r,r[1:]):tr[a][b]+=1
    total=sum(tr[last])
    for d in range(10):
        p=tr[last][d]/total
        if p<.1:s[d]+=(.1-p)*15
    rc=hist[-30:]
    for d in range(10):s[d]-=max(0,rc.count(d)-3)*.25
    rb=repeat_block(hist)
    pool=[d for d in range(10) if d!=last_bought and discard[d]<=step and d!=rb]
    if not pool:return None
    return max(pool,key=lambda d:s[d])
def main():
    old={}
    if os.path.exists(STATE):
        try:
            with open(STATE) as f:old=json.load(f)
        except:pass
    rows=fetch(); last_epoch=int(old.get("last_epoch",0)); hist=list(old.get("history",[]))[-2500:]
    watch=set(old.get("watch_digits",[])); rem=list(old.get("discard_remaining",[0]*10)); rem=(rem+[0]*10)[:10]
    discard=[int(x) for x in rem]; last_bought=old.get("last_bought"); wins=int(old.get("sim_wins",0)); matches=int(old.get("sim_matches",0)); processed=0
    new=[(int(ep),int(d)) for ep,d in rows if int(ep)>last_epoch]
    for ep,d in new:
        discard=[max(0,x-1) for x in discard]
        if d in watch:
            watch.remove(d);discard[d]=10
        hist.append(d);hist=hist[-2500:]
        pick=choose(hist,last_bought,discard,0)
        if pick is not None:
            last_bought=pick
            if d!=pick:wins+=1;watch.add(pick)
            else:matches+=1
        last_epoch=ep;processed+=1
    state={"updated_epoch":int(time.time()),"last_epoch":last_epoch,"symbol":"R_75","history":hist[-2500:],"watch_digits":sorted(watch),"discard_remaining":discard,"last_bought":last_bought,"repeat_blocked":repeat_block(hist),"sim_wins":wins,"sim_matches":matches,"new_ticks_processed":processed,"mode":"cloud simulation learning","guarantee":False}
    with open(STATE,"w") as f:json.dump(state,f,separators=(",",":"))
    print(json.dumps({k:v for k,v in state.items() if k!="history"}))
if __name__=="__main__":main()
