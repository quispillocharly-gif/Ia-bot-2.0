import json, math, urllib.request, time, os
URL="https://api.derivws.com/trading/v1/options/ticks/history"
STATE="randomness-2usd/learned-state.json"
def fetch():
    u=URL+"?ticks_history=R_75&count=5000&end=latest&style=ticks"
    with urllib.request.urlopen(u,timeout=30) as r: data=json.load(r)
    prices=data.get("history",{}).get("prices",[])
    return [int(str(x).replace(".","")[-1]) for x in prices]
def predict(seq,i):
    hist=seq[max(0,i-600):i]
    if len(hist)<120:return None
    scores=[0.0]*10; evidence=[0]*10
    for n in (20,50,100,300):
        z=hist[-min(n,len(hist)):]; c=[z.count(d) for d in range(10)]; sd=math.sqrt(len(z)*.1*.9)
        for d in range(10):
            zz=(c[d]-len(z)*.1)/sd
            if zz<0:scores[d]+=min(3,-zz);evidence[d]+=1
    last=hist[-1]; tr=[[1]*10 for _ in range(10)]
    for a,b in zip(hist,hist[1:]):tr[a][b]+=1
    tot=sum(tr[last])
    for d in range(10):
        p=tr[last][d]/tot
        if p<.1:scores[d]+=(.1-p)*20;evidence[d]+=1
    ranked=sorted(range(10),key=lambda d:scores[d],reverse=True);d=ranked[0]
    return d,scores[d],evidence[d],scores[d]-scores[ranked[1]]
def main():
    seq=fetch(); tested=matches=trades=0; by=[{"n":0,"m":0} for _ in range(10)]
    for i in range(120,len(seq)):
        q=predict(seq,i)
        if not q:continue
        d,s,e,margin=q;tested+=1
        if s>=2.6 and e>=2 and margin>=.15:
            trades+=1;bad=int(seq[i]==d);matches+=bad;by[d]["n"]+=1;by[d]["m"]+=bad
    rate=matches/trades if trades else None
    state={"updated_epoch":int(time.time()),"symbol":"R_75","ticks_analyzed":len(seq),"candidates_tested":tested,"qualified_predictions":trades,"matches":matches,"match_rate":rate,"baseline_match_rate":0.10,"by_digit":by,"method":"walk-forward Z20/50/100/300 + Markov1; no future leakage","guarantee":False}
    with open(STATE,"w") as f:json.dump(state,f,indent=2)
    print(json.dumps(state))
if __name__=="__main__":main()
