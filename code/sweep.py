import sys, random
sys.path.insert(0,'code')
from collections import defaultdict
from phase2 import run, agglom_savings, wsum
from beir import retrieve
from engine import opt_free

def exact_on_real_clusters(ds, k, max_q, trials=60, size=13, seed=0):
    """sample REAL overlapping request groups and compare greedy to the exact optimum"""
    reqs, ranks, w, nd = retrieve(ds, k, max_queries=max_q)
    inv=defaultdict(list)
    for i,R in enumerate(reqs):
        for x in R: inv[x].append(i)
    hubs=[x for x,S in inv.items() if len(S)>=4]
    rnd=random.Random(seed); ratios=[]
    for _ in range(trials):
        if not hubs: break
        x=rnd.choice(hubs); grp=inv[x][:]
        rnd.shuffle(grp); grp=grp[:size]
        sub=[reqs[i] for i in grp]
        sub=list(dict.fromkeys(sub))
        if len(sub)<4: continue
        U=sorted(set().union(*sub)); idx={c:j for j,c in enumerate(U)}
        S=[frozenset(idx[c] for c in R) for R in sub]
        ww={idx[c]:w[c] for c in U}
        cost,_,_=opt_free(S,ww)
        tot=sum(wsum(R,ww) for R in S)
        g=tot-agglom_savings(S,ww)
        if cost>0: ratios.append(g/cost)
    return ratios

if __name__=="__main__":
    for ds,mq in [("nfcorpus",None),("scifact",None),("fiqa",2500)]:
        for k in [5,8,16]:
            try: run(ds,k,max_q=mq)
            except Exception as e: print("  skip",ds,k,repr(e)[:80])
    print("\n=== greedy vs EXACT optimum on real overlapping request groups (m<=13) ===")
    for ds in ["nfcorpus","scifact"]:
        r=exact_on_real_clusters(ds,8,None)
        if r: print("  %-10s k=8 : greedy is %.2f%% above optimum on average (max %.2f%%), %d groups"
                    %(ds,100*(sum(r)/len(r)-1),100*(max(r)-1),len(r)))
