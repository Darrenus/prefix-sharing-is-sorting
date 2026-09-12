"""Anneal over instances to maximise OPT_SAV / greedy_SAV.  If this stays pinned at 2,
greedy is plausibly a tight 1/2-approximation for the savings objective."""
import random, itertools
from engine import opt_free, greedy

def sav_pair(R,w):
    tot=sum(sum(w[x] for x in S) for S in R)
    c,s,_=opt_free(R,w)
    return s, tot-greedy(R,w)

rnd=random.Random(int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 0)
overall=(1.0,None)
for n,m,kmax,weighted in [(8,8,4,False),(9,9,5,False),(8,10,4,False),
                          (7,8,4,True),(9,8,5,True),(10,10,5,False)]:
    U=list(range(n))
    w={x:(rnd.choice([1,2,3]) if weighted else 1) for x in U}
    def rnd_req(): 
        k=rnd.randint(2,min(kmax,n)); return frozenset(rnd.sample(U,k))
    cur=sorted({rnd_req() for _ in range(m*3)},key=lambda s:sorted(s))[:m]
    if len(cur)<m: continue
    def score(R):
        try:
            e,g=sav_pair(R,w)
        except Exception: return 0
        return e/g if g>0 else (e if e>0 else 0)
    cs=score(cur); best=(cs,list(cur))
    for it in range(2500):
        cand=list(cur); cand[rnd.randrange(len(cand))]=rnd_req()
        cand=sorted(set(cand),key=lambda s:sorted(s))
        if len(cand)!=m: continue
        ns=score(cand)
        T=0.08*(1-it/2500)+1e-3
        if ns>cs or rnd.random()<pow(2.718,(ns-cs)/T):
            cur,cs=cand,ns
            if ns>best[0]: best=(ns,list(cand))
    e,g=sav_pair(best[1],w)
    print("n=%2d m=%2d kmax=%d %s : worst OPT/greedy = %.4f  (opt=%s greedy=%s)"%(
          n,m,kmax,"weighted" if weighted else "unit  ",best[0],e,g), flush=True)
    print("     witness:", [sorted(x) for x in best[1]], "weights", {k:v for k,v in w.items()} if weighted else "unit", flush=True)
    if best[0]>overall[0]: overall=(best[0],best[1])
print()
print("OVERALL worst OPT_SAV/greedy_SAV found: %.4f"%overall[0])
