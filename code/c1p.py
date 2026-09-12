"""If {S_x} has the consecutive-ones property, requests can be laid on a line so that every chunk
is wanted by a contiguous run.  A balanced tree over that order makes each S_x an interval, whose
canonical decomposition has <= 2*ceil(log2 m) nodes -- the segment-tree bound.  So

    OPT_free  <=  sum_x w(x) * min(|S_x|, 2*ceil(log2 m)).

Question we can also ask of real data: how close is real retrieval to this bound?"""
import sys, math, random
sys.path.insert(0,'code')
from engine import opt_free

def balanced_decomp(m, S):
    """canonical decomposition size of set S (as leaf indices) in the balanced tree on [0,m)"""
    def rec(lo,hi):
        if lo>=hi: return 0
        inside=all(i in S for i in range(lo,hi))
        if inside: return 1
        if not any(i in S for i in range(lo,hi)): return 0
        mid=(lo+hi)//2
        return rec(lo,mid)+rec(mid,hi)
    return rec(0,m)

def c1p_instance(m, nchunks, rnd, maxlen=None):
    """requests 0..m-1 on a line; each chunk wanted by a random contiguous run"""
    S={}
    for x in range(nchunks):
        L=rnd.randint(2, maxlen or m)
        a=rnd.randint(0, m-L); S[x]=set(range(a,a+L))
    reqs=[frozenset(x for x in S if i in S[x]) for i in range(m)]
    return reqs, S

rnd=random.Random(4)
print("=== C1P instances: exact optimum vs the segment-tree bound ===")
print("   m  chunks | OPT_free | sum_x min(|S_x|, 2*ceil(lg m)) | sum_x 1 (absolute floor)")
for m,nc in [(8,8),(10,12),(12,14),(14,16),(16,18)]:
    tot=0; ok=0
    for t in range(6):
        reqs,S=c1p_instance(m,nc,rnd)
        reqs=[r for r in reqs if r]
        if len(reqs)<4: continue
        idx={i:j for j,i in enumerate(range(len(reqs)))}
        c,_,_=opt_free(reqs)
        bound=sum(min(len(S[x]), 2*math.ceil(math.log2(m))) for x in S if S[x])
        floor=len([x for x in S if S[x]])
        bal=sum(balanced_decomp(m,S[x]) for x in S)
        tot+=1
        if t==0: print("  %3d %6d  | %8d | %28d | %8d   (balanced-tree actual: %d)"%(m,nc,c,bound,floor,bal))
print()
print("=== how close is REAL retrieval to the segment-tree regime? ===")
from beir import retrieve
from phase2 import wsum, agglom_savings
for ds,k in [("nfcorpus",8),("scifact",8),("fiqa",8)]:
    reqs,ranks,w,nd=retrieve(ds,k,max_queries=2500)
    m=len(reqs)
    Sx={}
    for i,R in enumerate(reqs):
        for x in R: Sx.setdefault(x,set()).add(i)
    tot=sum(wsum(R,w) for R in reqs)
    ours=tot-agglom_savings(reqs,w)
    floor=sum(w[x] for x in Sx)                                  # every chunk at least once
    segbound=sum(w[x]*min(len(Sx[x]), 2*math.ceil(math.log2(m))) for x in Sx)
    print("  %-9s m=%4d: floor %9d | ours %9d (%.2fx floor) | no-sharing %9d (%.2fx floor) | seg-tree bound %9d"
          %(ds,m,floor,ours,ours/floor,tot,tot/floor,segbound))
