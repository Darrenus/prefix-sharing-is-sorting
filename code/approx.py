"""Bounds and algorithms for  SAV(H) = sum over internal nodes of w(intersection of its requests).

Upper bound (proved): every internal node v satisfies w(I(v)) <= w(R_i & R_j) for any i,j drawn
from v's two subtrees; choosing one such cross pair per internal node yields m-1 pairs that form a
SPANNING TREE on the requests.  Hence  SAV <= MaxSpanningTree( w(R_i & R_j) ),
and therefore  COST >= sum_i w(R_i) - MST  -- a poly-time lower bound on the optimum.
"""
import itertools, random
from engine import opt_free, greedy

def wt(A,B,w): return sum(w[x] for x in A & B)

def max_spanning_tree(reqs, w):
    m = len(reqs); INF=-1
    inT=[False]*m; best=[-1]*m; best[0]=0; tot=0
    for _ in range(m):
        u=max((i for i in range(m) if not inT[i]), key=lambda i:best[i])
        inT[u]=True; tot+=max(best[u],0)
        for v in range(m):
            if not inT[v]:
                c=wt(set(reqs[u]),set(reqs[v]),w)
                if c>best[v]: best[v]=c
    return tot

def matching_agglom(reqs, w):
    """merge a maximum-weight matching of clusters each round (log m rounds)"""
    nodes=[set(R) for R in reqs]; sav=0
    while len(nodes)>1:
        idx=list(range(len(nodes)))
        pairs=sorted(((wt(nodes[a],nodes[b],w),a,b) for a,b in itertools.combinations(idx,2)),
                     reverse=True)
        used=set(); new=[]
        for v,a,b in pairs:
            if a in used or b in used: continue
            used|={a,b}; sav+=v; new.append(nodes[a]&nodes[b])
        new += [nodes[i] for i in idx if i not in used]
        if len(new)==len(nodes): break
        nodes=new
    return sav

def sav_exact(reqs,w):
    c,s,_=opt_free(reqs,w); return s
def sav_greedy(reqs,w):
    tot=sum(sum(w[x] for x in R) for R in reqs); return tot-greedy(reqs,w)

rnd=random.Random(5)
print("=== is the agglomerative greedy a constant-factor approximation for SAV? ===")
print("   searching for the largest OPT_SAV / greedy_SAV shortfall")
worst=(1.0,None); worstm=(1.0,None); mstslack=[]
for trial in range(4000):
    n=rnd.randint(4,9); m=rnd.randint(4,9)
    U=list(range(n)); w={x:1 for x in U}
    R=set(); guard=0
    while len(R)<m and guard<300:
        guard+=1; k=rnd.randint(2,min(5,n)); R.add(frozenset(rnd.sample(U,k)))
    R=sorted(R,key=lambda s:sorted(s))
    if len(R)<4 or len(set().union(*R))<3: continue
    e=sav_exact(R,w)
    if e<=0: continue
    g=sav_greedy(R,w); mm=matching_agglom(R,w)
    mst=max_spanning_tree(R,w)
    mstslack.append(mst/e)
    if g>0 and e/g>worst[0]: worst=(e/g,[sorted(x) for x in R],g,e)
    if mm>0 and e/mm>worstm[0]: worstm=(e/mm,[sorted(x) for x in R],mm,e)
print("  worst OPT/greedy   shortfall: %.3f   witness %s (greedy=%s opt=%s)"%(worst[0],worst[1],worst[2],worst[3]))
print("  worst OPT/matching shortfall: %.3f   witness %s (match=%s opt=%s)"%(worstm[0],worstm[1],worstm[2],worstm[3]))
print("  MST upper bound slack MST/OPT_SAV: mean %.3f, max %.3f, min %.3f"%(
      sum(mstslack)/len(mstslack), max(mstslack), min(mstslack)))
