"""Numerical verification of:

  THEOREM.  Agglomerative greedy (merge the pair of clusters with the largest common
  intersection) achieves at least HALF the optimal savings, for arbitrary weights.
  Tight: the |S_x|=2 case is maximum matching, where greedy is exactly 1/2.

  Proof engine:  for every threshold t,   A(t) <= 2 G(t),
  where G(t) = #greedy merges of value >= t and A(t) = #optimal internal nodes of value >= t.
  We verify BOTH the conclusion and the per-threshold inequality.
"""
import random, itertools
from engine import opt_free

def greedy_values(reqs, w):
    nodes={i:set(reqs[i]) for i in range(len(reqs))}
    alive=set(nodes); nxt=len(reqs); vals=[]
    while len(alive)>1:
        best=None
        for a,b in itertools.combinations(sorted(alive),2):
            v=sum(w[x] for x in nodes[a]&nodes[b])
            if best is None or v>best[0]: best=(v,a,b)
        v,a,b=best; vals.append(v)
        nodes[nxt]=nodes[a]&nodes[b]; alive-={a,b}; alive.add(nxt); nxt+=1
    return vals

def opt_values(reqs, w):
    m=len(reqs); _,_,sp=opt_free(reqs,w)
    inter=[None]*(1<<m)
    for B in range(1,1<<m):
        low=B&-B; i=low.bit_length()-1; rest=B^low
        inter[B]=set(reqs[i]) if rest==0 else (inter[rest]&reqs[i])
    vals=[]
    def rec(B):
        if B&(B-1)==0: return
        vals.append(sum(w[x] for x in inter[B]))
        B1=sp[B]; rec(B1); rec(B^B1)
    rec((1<<m)-1)
    return vals

rnd=random.Random(23)
bad_ratio=0; bad_theta=0; tested=0; worst=0.0; worst_inst=None
for trial in range(3000):
    n=rnd.randint(4,10); m=rnd.randint(4,9)
    U=list(range(n)); w={x:rnd.choice([1,1,1,2,3,5]) for x in U}
    R=set(); guard=0
    while len(R)<m and guard<300:
        guard+=1; k=rnd.randint(2,min(6,n)); R.add(frozenset(rnd.sample(U,k)))
    R=sorted(R,key=lambda s:sorted(s))
    if len(R)<4: continue
    tested+=1
    gv=sorted(greedy_values(R,w),reverse=True); ov=sorted(opt_values(R,w),reverse=True)
    G,O=sum(gv),sum(ov)
    if O>0:
        r=O/G if G>0 else float('inf')
        if r>worst: worst=r; worst_inst=([sorted(x) for x in R],dict(w),G,O)
        if G*2 < O - 1e-9: bad_ratio+=1
    for th in sorted(set(ov+gv)):
        if th<=0: continue
        A=sum(1 for v in ov if v>=th); Gt=sum(1 for v in gv if v>=th)
        if A > 2*Gt: bad_theta+=1; print("  THETA VIOLATION", th, A, Gt, [sorted(x) for x in R]); break

print("instances tested: %d" % tested)
print("violations of  SAV_greedy >= SAV_opt / 2 :", bad_ratio)
print("violations of  A(theta) <= 2 G(theta)    :", bad_theta)
print("worst observed OPT/greedy savings ratio  : %.6f  (theory says <= 2)" % worst)
print("   witness:", worst_inst[0] if worst_inst else None)
print()
print("tightness check -- the |S_x|=2 case is maximum matching:")
R=[frozenset({0,1}),frozenset({1,2}),frozenset({2,3}),frozenset({3,4})]
w={x:1 for x in range(5)}
print("   path instance", [sorted(x) for x in R],
      " greedy=",sum(greedy_values(R,w))," opt=",sum(opt_values(R,w)))
