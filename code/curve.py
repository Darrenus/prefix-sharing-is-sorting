from collections import defaultdict
from sim import layout_agglom, rag_workload, nosh
from online2 import Trie, pop_order

def run(reqs, B, pop, tau):
    """persistent radix trie; within each batch, adaptively lay out residual groups
       of size >= tau, otherwise keep the global convention."""
    T = Trie()
    for i in range(0, len(reqs), B):
        placed = [(T.deepest_match(R), R) for R in reqs[i:i+B]]
        groups = defaultdict(list)
        for path, R in placed: groups[path].append(frozenset(set(R)-set(path)))
        for path, res in groups.items():
            ne = [r for r in res if r]
            if len(ne) >= tau:
                laid = layout_agglom(ne)
            else:
                laid = [tuple(sorted(r, key=pop_order(pop))) for r in ne]
            for s in laid: T.insert(path + s)
            for _ in range(len(res)-len(ne)): T.insert(path)
    return T.size

cfgs = [(4000,1500,8,40,1.1),(4000,1500,16,25,0.8),(4000,3000,10,80,1.3)]
print("savings in prefill tokens vs. a single global canonical order")
print("lookahead W:      1      8     32    128    512   2048   all(offline)")
for seed,(m,N,k,t,z) in enumerate(cfgs):
    reqs,_ = rag_workload(m=m,N=N,k=k,topics=t,zipf=z,seed=seed)
    pop = defaultdict(int)
    for R in reqs:
        for c in R: pop[c]+=1
    T = Trie()
    for R in reqs: T.insert(tuple(sorted(R, key=pop_order(pop))))
    g = T.size
    row = []
    for B in [1,8,32,128,512,2048,m]:
        best = min(run(reqs, B, pop, tau) for tau in (2,8,32))
        row.append(100*(g-best)/g)
    print("k=%-2d topics=%-2d  " % (k,t) + "  ".join("%5.1f%%"%x for x in row))
