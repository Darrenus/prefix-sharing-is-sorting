"""Batch-level adaptive layout on top of a persistent radix trie.
Realistic: LLM servers already batch. We only reorder within one batch."""
from collections import defaultdict
from sim import layout_agglom, rag_workload, nosh

class Trie:
    def __init__(self): self.root = {}; self.size = 0
    def insert(self, seq):
        node = self.root
        for c in seq:
            if c not in node: node[c] = [0, {}]; self.size += 1
            node[c][0] += 1; node = node[c][1]
    def deepest_match(self, R, beam=24):
        """longest path from root using only chunks of R (each once)."""
        best = ()
        frontier = [((), frozenset(R), self.root)]
        while frontier:
            nxt = []
            for path, rem, node in frontier:
                ext = [c for c in node if c in rem]
                if not ext and len(path) > len(best): best = path
                for c in ext:
                    p2 = path + (c,)
                    if len(p2) > len(best): best = p2
                    nxt.append((p2, rem - {c}, node[c][1]))
            nxt.sort(key=lambda x: -sum(self._cnt(x[2])) if x[2] else 0)
            frontier = nxt[:beam]
        return best

    @staticmethod
    def _cnt(node): return [v[0] for v in node.values()] or [0]

def pop_order(pop): return lambda c: (-pop.get(c,0), c)

def run_stream(reqs, B, mode, pop):
    T = Trie()
    for i in range(0, len(reqs), B):
        batch = reqs[i:i+B]
        placed = [(T.deepest_match(R), R) for R in batch]
        if mode == "cw":                      # CacheWeaver-style: residual in popularity order
            for path, R in placed:
                T.insert(path + tuple(sorted(set(R)-set(path), key=pop_order(pop))))
        elif mode == "batch":                 # ours: cluster residuals inside the batch
            groups = defaultdict(list)
            for path, R in placed: groups[path].append(frozenset(set(R)-set(path)))
            for path, res in groups.items():
                res_ne = [r for r in res if r]
                laid = layout_agglom(res_ne) if len(res_ne) > 1 else \
                       ([tuple(sorted(res_ne[0], key=pop_order(pop)))] if res_ne else [])
                for s in laid: T.insert(path + s)
                for _ in range(len(res)-len(res_ne)): T.insert(path)
    return T.size

def run_global(reqs, pop):
    T = Trie()
    for R in reqs: T.insert(tuple(sorted(R, key=pop_order(pop))))
    return T.size

for seed,(m,N,k,t,z) in enumerate([(3000,1500,8,40,1.1),(3000,1500,16,25,0.8),(3000,3000,10,80,1.3)]):
    reqs,_ = rag_workload(m=m,N=N,k=k,topics=t,zipf=z,seed=seed)
    pop = defaultdict(int)
    for R in reqs:
        for c in R: pop[c]+=1
    base = nosh(reqs)
    g  = run_global(reqs, pop)
    print("\n--- RAG m=%d N=%d k=%d topics=%d zipf=%.1f  (no-sharing=%d) ---"%(m,N,k,t,z,base))
    print("   %-38s %7d  %.3fx" % ("global canonical order", g, g/base))
    for B in [32, 128, 512]:
        cw = run_stream(reqs, B, "cw",    pop)
        ou = run_stream(reqs, B, "batch", pop)
        print("   batch=%-4d  deepest-match+pop %6d (%.3fx) | ONLINE BATCH-ADAPTIVE %6d (%.3fx)  -> %+.1f%% vs global"
              % (B, cw, cw/base, ou, ou/base, -100*(g-ou)/g))
