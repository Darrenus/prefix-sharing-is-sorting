import random
from collections import defaultdict
from sim import trie_cost, nosh, layout_agglom, rag_workload

def order_trie_from_layout(seqs):
    """turn a set of laid-out sequences into a policy trie: prefix -> ranked next chunks"""
    T = {}
    for s in seqs:
        node = T
        for c in s:
            if c not in node: node[c] = [0, {}]
            node[c][0] += 1
            node = node[c][1]
    return T

def place(R, T):
    """descend the policy trie greedily; fall back to popularity order"""
    node = T; rem = set(R); path = []
    while rem:
        cands = [c for c in node if c in rem]
        if not cands: break
        c = max(cands, key=lambda c:(node[c][0], -c))
        path.append(c); rem.discard(c); node = node[c][1]
    return tuple(path) + tuple(sorted(rem))

def run(reqs, warm_frac=0.3, label=""):
    W = int(len(reqs)*warm_frac)
    warm, test = reqs[:W], reqs[W:]
    pop = defaultdict(int)
    for R in reqs:
        for c in R: pop[c]+=1
    gk = lambda c: (-pop[c], c)

    # A: single global canonical order (production practice)
    A = [tuple(sorted(R, key=gk)) for R in reqs]
    # B: pure online greedy longest-prefix over the live trie
    from sim import layout_online_lpm
    B = layout_online_lpm(reqs)
    # C: ours -- learn an adaptive policy from the warmup, apply online to the rest
    warm_layout = layout_agglom(warm)
    T = order_trie_from_layout(warm_layout)
    C = list(warm_layout) + [place(R, T) for R in test]
    # D: ours, full offline (upper limit of the method)
    D = layout_agglom(reqs)

    base = nosh(reqs)
    print("\n--- %s  (m=%d, no-sharing=%d chunk-slots) ---" % (label, len(reqs), base))
    for nm, L in [("global canonical order", A), ("online greedy LPM", B),
                  ("ONLINE adaptive policy (ours)", C), ("offline hierarchy (ours, bound)", D)]:
        c = trie_cost(L)
        print("   %-32s %7d   %.3fx no-sharing" % (nm, c, c/base))
    a = trie_cost(A); c = trie_cost(C)
    print("   -> online ours saves %.1f%% of prefill vs global canonical order" % (100*(a-c)/a))

for seed,(m,N,k,t,z) in enumerate([(3000,1500,8,40,1.1),(3000,1500,16,25,0.8),(3000,3000,10,80,1.3)]):
    reqs,_ = rag_workload(m=m,N=N,k=k,topics=t,zipf=z,seed=seed)
    run(reqs, label="RAG m=%d N=%d k=%d topics=%d zipf=%.1f"%(m,N,k,t,z))
