import random, heapq, math
from collections import defaultdict

# ---------- cost ----------
def trie_cost(seqs, w=None):
    P = set()
    for s in seqs:
        for j in range(1, len(s)+1): P.add(s[:j])
    return len(P) if w is None else sum(w[p[-1]] for p in P)

def nosh(reqs, w=None):
    return sum(len(R) for R in reqs) if w is None else sum(sum(w[c] for c in R) for R in reqs)

# ---------- baseline 1: one global canonical order ----------
def layout_global(reqs, key):
    return [tuple(sorted(R, key=key)) for R in reqs]

# ---------- baseline 2: level-wise greedy hitting set ----------
def layout_greedy_hit(reqs):
    out = {}
    def rec(items, prefix):
        items = [(i, R) for (i, R) in items]
        done = [(i,R) for i,R in items if not R]
        for i,_ in done: out[i] = prefix
        items = [(i,R) for i,R in items if R]
        if not items: return
        rem = list(items)
        while rem:
            cnt = defaultdict(int)
            for _,R in rem:
                for c in R: cnt[c]+=1
            v = max(cnt, key=lambda c:(cnt[c], -c))
            grp = [(i,R-{v}) for i,R in rem if v in R]
            rem = [(i,R) for i,R in rem if v not in R]
            rec(grp, prefix+(v,))
    rec(list(enumerate(reqs)), ())
    return [out[i] for i in range(len(reqs))]

# ---------- proposed: agglomerative common-prefix hierarchy ----------
def layout_agglom(reqs):
    n = len(reqs)
    nodes = {i: (reqs[i], [i]) for i in range(n)}      # id -> (intersection, members)
    alive = set(range(n)); nxt = n
    children = {}
    # priority queue on -|intersection|
    pq = []
    ids = sorted(alive)
    for a in range(len(ids)):
        for b in range(a+1, len(ids)):
            i,j = ids[a], ids[b]
            heapq.heappush(pq, (-len(nodes[i][0] & nodes[j][0]), i, j))
    while len(alive) > 1:
        while True:
            negs, i, j = heapq.heappop(pq)
            if i in alive and j in alive: break
        inter = nodes[i][0] & nodes[j][0]
        k = nxt; nxt += 1
        nodes[k] = (inter, nodes[i][1] + nodes[j][1])
        children[k] = (i, j)
        alive.discard(i); alive.discard(j)
        for o in alive:
            heapq.heappush(pq, (-len(nodes[o][0] & inter), min(o,k), max(o,k)))
        alive.add(k)
    root = next(iter(alive))
    seq = {i: [] for i in range(n)}
    def emit(node, already):
        inter = nodes[node][0]
        new = sorted(inter - already)
        for m in nodes[node][1]: seq[m].extend(new)
        cur = already | inter
        if node in children:
            for ch in children[node]: emit(ch, cur)
        else:
            rest = sorted(reqs[node] - cur)
            seq[node].extend(rest)
    emit(root, set())
    return [tuple(seq[i]) for i in range(n)]

# ---------- online greedy longest-usable-prefix (CacheWeaver-style) ----------
def layout_online_lpm(reqs):
    root = {}
    outs = []
    for R in reqs:
        node = root; rem = set(R); path = []
        while True:
            cands = [c for c in node if c in rem]
            if not cands: break
            c = max(cands, key=lambda c: (node[c][1], -c))
            path.append(c); rem.discard(c); node[c] = (node[c][0], node[c][1]+1); node = node[c][0]
        for c in sorted(rem):
            if c not in node: node[c] = ({}, 0)
            node[c] = (node[c][0], node[c][1]+1)
            path.append(c); node = node[c][0]
        outs.append(tuple(path))
    return outs

# ---------- workloads ----------
def leave_one_out(n): return [frozenset(set(range(n))-{i}) for i in range(n)]

def rag_workload(m=1500, N=1200, k=8, topics=40, zipf=1.1, seed=0):
    rnd = random.Random(seed)
    pool = {}
    for t in range(topics):
        size = max(k+2, int(N/topics*2))
        pool[t] = rnd.sample(range(N), min(size, N))
    wts = {}
    for t in range(topics):
        wts[t] = [1.0/((r+1)**zipf) for r in range(len(pool[t]))]
    reqs = []
    for _ in range(m):
        t = rnd.randrange(topics)
        chosen = set()
        while len(chosen) < k:
            chosen.add(rnd.choices(pool[t], weights=wts[t])[0])
        reqs.append(frozenset(chosen))
    return reqs, N

def report(name, reqs, N=None):
    pop = defaultdict(int)
    for R in reqs:
        for c in R: pop[c]+=1
    base = nosh(reqs)
    res = {
      "no sharing"                    : base,
      "global order (by popularity)"  : trie_cost(layout_global(reqs, lambda c:(-pop[c], c))),
      "global order (by chunk id)"    : trie_cost(layout_global(reqs, lambda c:c)),
      "online greedy LPM"             : trie_cost(layout_online_lpm(reqs)),
      "level-wise greedy hitting set" : trie_cost(layout_greedy_hit(reqs)),
      "agglomerative hierarchy (ours)": trie_cost(layout_agglom(reqs)),
    }
    print("\n=== %s   (m=%d requests, %d distinct chunk-slots) ===" % (name, len(reqs), base))
    best = min(res.values())
    for k_,v in res.items():
        print("  %-32s %8d   (%.3fx of no-sharing, %+.1f%% vs best)" % (k_, v, v/base, 100*(v-best)/best))

for n in [16, 32, 64]:
    report("leave-one-out L_%d (adversarial)" % n, leave_one_out(n))
reqs, N = rag_workload()
report("synthetic RAG: 1500 queries, 8 chunks each, 40 topics, Zipf 1.1", reqs, N)
reqs, N = rag_workload(m=1500, N=1200, k=16, topics=25, zipf=0.8, seed=7)
report("synthetic RAG: 16 chunks/query, 25 topics, Zipf 0.8", reqs, N)
