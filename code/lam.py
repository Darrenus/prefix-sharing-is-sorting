"""Laminar relaxation of MPT.

Derivation:  cost = sum_d |P_d| over a refining chain of partitions of the requests,
where a block B may survive only to depth f(B) = |intersection of B's requests|.
Splitting as late as possible is optimal, so

    cost(H) = sum_{B in H} [ f(B) - f(parent(B)) ]
            = sum_i |R_i|  -  sum_{B internal} f(B)        (binary H, which is WLOG)

and, counting per chunk instead of per node,

    LAM(H) = sum_{x in U} w(x) * t_x(H),
    t_x(H) = number of MAXIMAL clusters of H that partition S_x = {i : x in R_i}.

So: build ONE binary hierarchy over the requests; every chunk is paid for once per
maximal cluster it gets fragmented across.  Minimise total fragmentation.
"""
import itertools
from functools import lru_cache

def build(reqs):
    m = len(reqs)
    U = sorted(set().union(*reqs))
    S = {x: frozenset(i for i in range(m) if x in reqs[i]) for x in U}
    return m, U, S

def exact_LAM(reqs, w=None):
    """DP over subsets: M(B) = f(B) + max split; LAM = sum k_i - M(full)."""
    m, U, S = build(reqs)
    w = w or {x: 1 for x in U}
    full = (1 << m) - 1
    inter = [None]*(1 << m)
    for B in range(1, 1 << m):
        low = B & -B; i = low.bit_length()-1
        rest = B ^ low
        inter[B] = set(reqs[i]) if rest == 0 else (inter[rest] & reqs[i])
    fv = [0]*(1 << m)
    for B in range(1, 1 << m):
        fv[B] = sum(w[x] for x in inter[B])
    M = [0]*(1 << m); best_split = [None]*(1 << m)
    for B in range(1, 1 << m):
        if bin(B).count('1') < 2: continue
        low = B & -B
        b = 0; rest = B ^ low; sub = rest
        bestv = -1
        while True:                      # iterate subsets of rest; pair with low
            B1 = sub | low; B2 = B ^ B1
            v = M[B1] + M[B2]
            if v > bestv: bestv = v; b = B1
            if sub == 0: break
            sub = (sub-1) & rest
        M[B] = fv[B] + bestv; best_split[B] = b
    tot = sum(sum(w[x] for x in R) for R in reqs)
    return tot - M[full], M[full]

def LAM_of_hierarchy(reqs, splits, w=None):
    """cross-check: count per-chunk fragmentation of a given binary hierarchy"""
    m, U, S = build(reqs); w = w or {x:1 for x in U}
    full = (1<<m)-1
    def maximal_count(B, Sx):
        if (B & ~Sx) == 0: return 1        # entirely inside S_x -> one maximal cluster
        if (B & Sx) == 0: return 0
        B1 = splits[B]; return maximal_count(B1, Sx) + maximal_count(B ^ B1, Sx)
    return sum(w[x]*maximal_count(full, sum(1<<i for i in S[x])) for x in U)

def greedy_agglom_savings(reqs, w=None):
    """the algorithm from the paper, scored on the LAM objective"""
    m, U, S = build(reqs); w = w or {x:1 for x in U}
    nodes = {i: (set(reqs[i]), 1<<i) for i in range(m)}
    alive = set(range(m)); nxt = m; sav = 0
    while len(alive) > 1:
        best = None
        for a, b in itertools.combinations(sorted(alive), 2):
            v = sum(w[x] for x in nodes[a][0] & nodes[b][0])
            if best is None or v > best[0]: best = (v, a, b)
        v, a, b = best; sav += v
        nodes[nxt] = (nodes[a][0] & nodes[b][0], nodes[a][1] | nodes[b][1])
        alive.discard(a); alive.discard(b); alive.add(nxt); nxt += 1
    return sav

if __name__ == "__main__":
    print("sanity: leave-one-out L_n  (LAM should be min external path length = merge-sort number)")
    def closed(n):
        l = (n-1).bit_length() if n>1 else 0; return n*l - 2**l + n
    for n in range(2, 13):
        reqs = [frozenset(set(range(n))-{i}) for i in range(n)]
        lam,_ = exact_LAM(reqs)
        print("   n=%2d  LAM=%4d  merge-sort formula=%4d  %s" % (n, lam, closed(n), "OK" if lam==closed(n) else "MISMATCH"))
