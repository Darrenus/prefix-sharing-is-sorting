import itertools, sys
from functools import lru_cache
INF = float('inf')

W = {}          # global weight map, default 1
def w(v): return W.get(v, 1)

_memo_node = {}
def solve(tails):
    """tails: frozenset of frozensets = the remaining chunk-sets of all requests
       sitting at one trie node. Returns min weight of the subtrie below it."""
    tails = frozenset(t for t in tails if t)          # finished requests vanish
    if not tails: return 0
    if tails in _memo_node: return _memo_node[tails]
    order = sorted(tails, key=lambda t: (len(t), sorted(t)))
    res = _groups(tuple(order), frozenset())
    _memo_node[tails] = res
    return res

_memo_grp = {}
def _groups(S, used):
    """S: tuple of remaining request-tails at this node still unassigned to a child.
       used: labels already spent on siblings."""
    if not S: return 0
    key = (S, used)
    if key in _memo_grp: return _memo_grp[key]
    R0 = S[0]
    rest = S[1:]
    best = INF
    for v in sorted(R0 - used):
        cand = [R for R in rest if v in R]
        # every subset of cand joins R0 in the child labelled v
        for r in range(len(cand)+1):
            for T in itertools.combinations(cand, r):
                grp = frozenset([R0 - {v}] + [R - {v} for R in T])
                remaining = tuple(R for R in rest if R not in T)
                c = w(v) + solve(grp) + _groups(remaining, used | {v})
                if c < best: best = c
    _memo_grp[key] = best
    return best

def opt_free(reqs):
    _memo_node.clear(); _memo_grp.clear()
    return solve(frozenset(frozenset(r) for r in reqs))

def opt_global(reqs, universe):
    best = INF; arg = None
    U = sorted(universe)
    for perm in itertools.permutations(U):
        pos = {x:i for i,x in enumerate(perm)}
        P = set()
        for R in reqs:
            s = tuple(sorted(R, key=lambda z: pos[z]))
            for j in range(1,len(s)+1): P.add(s[:j])
        c = sum(w(p[-1]) for p in P)
        if c < best: best = c; arg = perm
    return best, arg
