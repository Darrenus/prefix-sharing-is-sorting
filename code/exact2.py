import itertools, sys
from functools import lru_cache
sys.setrecursionlimit(100000)

_memo = {}
_canon_cache = {}

def canon(tails):
    """canonical form of a set of tails under relabelling of the universe"""
    key = tails
    if key in _canon_cache: return _canon_cache[key]
    U = sorted(set().union(*tails)) if tails else []
    best = None
    for perm in itertools.permutations(range(len(U))):
        mp = {U[i]: perm[i] for i in range(len(U))}
        f = frozenset(frozenset(mp[x] for x in t) for t in tails)
        rep = tuple(sorted(tuple(sorted(t)) for t in f))
        if best is None or rep < best: best = rep
    res = frozenset(frozenset(t) for t in best)
    _canon_cache[key] = res
    return res

def solve(tails, use_canon=True):
    tails = frozenset(t for t in tails if t)
    if not tails: return 0
    key = canon(tails) if use_canon and len(set().union(*tails)) <= 7 else tails
    if key in _memo: return _memo[key]
    lst = sorted(tails, key=lambda t: (len(t), sorted(t)))
    best = None
    for assign in itertools.product(*[sorted(t) for t in lst]):
        groups = {}
        for t, v in zip(lst, assign):
            groups.setdefault(v, []).append(t - {v})
        c = len(groups) + sum(solve(frozenset(g), use_canon) for g in groups.values())
        if best is None or c < best: best = c
    _memo[key] = best
    return best

def leave_one_out(n):
    return frozenset(frozenset(set(range(n)) - {i}) for i in range(n))

def glob_opt(reqs, n):
    best = None
    for perm in itertools.permutations(range(n)):
        pos = {x:i for i,x in enumerate(perm)}
        P = set()
        for R in reqs:
            s = tuple(sorted(R, key=lambda z: pos[z]))
            for j in range(1,len(s)+1): P.add(s[:j])
        if best is None or len(P) < best: best = len(P)
    return best

def S(t, memo={1:0}):
    if t in memo: return memo[t]
    best = None
    for parts in partitions(t):
        if len(parts) < 2: continue
        c = t*(len(parts)-1) + sum(S(p) for p in parts)
        if best is None or c < best: best = c
    memo[t] = best; return best

def partitions(n, maxp=None):
    if maxp is None: maxp = n
    if n == 0: yield []; return
    for p in range(min(n,maxp), 0, -1):
        for rest in partitions(n-p, p):
            yield [p]+rest

def mergesort_formula(n):
    import math
    L = (n-1).bit_length() if n>1 else 0   # ceil(log2 n)
    return n*L - 2**L + n

if __name__ == "__main__":
    print(" n | OPT_free(exact) | S(n) recursion | mergesort formula | OPT_global | ratio")
    for n in range(2,7):
        reqs = leave_one_out(n)
        _memo.clear()
        f = solve(reqs)
        g = glob_opt(reqs, n)
        print("%2d | %15d | %14d | %17d | %10d | %.3f" % (n, f, S(n), mergesort_formula(n), g, g/f))
    print()
    print("closed-form check, S(n) vs n*ceil(lg n)-2^ceil(lg n)+n, n=1..24:")
    print([S(n) for n in range(1,25)])
    print([mergesort_formula(n) for n in range(1,25)])
    