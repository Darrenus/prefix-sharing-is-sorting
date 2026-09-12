import itertools, random, sys
from exact2 import solve, glob_opt, _memo
print("search: max OPT_global/OPT_free over small hypergraphs (exact)")
print(" n  k | worst ratio found | witness")
random.seed(1)
for n in range(4,7):
    for k in range(3,n):
        allk = [frozenset(c) for c in itertools.combinations(range(n),k)]
        best = (1.0, None)
        # exhaustive over all sub-families when small, else sample
        fams = []
        for m in range(2, min(len(allk), 8)+1):
            combos = list(itertools.combinations(allk, m))
            if len(combos) > 400: combos = random.sample(combos, 400)
            fams.extend(combos)
        for F in fams:
            if len(set().union(*F)) != n: continue
            _memo.clear()
            f = solve(frozenset(F)); g = glob_opt(F, n)
            if g/f > best[0]: best = (g/f, F, f, g)
        if best[1]:
            print("%2d %2d | %17.4f | m=%d free=%d glob=%d  %s" %
                  (n,k,best[0],len(best[1]),best[2],best[3],
                   sorted(tuple(sorted(x)) for x in best[1])))
