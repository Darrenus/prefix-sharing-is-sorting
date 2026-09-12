import random, itertools, sys
from lam import exact_LAM, greedy_agglom_savings
import exact2
from exact2 import solve

def opt_free(reqs):
    exact2._memo.clear(); exact2._memo_grp = {} if hasattr(exact2,'_memo_grp') else None
    return solve(frozenset(reqs), use_canon=False)

random.seed(11)
worst = (1.0, None); ngap = 0; ntot = 0
worst_g = (1.0, None)
for trial in range(1500):
    n = random.randint(4,6); m = random.randint(3,6)
    U = list(range(n))
    R = set()
    guard = 0
    while len(R) < m and guard < 200:
        guard += 1
        k = random.randint(2, min(4, n))
        R.add(frozenset(random.sample(U, k)))
    R = sorted(R, key=lambda s: sorted(s))
    if len(R) < 3: continue
    if len(set().union(*R)) < 3: continue
    ntot += 1
    f = opt_free(R)
    lam, sav = exact_LAM(R)
    if lam < f:
        ngap += 1
        if f/lam > worst[0]: worst = (f/lam, [sorted(x) for x in R], lam, f)
    g = greedy_agglom_savings(R)
    if sav > 0 and g < sav:
        if sav/max(g,1e-9) > worst_g[0]: worst_g = (sav/g, [sorted(x) for x in R], g, sav)

print("instances tested: %d" % ntot)
print("LAM < OPT_free in %d of them (%.1f%%)" % (ngap, 100*ngap/ntot))
print("worst OPT_free/LAM ratio: %.4f" % worst[0])
if worst[1]: print("   witness:", worst[1], " LAM=%d OPT=%d" % (worst[2], worst[3]))
print("worst greedy-savings shortfall vs exact LAM savings: %.4f" % worst_g[0])
if worst_g[1]: print("   witness:", worst_g[1], " greedy=%s exact=%s" % (worst_g[2], worst_g[3]))
