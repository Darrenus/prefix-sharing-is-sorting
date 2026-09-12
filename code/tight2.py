import random, itertools
from lam import exact_LAM, greedy_agglom_savings
import exact2
from exact2 import solve

def opt_free_w(reqs, w):
    """exact OPT_free with weights: brute force over layouts (small only)"""
    best = None
    pools = [list(itertools.permutations(sorted(R))) for R in reqs]
    for combo in itertools.product(*pools):
        P = set()
        for s in combo:
            for j in range(1,len(s)+1): P.add(s[:j])
        c = sum(w[p[-1]] for p in P)
        if best is None or c < best: best = c
    return best

random.seed(101)
print("=== A. tightness of LAM vs exact OPT_free, WEIGHTED, adversarial shapes ===")
gaps = 0; tot = 0; worst = (1.0, None)
for trial in range(900):
    n = random.randint(4,7); m = random.randint(3,5)
    U = list(range(n))
    w = {x: random.choice([1,1,2,3,5]) for x in U}
    R = set()
    guard = 0
    while len(R) < m and guard < 300:
        guard += 1
        k = random.randint(2, min(4,n))
        R.add(frozenset(random.sample(U,k)))
    R = sorted(R, key=lambda s: sorted(s))
    if len(R) < 3 or len(set().union(*R)) < 3: continue
    if sum(len(x) for x in R) > 18: continue
    tot += 1
    f = opt_free_w(R, w)
    lam,_ = exact_LAM(R, w)
    if lam != f:
        gaps += 1
        if f/lam > worst[0]: worst = (f/lam, [sorted(x) for x in R], dict(w), lam, f)
print("  weighted instances: %d,  LAM != OPT_free in %d,  worst ratio %.4f" % (tot, gaps, worst[0]))
if worst[1]: print("  witness:", worst[1], worst[2], "LAM=%s OPT=%s"%(worst[3],worst[4]))

print()
print("=== B. structured families (designed to break the SDR/Hall step) ===")
fams = {
 "all k-subsets of [n]": [ [frozenset(c) for c in itertools.combinations(range(n),k)]
                            for n,k in [(5,2),(5,3),(6,3),(6,4),(7,5)] ],
 "sunflower + core":     [ [frozenset({0,1}|{2+i,2+i+1}) for i in range(0,6,2)] + [frozenset({0,1,8})] ],
 "nested chains":        [ [frozenset(range(j)) for j in range(2,7)] ],
 "two conflicting blocks":[ [frozenset({0,1,2}),frozenset({0,1,3}),frozenset({2,3,4}),frozenset({2,3,5}),frozenset({0,2,4})] ],
}
for name, insts in fams.items():
    for R in insts:
        R = sorted(set(R), key=lambda s: sorted(s))
        if sum(len(x) for x in R) > 26: 
            print("  %-24s m=%d  (too big for exact OPT, skipped)"%(name,len(R))); continue
        exact2._memo.clear()
        f = solve(frozenset(R), use_canon=False)
        lam,_ = exact_LAM(R)
        print("  %-24s m=%2d  LAM=%3d  OPT_free=%3d  %s" % (name, len(R), lam, f, "TIGHT" if lam==f else "*** GAP ***"))

print()
print("=== C. k=2 sanity: LAM should equal tau(G) + m ===")
def tau(edges, n):
    best = n
    for r in range(n+1):
        for C in itertools.combinations(range(n), r):
            S=set(C)
            if all(S & e for e in edges): return r
    return best
for trial in range(6):
    n = random.randint(4,7); 
    E = set()
    while len(E) < random.randint(4,8):
        a,b = random.sample(range(n),2); E.add(frozenset({a,b}))
    E = sorted(E, key=lambda s: sorted(s))
    lam,_ = exact_LAM(E)
    print("   n=%d m=%d  LAM=%2d   tau+m=%2d  %s" % (n,len(E),lam,tau(E,n)+len(E), "OK" if lam==tau(E,n)+len(E) else "MISMATCH"))
