import random, itertools
from engine import opt_free, opt_global
def ratio(R):
    o,_,_ = opt_free(R)
    return opt_global(R)/o, o
best_overall = {}
rnd = random.Random(3)
for n,k,m,iters in [(6,3,7,4000),(6,4,7,4000),(7,3,8,2500),(7,4,8,2500),(7,5,8,2500),(7,6,8,1500)]:
    allk = [frozenset(c) for c in itertools.combinations(range(n),k)]
    if len(allk) < m: continue
    cur = sorted(rnd.sample(allk, m), key=lambda s: sorted(s))
    cr,_ = ratio(cur); best = (cr, list(cur))
    T0 = 0.05
    for it in range(iters):
        T = T0*(1-it/iters)+1e-4
        cand = list(cur)
        cand[rnd.randrange(m)] = rnd.choice(allk)
        cand = sorted(set(cand), key=lambda s: sorted(s))
        if len(cand) != m or len(set().union(*cand)) != n: continue
        nr,_ = ratio(cand)
        if nr > cr or rnd.random() < pow(2.718, (nr-cr)/T):
            cur, cr = cand, nr
            if nr > best[0]: best = (nr, list(cand))
    o,_,_ = opt_free(best[1]); g = opt_global(best[1])
    best_overall[(n,k)] = best[0]
    print("n=%d k=%d m=%d : best ratio %.4f  (OPT_free=%d, OPT_global=%d)" % (n,k,m,best[0],o,g))
    print("    witness:", [''.join(str(z) for z in sorted(x)) for x in best[1]], flush=True)
print()
print("best ratio by request size k:")
for (n,k),v in sorted(best_overall.items(), key=lambda t:(t[0][1],t[0][0])):
    print("   k=%d (n=%d): %.4f" % (k,n,v))
