from core import *
import itertools, random

# ---- claim A: for k<=2 there is never a gap (global order is WLOG) ----
bad = 0
random.seed(0)
for trial in range(4000):
    n = random.randint(3,6); m = random.randint(2,6)
    U = list(range(n))
    reqs = set()
    while len(reqs) < m:
        sz = random.choice([1,2])
        reqs.add(frozenset(random.sample(U, sz)))
    reqs = list(reqs)
    if opt_free(reqs)[0] != opt_global(reqs,U)[0]:
        bad += 1; print("k<=2 GAP!", reqs); break
print("claim A (k<=2, %d random instances): counterexamples = %d" % (4000, bad))

# ---- minimal counterexample search for uniform k=3 ----
found = []
for n in range(4,7):
    U = list(range(n))
    triples = [frozenset(t) for t in itertools.combinations(U,3)]
    for m in range(2,6):
        if len(triples) < m: continue
        tot = 0
        hit = None
        for reqs in itertools.combinations(triples, m):
            # require the universe actually be used, else it's an (n-1) instance
            if len(set().union(*reqs)) != n: continue
            tot += 1
            f = opt_free(list(reqs))[0]
            g = opt_global(list(reqs),U)[0]
            if g > f:
                hit = (reqs,f,g); break
        print("n=%d m=%d : scanned %d, gap found: %s" % (n,m,tot, "YES" if hit else "no"))
        if hit:
            found.append((n,m,hit)); break
    if found: break

if found:
    n,m,(reqs,f,g) = found[0]
    nm = 'abcdefg'
    sh = lambda s: ''.join(nm[z] for z in sorted(s))
    print("\n=== MINIMAL COUNTEREXAMPLE (k=3) ===")
    print("universe size n=%d, requests m=%d" % (n,m))
    print("requests:", [sh(r) for r in reqs])
    print("free opt =", f, "  global opt =", g)
    print("free  witness:", [''.join(nm[z] for z in s) for s in opt_free(list(reqs))[1]])
    go = opt_global(list(reqs),list(range(n)))[1]
    print("global witness order:", ''.join(nm[z] for z in go[0]), [''.join(nm[z] for z in s) for s in go[1]])
