from exact import *
import itertools
print("=== complete k-uniform hypergraph K_n^(k): all k-subsets of [n] ===")
print(" n  k    m   free  global  gap  ratio")
for n in range(3,8):
  for k in range(2, n):
    reqs = [frozenset(c) for c in itertools.combinations(range(n), k)]
    if len(reqs) > 20: continue
    f = opt_free(reqs)
    g,_ = opt_global(reqs, range(n))
    print("%2d %2d %4d %6d %7d %4d  %.4f" % (n,k,len(reqs),f,g,g-f,g/f))
