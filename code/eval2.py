import random, itertools
from engine import opt_free, greedy, opt_global

def rag_instance(m, N, k, poolsize, zipf, rnd):
    pool = rnd.sample(range(N), poolsize)
    wts = [1.0/((r+1)**zipf) for r in range(poolsize)]
    R = set()
    guard = 0
    while len(R) < m and guard < 500:
        guard += 1
        c = set()
        while len(c) < k: c.add(rnd.choices(pool, weights=wts)[0])
        R.add(frozenset(c))
    return sorted(R, key=lambda s: sorted(s))

rnd = random.Random(7)
print("=== how far is the agglomerative heuristic from the TRUE optimum? ===")
print("   (RAG-shaped instances, exact optimum via the structure theorem)")
print("  m   k  pool | OPT_free  greedy  gap    | global   global/OPT")
rows = []
for (m,k,pool,z) in [(10,5,14,1.0),(12,6,18,1.0),(12,8,16,0.7),(14,6,20,1.2),(14,8,22,0.9),(15,7,24,1.0)]:
    accs=[]; gg=[]
    for t in range(12):
        R = rag_instance(m, 60, k, pool, z, rnd)
        if len(R) < m: continue
        o,_,_ = opt_free(R); g = greedy(R)
        accs.append(g/o)
        if len(set().union(*R)) <= 8:
            gg.append(opt_global(R)/o)
    if accs:
        print("  %2d  %2d  %4d | exact       %5.3fx avg (max %5.3fx) | %s"
              % (m,k,pool, sum(accs)/len(accs), max(accs),
                 ("%.3fx"%(sum(gg)/len(gg))) if gg else "n/a (universe>8)"))
        rows.append((m,k,sum(accs)/len(accs),max(accs)))
print()
print("summary: greedy is on average %.2f%% above optimum, worst seen %.2f%%"
      % (100*(sum(r[2] for r in rows)/len(rows)-1), 100*(max(r[3] for r in rows)-1)))
