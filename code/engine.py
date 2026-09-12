"""THE STRUCTURE THEOREM.

    OPT_free(I) = min over binary hierarchies H on the REQUESTS of
                     sum_{x in U} w(x) * t_x(H)
    where S_x = {i : x in R_i}  and  t_x(H) = number of maximal H-clusters inside S_x.

Equivalently (a binary hierarchy on m leaves has exactly m-1 internal nodes):

    OPT_free(I) = sum_i w(R_i)  -  max_H sum_{v internal} w( intersection of v's requests )

The permutations vanish.  Choosing chunk orders IS choosing one hierarchy over requests,
and t_x is exactly the canonical-decomposition size of S_x in that tree -- the segment-tree
quantity.  Consequences: an O(3^m) exact algorithm (vs brute force over prod |R_i|!),
a two-line proof of the leave-one-out theorem, and a bridge to hierarchical clustering.
"""
import itertools

def opt_free(reqs, w=None):
    """exact, via subset DP on the hierarchy.  Returns (cost, savings, split table)."""
    m = len(reqs)
    U = sorted(set().union(*reqs))
    w = w or {x: 1 for x in U}
    inter = [0]*(1 << m); fv = [0]*(1 << m)
    sets  = [None]*(1 << m)
    for B in range(1, 1 << m):
        low = B & -B; i = low.bit_length()-1; rest = B ^ low
        sets[B] = set(reqs[i]) if rest == 0 else (sets[rest] & reqs[i])
        fv[B] = sum(w[x] for x in sets[B])
    M = [0]*(1 << m); sp = [0]*(1 << m)
    for B in range(1, 1 << m):
        if B & (B-1) == 0: continue                      # singleton
        low = B & -B; rest = B ^ low; sub = rest; best = -1; ba = 0
        while True:
            B1 = sub | low
            if B1 != B:                       # both sides must be non-empty
                v = M[B1] + M[B ^ B1]
                if v > best: best = v; ba = B1
            if sub == 0: break
            sub = (sub-1) & rest
        M[B] = fv[B] + best; sp[B] = ba
    tot = sum(sum(w[x] for x in R) for R in reqs)
    return tot - M[(1<<m)-1], M[(1<<m)-1], sp

def opt_global(reqs, w=None):
    U = sorted(set().union(*reqs)); w = w or {x:1 for x in U}
    best = None
    for perm in itertools.permutations(U):
        pos = {x:i for i,x in enumerate(perm)}
        P = set()
        for R in reqs:
            s = tuple(sorted(R, key=lambda z: pos[z]))
            for j in range(1,len(s)+1): P.add(s[:j])
        c = sum(w[p[-1]] for p in P)
        if best is None or c < best: best = c
    return best

def greedy(reqs, w=None):
    U = sorted(set().union(*reqs)); w = w or {x:1 for x in U}
    nodes = {i: set(reqs[i]) for i in range(len(reqs))}
    alive = set(nodes); nxt = len(reqs); sav = 0
    while len(alive) > 1:
        best = None
        for a,b in itertools.combinations(sorted(alive),2):
            v = sum(w[x] for x in nodes[a] & nodes[b])
            if best is None or v > best[0]: best = (v,a,b)
        v,a,b = best; sav += v
        nodes[nxt] = nodes[a] & nodes[b]; alive -= {a,b}; alive.add(nxt); nxt += 1
    tot = sum(sum(w[x] for x in R) for R in reqs)
    return tot - sav

if __name__ == "__main__":
    import time
    def closed(n):
        l = (n-1).bit_length() if n>1 else 0; return n*l - 2**l + n
    print("A. leave-one-out L_n via the structure theorem (S_x = complement of leaf x,")
    print("   so t_x = depth(x) and LAM = minimum EXTERNAL PATH LENGTH of a binary tree):")
    for n in range(2,17):
        R = [frozenset(set(range(n))-{i}) for i in range(n)]
        t0=time.time(); c,_,_ = opt_free(R); dt=time.time()-t0
        print("   n=%2d  OPT_free=%4d  min-external-path-length=%4d  %s  (%.2fs)"
              % (n,c,closed(n),"OK" if c==closed(n) else "MISMATCH",dt))
