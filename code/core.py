import itertools

def trie_nodes(seqs):
    P = set()
    for s in seqs:
        for j in range(1, len(s)+1):
            P.add(s[:j])
    return P

def cost(seqs, w=None):
    P = trie_nodes(seqs)
    if w is None:
        return len(P)
    return sum(w[p[-1]] for p in P)

def opt_free(reqs, w=None):
    """min trie cost when every request may permute its own chunks independently"""
    best = None; arg = None
    pools = [list(itertools.permutations(sorted(R))) for R in reqs]
    for combo in itertools.product(*pools):
        c = cost(combo, w)
        if best is None or c < best:
            best = c; arg = combo
    return best, arg

def opt_global(reqs, universe, w=None):
    """min trie cost when ONE global order of the universe must sort every request"""
    best = None; arg = None
    U = sorted(universe)
    for perm in itertools.permutations(U):
        pos = {x: i for i, x in enumerate(perm)}
        seqs = tuple(tuple(sorted(R, key=lambda z: pos[z])) for R in reqs)
        c = cost(seqs, w)
        if best is None or c < best:
            best = c; arg = (perm, seqs)
    return best, arg
