import sys
sys.setrecursionlimit(100000)
from functools import lru_cache

def partitions(n, maxp=None):
    if maxp is None: maxp = n
    if n == 0: yield []; return
    for p in range(min(n,maxp),0,-1):
        for rest in partitions(n-p,p):
            yield [p]+rest

@lru_cache(None)
def S_full(t):           # min over ALL partitions into >=2 parts
    if t == 1: return 0
    best = None
    for parts in partitions(t):
        if len(parts) < 2: continue
        c = t*(len(parts)-1) + sum(S_full(p) for p in parts)
        if best is None or c < best: best = c
    return best

@lru_cache(None)
def S_bin(t):            # balanced binary split only
    if t == 1: return 0
    return t + S_bin((t+1)//2) + S_bin(t//2)

def closed(n):
    L = (n-1).bit_length() if n > 1 else 0
    return n*L - 2**L + n

bad = [n for n in range(1,41) if not (S_full(n)==S_bin(n)==closed(n))]
print("n=1..40 : S_full == S_bin == closed-form ?  mismatches:", bad)
bad2 = [n for n in range(1,4001) if S_bin(n)!=closed(n)]
print("n=1..4000: balanced-binary recursion == closed form ? mismatches:", bad2)

print()
print("=== the separation on the leave-one-out family L_n ===")
print("  n |  OPT_free = n*ceil(lg n)-2^ceil(lg n)+n | OPT_global=(n-1)(n+2)/2 |   ratio")
for n in [4,6,8,16,32,64,128,256,1024,4096]:
    f = closed(n); g = (n-1)*(n+2)//2
    print("%5d | %38d | %23d | %7.2fx" % (n,f,g,g/f))
