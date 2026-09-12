"""Quality-preserving variant: pin the top-t most relevant chunks at the END of the prompt
(closest to the question), free-permute only the rest.  Measures what the constraint costs."""
import sys
sys.path.insert(0,'code')
from collections import defaultdict
from beir import retrieve
from phase2 import trie_tokens, wsum, agglom_savings

def layout_free_prefix(free_sets, w):
    """agglomerative on the free part; returns the actual sequences"""
    import heapq
    n=len(free_sets)
    node={i:set(free_sets[i]) for i in range(n)}
    members={i:[i] for i in range(n)}; kids={}
    live=set(range(n)); nxt=n
    inv=defaultdict(set)
    for i in live:
        for x in node[i]: inv[x].add(i)
    pq=[]; seen=set()
    for x,S in inv.items():
        S=sorted(S)
        for a in range(len(S)):
            for b in range(a+1,len(S)):
                p=(S[a],S[b])
                if p in seen: continue
                seen.add(p); v=wsum(node[p[0]]&node[p[1]],w)
                if v>0: heapq.heappush(pq,(-v,p[0],p[1]))
    while pq:
        nv,a,b=heapq.heappop(pq)
        if a not in live or b not in live: continue
        v=wsum(node[a]&node[b],w)
        if v!=-nv:
            if v>0: heapq.heappush(pq,(-v,a,b))
            continue
        if v<=0: break
        c=nxt; nxt+=1
        node[c]=node[a]&node[b]; members[c]=members[a]+members[b]; kids[c]=(a,b)
        for x in node[a]: inv[x].discard(a)
        for x in node[b]: inv[x].discard(b)
        live-={a,b}
        cand=set()
        for x in node[c]: cand|=inv[x]; inv[x].add(c)
        live.add(c)
        for d in cand:
            if d!=c and d in live:
                vv=wsum(node[c]&node[d],w)
                if vv>0: heapq.heappush(pq,(-vv,min(c,d),max(c,d)))
    seq={i:[] for i in range(n)}
    def emit(v, done):
        new=sorted(node[v]-done, key=lambda z:(-w[z],z))
        for mm in members[v]: seq[mm].extend(new)
        cur=done|node[v]
        if v in kids:
            for ch in kids[v]: emit(ch,cur)
        else:
            seq[v].extend(sorted(set(free_sets[v])-cur, key=lambda z:(-w[z],z)))
    for r in sorted(live): emit(r,set())
    return [tuple(seq[i]) for i in range(n)]

def run(ds,k,max_q=None):
    reqs,ranks,w,nd=retrieve(ds,k,max_queries=max_q)
    tot=sum(wsum(R,w) for R in reqs)
    prod=trie_tokens([tuple(r) for r in ranks],w)
    print("\n=== %s k=%d | %d queries | production (relevance order) = %d tok ==="%(ds,k,len(reqs),prod))
    for t in [0,1,2,3]:
        pinned=[tuple(r[:t]) for r in ranks]          # top-t most relevant, kept in order
        free=[frozenset(set(r)-set(r[:t])) for r in ranks]
        nz=[i for i in range(len(free))]
        lay=layout_free_prefix([free[i] for i in nz],w)
        seqs=[lay[j]+tuple(reversed(pinned[nz[j]])) for j in range(len(nz))]
        c=trie_tokens(seqs,w)
        print("   pin top-%d at the end : %9d tok  %+6.1f%% vs production   (%.3fx no-sharing)"
              %(t,c,100*(c-prod)/prod,c/tot))

if __name__=="__main__":
    run("nfcorpus",8); run("scifact",8)
