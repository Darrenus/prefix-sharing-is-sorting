import heapq, sys, time, random
from collections import defaultdict
sys.path.insert(0,'code')
from beir import retrieve

def trie_tokens(seqs, w):
    P=set()
    for s in seqs:
        for j in range(1,len(s)+1): P.add(s[:j])
    return sum(w[p[-1]] for p in P)

def wsum(S,w): return sum(w[x] for x in S)

def agglom_savings(reqs, w, want_edges=False):
    """inverted-index agglomerative; stops when no positive merge remains"""
    live={i:set(R) for i,R in enumerate(reqs)}
    inv=defaultdict(set)
    for i,R in live.items():
        for x in R: inv[x].add(i)
    def val(a,b): return wsum(live[a]&live[b], w)
    pq=[]; seen=set()
    for x,S in inv.items():
        S=sorted(S)
        for ii in range(len(S)):
            for jj in range(ii+1,len(S)):
                a,b=S[ii],S[jj]
                if (a,b) in seen: continue
                seen.add((a,b)); v=val(a,b)
                if v>0: heapq.heappush(pq,(-v,a,b))
    nxt=len(reqs); sav=0
    while pq:
        nv,a,b=heapq.heappop(pq)
        if a not in live or b not in live: continue
        v=val(a,b)
        if v != -nv:
            if v>0: heapq.heappush(pq,(-v,a,b))
            continue
        if v<=0: break
        sav+=v; c=nxt; nxt+=1
        live[c]=live[a]&live[b]
        for x in live[a]: inv[x].discard(a)
        for x in live[b]: inv[x].discard(b)
        del live[a]; del live[b]
        cand=set()
        for x in live[c]:
            cand|=inv[x]; inv[x].add(c)
        for d in cand:
            if d==c or d not in live: continue
            vv=wsum(live[c]&live[d],w)
            if vv>0: heapq.heappush(pq,(-vv,min(c,d),max(c,d)))
    return sav

def mst_bound(reqs, w):
    """max spanning forest on positive w(R_i & R_j) -- Kruskal over candidate edges"""
    inv=defaultdict(list)
    for i,R in enumerate(reqs):
        for x in R: inv[x].append(i)
    E={}
    for x,S in inv.items():
        if len(S)>400: continue                       # cap: hub chunks
        for ii in range(len(S)):
            for jj in range(ii+1,len(S)):
                a,b=S[ii],S[jj]
                if (a,b) not in E: E[(a,b)]=wsum(set(reqs[a])&set(reqs[b]),w)
    par=list(range(len(reqs)))
    def find(u):
        while par[u]!=u: par[u]=par[par[u]]; u=par[u]
        return u
    tot=0
    for (a,b),v in sorted(E.items(), key=lambda kv:-kv[1]):
        ra,rb=find(a),find(b)
        if ra!=rb: par[ra]=rb; tot+=v
    return tot

def greedy_hitting(reqs, w):
    out={}
    def rec(items, pref):
        for i,R in items:
            if not R: out[i]=pref
        items=[(i,R) for i,R in items if R]
        if not items: return
        rem=list(items)
        while rem:
            cnt=defaultdict(int)
            for _,R in rem:
                for c in R: cnt[c]+=w[c]
            v=max(cnt,key=lambda c:(cnt[c],-c))
            grp=[(i,R-{v}) for i,R in rem if v in R]
            rem=[(i,R) for i,R in rem if v not in R]
            rec(grp,pref+(v,))
    rec([(i,set(R)) for i,R in enumerate(reqs)],())
    return [out[i] for i in range(len(reqs))]

def online_lpm(reqs, w, pop):
    root={}; outs=[]
    for R in reqs:
        node=root; rem=set(R); path=[]
        while True:
            c=[x for x in node if x in rem]
            if not c: break
            x=max(c,key=lambda z:(node[z][1],-z))
            path.append(x); rem.discard(x); node[x][1]+=1; node=node[x][0]
        for x in sorted(rem,key=lambda z:(-pop[z],z)):
            if x not in node: node[x]=[{},0]
            node[x][1]+=1; path.append(x); node=node[x][0]
        outs.append(tuple(path))
    return outs

def run(ds, k, max_q=None, max_docs=None):
    t0=time.time()
    reqs, ranks, w, ndocs = retrieve(ds, k, max_docs=max_docs, max_queries=max_q)
    m=len(reqs); U=set().union(*reqs)
    pop=defaultdict(int)
    for R in reqs:
        for x in R: pop[x]+=1
    tot=sum(wsum(R,w) for R in reqs)
    shared=sum(1 for x in U if pop[x]>1)
    print("\n=== %s | corpus %d docs | %d queries | k=%d | retrieval %.0fs ==="%(ds,ndocs,m,k,time.time()-t0))
    print("  overlap: %d distinct chunks for %d chunk-slots (%.2fx dedup);  %.1f%% of chunks wanted by >1 query;  mean |S_x| = %.2f"
          %(len(U), m*k, m*k/max(len(U),1), 100*shared/max(len(U),1), sum(pop.values())/max(len(U),1)))
    print("  no sharing: %d prefill tokens (mean %.0f tok/chunk)"%(tot, tot/(m*k)))
    res={}
    res["relevance order (production)"]=trie_tokens([tuple(r) for r in ranks], w)
    res["global order by popularity"]=trie_tokens([tuple(sorted(R,key=lambda z:(-pop[z],z))) for R in reqs], w)
    res["global order by doc id"]=trie_tokens([tuple(sorted(R)) for R in reqs], w)
    res["online greedy LPM"]=trie_tokens(online_lpm(reqs,w,pop), w)
    res["level-wise greedy hitting set"]=trie_tokens(greedy_hitting(reqs,w), w)
    sav=agglom_savings(reqs,w); res["agglomerative (ours)"]=tot-sav
    lb=tot-mst_bound(reqs,w)
    base=res["relevance order (production)"]
    for nm,v in sorted(res.items(), key=lambda kv:-kv[1]):
        print("   %-32s %9d tok  %.3fx no-sharing   %+6.1f%% vs production"%(nm,v,v/tot,100*(v-base)/base))
    print("   %-32s %9d tok  %.3fx  (spanning-tree lower bound on the optimum)"%("LOWER BOUND",lb,lb/tot))
    return res, tot, lb

if __name__=="__main__":
    run("nfcorpus", 8, max_q=1200)
