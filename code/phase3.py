import sys
sys.path.insert(0,'code')
from collections import defaultdict
from beir import retrieve
from phase2 import trie_tokens, wsum, greedy_hitting, online_lpm, agglom_savings
from tiered import layout_free_prefix
from cache import replay

def layouts(ds,k,max_q=None):
    reqs,ranks,w,nd=retrieve(ds,k,max_queries=max_q)
    pop=defaultdict(int)
    for R in reqs:
        for x in R: pop[x]+=1
    L={}
    L["relevance order (production)"]=[tuple(r) for r in ranks]
    L["global order by popularity"]=[tuple(sorted(R,key=lambda z:(-pop[z],z))) for R in reqs]
    L["online greedy LPM"]=list(online_lpm(reqs,w,pop))
    L["level-wise greedy hitting set"]=list(greedy_hitting(reqs,w))
    L["agglomerative (ours)"]=list(layout_free_prefix(list(reqs),w))
    return reqs,L,w,pop

def main(ds,k,max_q=None):
    reqs,L,w,pop=layouts(ds,k,max_q)
    one=max(sum(w[x] for x in R) for R in reqs)      # one request's worth of context
    tot=sum(wsum(R,w) for R in reqs)
    print("\n=== %s k=%d | %d queries | one request = %d tok | unbounded-cache trie sizes ==="%(ds,k,len(reqs),one))
    caps=[one*1, one*4, one*16, one*64, one*256, one*1024, 10**12]
    hdr="  %-32s"%"cache capacity ->"
    for c in caps: hdr+= "%10s"%(("%dx"%(c//one)) if c<10**12 else "inf")
    print(hdr)
    base={}
    for nm,seqs in L.items():
        row="  %-32s"%nm
        for c in caps:
            v=replay(seqs,w,c); base.setdefault(nm,{})[c]=v
            row+="%10d"%v
        print(row, flush=True)
    print("  %-32s"%"(ratio to production, per capacity)")
    prod=base["relevance order (production)"]
    for nm in L:
        row="  %-32s"%nm
        for c in caps: row+="%9.1f%%"%(100*(base[nm][c]-prod[c])/prod[c])
        print(row)
    return base,caps,one,reqs,L,w

if __name__=="__main__":
    main("nfcorpus",8)
