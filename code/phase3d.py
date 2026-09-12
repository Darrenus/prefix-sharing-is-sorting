"""Cache capacity and reorder window are SUBSTITUTES.  Map the frontier."""
import sys
sys.path.insert(0,'code')
from phase3 import layouts
from cache import replay

def grid(ds,k,max_q=None):
    reqs,L,w,pop=layouts(ds,k,max_q)
    one=max(sum(w[x] for x in R) for R in reqs)
    seqs=L["agglomerative (ours)"]
    inf=replay(sorted(seqs),w,10**12)
    prod_stream=replay(L["relevance order (production)"],w,one*4)
    Ws=sorted({w for w in [1,128,512,2048,len(seqs)] if w<=len(seqs)})
    Cs=[one*1,one*4,one*16,one*64,one*256,one*1024]
    print("\n=== %s k=%d | %d queries | excess over the unbounded-cache optimum (%d tok) ==="%(ds,k,len(seqs),inf))
    print("   (production, stream order, 4x cache = %+.0f%% over that optimum)"%(100*(prod_stream-inf)/inf))
    hdr="   W \\ cache      "
    for c in Cs: hdr+="%9s"%("%dx"%(c//one))
    print(hdr)
    for W in Ws:
        ss=[]
        for i in range(0,len(seqs),W): ss.extend(sorted(seqs[i:i+W]))
        row="   W=%-12s"%(W if W<len(seqs) else "all")
        for c in Cs:
            v=replay(ss,w,c); row+="%8.0f%%"%(100*(v-inf)/inf)
        print(row, flush=True)

if __name__=="__main__":
    grid("nfcorpus",8); grid("scifact",8)
