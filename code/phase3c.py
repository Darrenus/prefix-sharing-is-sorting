import sys
sys.path.insert(0,'code')
from phase3 import layouts
from phase2 import wsum
from cache import replay

def tightness(ds,k,max_q=None):
    reqs,L,w,pop=layouts(ds,k,max_q)
    one=max(sum(w[x] for x in R) for R in reqs)
    seqs=sorted(L["agglomerative (ours)"])
    inf=replay(seqs,w,10**12)
    print("\n=== %s k=%d : is one request's context exactly enough? (DFS order, ours) ==="%(ds,k))
    print("   unbounded-cache cost = %d"%inf)
    for f in [0.25,0.5,0.75,0.9,1.0,1.25]:
        v=replay(seqs,w,int(one*f))
        print("     cap=%.2fx one-request (%8d tok): %9d   %+.2f%% vs unbounded"%(f,int(one*f),v,100*(v-inf)/inf))

def windowed(ds,k,max_q=None):
    """deployable version: sort only inside a reorder window of W arrivals"""
    reqs,L,w,pop=layouts(ds,k,max_q)
    one=max(sum(w[x] for x in R) for R in reqs)
    seqs=L["agglomerative (ours)"]
    prod=L["relevance order (production)"]
    inf=replay(sorted(seqs),w,10**12)
    print("\n=== %s k=%d : reorder window W, cache = 4x one request ==="%(ds,k))
    print("   production in stream order, cache 4x : %d"%replay(prod,w,one*4))
    for W in [1,8,32,128,512,2048,len(seqs)]:
        ss=[]
        for i in range(0,len(seqs),W): ss.extend(sorted(seqs[i:i+W]))
        v=replay(ss,w,one*4)
        print("     W=%-6s %9d tok   %+6.1f%% vs unbounded-cache optimum"%(
              W if W<len(seqs) else "all", v, 100*(v-inf)/inf))

if __name__=="__main__":
    tightness("nfcorpus",8); tightness("scifact",8)
    windowed("nfcorpus",8)
