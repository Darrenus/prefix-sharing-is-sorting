"""The hierarchy gives a SCHEDULE as well as a layout.

Serving requests in the trie's DFS order (= sorting the stream by the laid-out sequence) makes
every trie node's uses contiguous in time, so a cache holding one root-to-leaf path suffices to
reach the unbounded-cache optimum.  This applies to ANY layout, so it is a fair, method-agnostic
comparison -- and it turns cache capacity from a hard constraint into a non-issue."""
import sys
sys.path.insert(0,'code')
from phase3 import layouts
from phase2 import wsum
from cache import replay

def main(ds,k,max_q=None):
    reqs,L,w,pop=layouts(ds,k,max_q)
    one=max(sum(w[x] for x in R) for R in reqs)
    caps=[one*1, one*2, one*4, one*16, one*64, 10**12]
    print("\n=== %s k=%d | %d queries | one request = %d tok ==="%(ds,k,len(reqs),one))
    print("   arrival order = the real query stream   vs   DFS order (sorted by laid-out sequence)")
    hdr="  %-30s %-8s"%("","order")
    for c in caps: hdr+="%10s"%(("%dx"%(c//one)) if c<10**12 else "inf")
    print(hdr)
    for nm,seqs in L.items():
        for tag,ss in [("stream",seqs),("DFS",sorted(seqs))]:
            row="  %-30s %-8s"%(nm if tag=="stream" else "",tag)
            for c in caps: row+="%10d"%replay(ss,w,c)
            print(row, flush=True)
    print()
    print("   -> under DFS order, a cache of ONE request's context already reaches the")
    print("      unbounded-cache cost; the finite-cache penalty disappears entirely.")

if __name__=="__main__":
    main("nfcorpus",8)
