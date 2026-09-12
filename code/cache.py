"""Finite-capacity radix KV cache, the way SGLang/vLLM actually behave:
a cached node implies its ancestors are cached, and eviction removes LRU *leaves*."""
import heapq, sys
sys.path.insert(0,'code')

class RadixCache:
    def __init__(self, cap):
        self.cap=cap; self.size=0; self.clock=0
        # node 0 = root
        self.par={0:-1}; self.w={0:0}; self.kid={0:{}}; self.nk={0:0}; self.lu={0:0}
        self.nxt=1; self.heap=[]
    def _touch(self,v):
        self.clock+=1
        while v!=0:
            self.lu[v]=self.clock
            if self.nk[v]==0: heapq.heappush(self.heap,(self.lu[v],v))
            v=self.par[v]
    def _evict(self):
        while self.size>self.cap and self.heap:
            lu,v=heapq.heappop(self.heap)
            if v not in self.w or self.nk[v]!=0 or self.lu[v]!=lu: continue
            p=self.par[v]
            self.size-=self.w[v]
            c=next(k for k,val in self.kid[p].items() if val==v)
            del self.kid[p][c]; self.nk[p]-=1
            del self.par[v]; del self.w[v]; del self.kid[v]; del self.nk[v]; del self.lu[v]
            if self.nk[p]==0 and p!=0: heapq.heappush(self.heap,(self.lu[p],p))
    def serve(self, seq, w):
        """returns tokens that had to be RECOMPUTED"""
        v=0; i=0
        while i<len(seq) and seq[i] in self.kid[v]:
            v=self.kid[v][seq[i]]; i+=1
        self._touch(v if v else 0)
        cost=0
        for c in seq[i:]:
            u=self.nxt; self.nxt+=1
            self.kid[v][c]=u; self.par[u]=v; self.w[u]=w[c]; self.kid[u]={}
            self.nk[u]=0; self.lu[u]=self.clock; self.nk[v]+=1
            self.size+=w[c]; cost+=w[c]
            v=u
        self._touch(v)
        self._evict()
        return cost

def replay(seqs, w, cap):
    C=RadixCache(cap); tot=0
    for s in seqs: tot+=C.serve(s,w)
    return tot
