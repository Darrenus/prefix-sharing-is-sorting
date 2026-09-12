"""Lookahead sweep on REAL retrieval traces: persistent radix trie + batch-level adaptive layout."""
import sys
sys.path.insert(0,'code')
from collections import defaultdict
from beir import retrieve
from phase2 import trie_tokens, wsum
from tiered import layout_free_prefix

class Trie:
    def __init__(self): self.root={}; self.tok=0
    def insert(self,seq,w):
        node=self.root
        for c in seq:
            if c not in node: node[c]=[{},0]; self.tok+=w[c]
            node[c][1]+=1; node=node[c][0]
    def deepest(self,R,beam=16):
        best=(); frontier=[((),frozenset(R),self.root)]
        while frontier:
            nxt=[]
            for path,rem,node in frontier:
                ext=[c for c in node if c in rem]
                if not ext and len(path)>len(best): best=path
                for c in ext:
                    p2=path+(c,)
                    if len(p2)>len(best): best=p2
                    nxt.append((p2,rem-{c},node[c][0]))
            nxt.sort(key=lambda z:-len(z[0])); frontier=nxt[:beam]
        return best

def run_stream(reqs,w,pop,B,tau=8,adaptive=True):
    T=Trie()
    for i in range(0,len(reqs),B):
        batch=reqs[i:i+B]
        placed=[(T.deepest(R),R) for R in batch]
        groups=defaultdict(list)
        for path,R in placed: groups[path].append(frozenset(set(R)-set(path)))
        for path,res in groups.items():
            ne=[r for r in res if r]
            if adaptive and len(ne)>=tau:
                lay=layout_free_prefix(ne,w)
            else:
                lay=[tuple(sorted(r,key=lambda z:(-pop[z],z))) for r in ne]
            for s in lay: T.insert(path+s,w)
            for _ in range(len(res)-len(ne)): T.insert(path,w)
    return T.tok

def main(ds,k,max_q=None):
    reqs,ranks,w,nd=retrieve(ds,k,max_queries=max_q)
    pop=defaultdict(int)
    for R in reqs:
        for x in R: pop[x]+=1
    tot=sum(wsum(R,w) for R in reqs)
    prod=trie_tokens([tuple(r) for r in ranks],w)
    glob=trie_tokens([tuple(sorted(R,key=lambda z:(-pop[z],z))) for R in reqs],w)
    print("\n=== %s k=%d, %d real queries ==="%(ds,k,len(reqs)))
    print("   no sharing %d | production (relevance order) %d | best global order %d"%(tot,prod,glob))
    print("   lookahead W :   savings vs production   /   vs best global order")
    for B in [1,8,32,128,512,len(reqs)]:
        c=run_stream(reqs,w,pop,B)
        print("     W=%-6s %9d tok   %+6.1f%%   %+6.1f%%"%(B if B<len(reqs) else "all",c,
              100*(c-prod)/prod, 100*(c-glob)/glob), flush=True)

if __name__=="__main__":
    main("nfcorpus",8); main("scifact",8)
