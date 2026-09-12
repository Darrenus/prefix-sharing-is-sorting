"""Pure-python BM25 over a BEIR dataset -> real retrieval sets, real token counts."""
import json, math, os, re, sys
from collections import defaultdict
import tiktoken

TOK = re.compile(r"[a-z0-9]+")
ENC = tiktoken.get_encoding("cl100k_base")

def load(ds, root="data", max_docs=None, max_queries=None):
    cdir = os.path.join(root, ds)
    docs, ids = [], []
    with open(os.path.join(cdir, "corpus.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            ids.append(o["_id"]); docs.append(((o.get("title") or "") + " " + (o.get("text") or "")).strip())
            if max_docs and len(ids) >= max_docs: break
    qs, qids = [], []
    with open(os.path.join(cdir, "queries.jsonl")) as f:
        for line in f:
            o = json.loads(line); qids.append(o["_id"]); qs.append(o["text"])
            if max_queries and len(qs) >= max_queries: break
    return ids, docs, qids, qs

class BM25:
    def __init__(self, docs, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.inv = defaultdict(list); self.dl = []
        for i, d in enumerate(docs):
            tf = defaultdict(int)
            for t in TOK.findall(d.lower()): tf[t] += 1
            self.dl.append(sum(tf.values()) or 1)
            for t, c in tf.items(): self.inv[t].append((i, c))
        self.N = len(docs); self.avgdl = sum(self.dl)/max(self.N,1)
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5)/(len(p) + 0.5)) for t, p in self.inv.items()}
    def top(self, q, k):
        sc = defaultdict(float)
        for t in set(TOK.findall(q.lower())):
            if t not in self.inv: continue
            idf = self.idf[t]
            for i, c in self.inv[t]:
                sc[i] += idf * c*(self.k1+1)/(c + self.k1*(1 - self.b + self.b*self.dl[i]/self.avgdl))
        return [i for i,_ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]

def retrieve(ds, k, max_docs=None, max_queries=None, root="data"):
    ids, docs, qids, qs = load(ds, root, max_docs, max_queries)
    bm = BM25(docs)
    reqs, ranks = [], []
    for q in qs:
        t = bm.top(q, k)
        if len(t) >= 2: reqs.append(frozenset(t)); ranks.append(t)
    ntok = {}
    used = set().union(*reqs) if reqs else set()
    for i in sorted(used): ntok[i] = max(1, len(ENC.encode(docs[i])))
    return reqs, ranks, ntok, len(docs)
