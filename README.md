# Prefix Sharing Is a Sorting Problem

Code and data pipeline for the paper *Prefix Sharing Is a Sorting Problem* (Rong He).

LLM serving reuses KV cache by exact prefix match. When a prompt is assembled from a *set* of
reusable pieces — retrieved passages, tool definitions, few-shot exemplars — their order is a free
design variable, and every deployed system fixes it with a single global convention. This
repository contains the exact solvers, the verification scripts for every theorem, and the
retrieval-trace evaluation.

## Layout

- `paper.md` / `paper.tex` / `paper.pdf` — the paper
- `code/` — solvers, verification, and evaluation scripts
- `get_data.sh` — downloads the BEIR corpora used in Sections 8–9 (not committed)

## Reproducing

```bash
./get_data.sh                  # BEIR corpora (nfcorpus, scifact, fiqa)
pip install tiktoken           # real token counts

python3 code/engine.py         # the structure theorem: O(3^m) exact solver
python3 code/halfproof.py      # Theorem 6 and its per-threshold engine A(t) <= 2G(t)
python3 code/t2.py             # the minimal counterexample (Theorem 4)
python3 code/tight.py          # LAM == OPT_free on 1500 random instances
python3 code/phase2.py         # real BM25 retrieval traces, real token counts
python3 code/sweep.py          # 3 corpora x 3 depths
python3 code/phase3b.py        # Theorem 7: DFS order + one-request cache
python3 code/phase3d.py        # the capacity / reorder-window frontier
```

Every number in the paper comes from these scripts. `code/md2tex.py` and `code/assemble.py`
build `paper.tex` from `paper.md`.

## License

Paper: CC BY 4.0. Code: MIT.
