# Interactive retrieval demo

A self-contained static site that shows, for a handful of intuitive fitness
queries, what the **off-the-shelf** embedding model vs. our **fine-tuned** model
retrieve at **rank 1** — chosen to make the fine-tuning improvement obvious.

## View it

Just open `index.html` in a browser (double-click works — the data is loaded via
the `data.js` `<script>` tag, so there's no `fetch`/CORS issue and no server is
needed). A retriever selector switches between the three models; each shows its
held-out metrics (baseline vs fine-tuned) and 5 rank-1 examples.

## What's shown

- **Retriever used** — pick one of the three fine-tuned models (BGE-small,
  all-MiniLM-L6-v2, bge-micro-v2) at its best epoch; HF id + param count shown.
- **Metrics** — Recall@{1,5,10} and NDCG@10, fine-tuned vs baseline, with deltas
  (macro-averaged over all 1283 held-out NCSA queries; numbers come straight from
  `experiments/epoch_sweep/results/`).
- **Per-query rank-1 comparison** — for each predefined query: the passage each
  model ranks #1 (✓ correct / ✗ wrong), plus where the *correct* passage landed
  in the baseline ranking (e.g. "#28 → #1"). The picks are all cases where the
  baseline ranked a topically-related but wrong passage first (a hard negative)
  and fine-tuning surfaced the right one.

## Regenerate the data

The numbers are real: queries are the held-out test split, retrieval is cosine
top-1 over the book-scoped NCSA corpus, run with the actual saved checkpoints.

```bash
source ~/.venvs/.rag/bin/activate
python website/build_data.py     # embeds corpus w/ baseline + fine-tuned, mines rank-1 flips -> _all_wins.json
python website/curate.py micro   # (optional) inspect candidate examples per model
python website/make_data_js.py   # writes data.js from the hand-picked queries in PICKS
```

`build_data.py` caches corpus embeddings under `.cache/` (gitignored). Edit the
`PICKS` dict in `make_data_js.py` to change which queries the site shows.
