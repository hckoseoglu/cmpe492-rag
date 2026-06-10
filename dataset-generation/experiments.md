# Experiments & Poster Log

Running record of the retriever fine-tuning experiments and the CMPE 492 poster.
Read this first when resuming in a fresh session. Pipeline/architecture details
live in `CLAUDE.md`; this file is the *results + reproduction* layer.

Environment: `source ~/.venvs/.rag/bin/activate` (venv is **not** in-repo).

---

## 1. Epoch sweep — fine-tuned retriever vs off-the-shelf baseline

**Question.** Does fine-tuning the embedder on our synthetic `(query, positive,
hard-negative)` triplets beat the off-the-shelf model, and how many epochs before
it overfits?

**Models (descending size — this is the order both poster figures use):**

| tag        | HF id                                   | params | role            |
| :--------- | :-------------------------------------- | -----: | :-------------- |
| `bgesmall` | `BAAI/bge-small-en-v1.5`                | ~33.4M | base + baseline |
| `minilm`   | `sentence-transformers/all-MiniLM-L6-v2`| ~22.7M | base + baseline |
| `micro`    | `TaylorAI/bge-micro-v2`                 | ~17.4M | base + baseline |

**Setup (held constant; only model × epoch vary):**
- Train data: `triplets_filtered/` (Step-4 judge output after `filter_negatives`).
- Loss: `MNRL` (plain — runs on M1 `mps`; cached loss is CUDA-only).
- Split: **by query** (`split.py`), 80/20, seeded — no query leak across train/test.
- Eval corpus: **book-scoped** via `--corpus-from-test` (the NCSA book the training
  data covers), top-10 cosine retrieval, macro-avg with percentile bootstrap 95% CIs.
- Per-epoch snapshots saved (`--save-each-epoch`); each snapshot evaluated separately,
  so epoch 1/2/3 are independent eval points, not a single end-state.
- `micro` additionally trained with `--eval-loss` (held-out MNRL eval loss per epoch).

**Reproduce:**
```bash
source ~/.venvs/.rag/bin/activate
bash experiments/run_epoch_sweep.sh   # bgesmall + minilm -> sweep.log
bash experiments/run_micro.sh         # micro (+ --eval-loss) -> sweep_micro.log, regenerates all plots
```

**Artifacts:**
- `experiments/epoch_sweep/results/<tag>_epoch<N>.json` — per-(model,epoch) eval.
- `experiments/epoch_sweep/sweep.log`, `sweep_micro.log` — training stdout (loss lines).
- `experiments/epoch_sweep/plots/{overall_metrics,loss_curves,delta,by_style_recall5}.png`,
  `summary.csv`.

---

## 2. Results (from `summary.csv`)

Baseline vs **best** fine-tuned epoch, points = ×100. Δ vs baseline in parentheses.

| model    | baseline R@5 | best R@5 (ep) | best NDCG@10 (ep) | ΔR@5  | ΔNDCG |
| :------- | -----------: | :------------ | :---------------- | ----: | ----: |
| bgesmall | 84.31        | **86.34** (1) | **83.80** (1)     | +2.03 | +1.60 |
| minilm   | 79.99        | **84.29** (1) | **82.10** (1)     | +4.30 | +4.11 |
| micro    | 79.39        | **83.83** (2) | **81.70** (2)     | +4.44 | +4.15 |

**Findings:**
1. **All three models beat baseline** on every metric at their peak epoch.
2. **Bigger = higher absolute, smaller gain.** `bgesmall` is the strongest retriever
   outright but improves least (+2.0 R@5) — it is already near ceiling. The smaller
   `minilm`/`micro` have more headroom and gain ~**+4.3 / +4.4 R@5**.
3. **Overfitting onset is size-dependent.** `bgesmall` and `minilm` peak at **epoch 1**
   and decline after; `micro` peaks at **epoch 2**. So "one epoch suffices" holds for the
   larger two; the smallest tolerates a second epoch.
4. **`micro` is the value pick:** at 17.4M params it nearly matches `bgesmall`'s
   fine-tuned NDCG (81.7 vs 83.8) from a much weaker baseline.

---

## 3. The "flat validation loss" question (resolved — not a bug)

Two distinct things were being conflated:

- **The flat dashed line in `loss_curves.png` is `baseline NDCG@10`**, drawn as a
  constant horizontal reference (`ax2.axhline`). The off-the-shelf model is never
  trained, so it cannot move. Flat by design, not a loss, not a bug.
- **A real held-out MNRL eval loss** was only logged for `micro` (`--eval-loss`):
  `0.3094 → 0.3089 → 0.3122` (epochs 1→2→3). Near-flat (Δ < 0.004) while training loss
  plunges. This is **expected for in-batch contrastive loss**, because:
  - MNRL loss is dominated by in-batch negatives; on train the model partly *memorizes*
    the specific rows, on eval (unseen anchors) it cannot — so eval barely moves.
  - BGE-family models already separate query/positive from negatives well, so eval loss
    starts near its floor with little headroom.
  - It only contrasts the few negatives in each eval batch, vs NDCG@10 which ranks the
    whole corpus — so it is a low-amplitude signal.
  - Sanity check it is alive, not stuck: the three values **differ** (a broken eval gives
    identical numbers), and the **minimum (epoch 2) coincides with `micro`'s NDCG peak
    (epoch 2)** — weak but consistent.

**Takeaway:** trust **eval NDCG@10** for the overfitting call (it peaks early and
declines); MNRL eval loss is too blunt and was dropped from the poster figure.

---

## 4. Plot scripts & the title changes made this session

- `finetune/plot_epoch_sweep.py` → `overall_metrics.png` (+ `delta`, `by_style`, csv).
  - Title now **"Fine-tuned vs Baseline Metrics by Epoch (95% CI)"**.
  - Model lines/legend ordered **largest→smallest** (`_MODEL_SIZE`) to match the loss plot.
- `finetune/plot_loss_curves.py` → `loss_curves.png`.
  - Title now **"Training Loss / Eval Metric by Epoch (descending order of model size)"**.
  - Subplots ordered **largest→smallest** (`bgesmall, minilm, micro`).
  - Val-loss line **removed** (off-title; see §3). Series are training loss (left axis),
    eval NDCG@10 + baseline reference (right axis).

Regenerate + sync into the poster:
```bash
RES=experiments/epoch_sweep/results; PLOTS=experiments/epoch_sweep/plots
python -m finetune.plot_epoch_sweep --results-dir "$RES" --out-dir "$PLOTS"
python -m finetune.plot_loss_curves \
  --sweep-log experiments/epoch_sweep/sweep.log experiments/epoch_sweep/sweep_micro.log \
  --results-dir "$RES" --out "$PLOTS/loss_curves.png" --metric ndcg@10
cp "$PLOTS/overall_metrics.png" "$PLOTS/loss_curves.png" poster/
```

---

## 5. Poster (`poster/poster.tex`)

A0 **landscape** `beamerposter`, Boğaziçi navy theme. Builds clean (no overfull) with:
```bash
cd poster && latexmk -pdf poster.tex   # or: pdflatex poster.tex
```

**Layout (5 bands):** header (BOUN logo · title · 2nd-logo placeholder) → 3-column zone
(`Introduction`+`Methodology` | **flow diagram centerpiece** | `Models`+`Conclusions`+
`Future Work`+`References`) → full-width `Results & Models` band (the two wide figures live
here) → footer.

**Assets in `poster/`:** `boun-logo.png`, `flow.jpg` (6-stage pipeline, centerpiece),
`overall_metrics.png`, `loss_curves.png`, `methodology.txt` (source text),
`poster_instruction.pdf` (CMPE 492 rules: A0 landscape, 85/36/24/18 pt minimums).

**Design constraint:** the flow image is steep portrait (ratio 0.57) and A0 is short
(841 mm), so it is sized by **height** (`height=0.42\paperheight`), not width — otherwise it
overflows the page.

**Remaining placeholders** (red `[...]` in the PDF, search `\todo` in the `.tex`): headline
recall deltas in the Results bullets, final Conclusion sentence, all Future Work items, the
3 references, 2nd logo, footer email/GitHub. The figures themselves now hold the real
3-model numbers — the bullets just need transcribing from §2.

---

## 6. Open / next

- Fill the poster placeholders from §2 numbers before printing (allow ~3 days to print).
- Optional: headline **bge-m3** fine-tune on GCP L4 (cached loss, larger batch) — needs CUDA;
  the M1 sweep used `mnrl` on `mps`.
- Optional: add a reranker stage; end-to-end RAG answer-quality eval.
