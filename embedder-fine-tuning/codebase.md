# Codebase Overview

## What This Does

Synthetic dataset generation + retriever fine-tuning pipeline for a RAG-based AI personal trainer. The pipeline processes fitness/exercise science PDFs through 5 stages:

1. **Agentic Chunking** (implemented) -- LLM decomposes PDF text into atomic propositions, then groups them into thematic chunks with summaries. Output: JSONL files in `./chunks/`.
2. **Positive Pair Generation** (implemented) -- For each chunk, generates one formal and one informal user query (separate LLM calls). A judge LLM validates each query against the chunk; on rejection, the failure reason is fed back to the generator (up to 2 retries). Non-substantive chunks (TOC, author bios, copyright pages, references, captions) are detected and skipped. Output: JSONL files in `./pairs/`.
3. **Hard Negative Candidate Mining** (implemented) -- For each query, hybrid search (BM25 + BGE-M3 dense, fused with RRF) over the chunk corpus produces top-K candidates with the source chunk excluded. Output: JSONL files in `./candidates/`.
4. **Expert Judge Validation** (implemented) -- A judge LLM labels each candidate as `positive`, `hard_negative`, or `irrelevant`. The source chunk is treated as a positive without a judge call, and its content is hydrated from `chunks/<file>.jsonl` so source positives contribute triplets directly. Output (in `./triplets/`): per-query records, exploded `(query, positive, hard_negative)` triplets, and a per-call `judge_debug.jsonl` log for prompt iteration. Always uses **`DeepSeek-R1-0528-Qwen3-8B`** (Ollama tag `deepseek-r1:8b` locally; HF id `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` on GCP/vLLM) — its chain-of-thought goes into the schema's `reason` field, which is what makes the prompt's self-check rules actually fire. Calibrated at POSITIVE = 6/6 (100%) on the test suite; gemma2:9b was 4/6 with self-contradictory false hard_negatives that would poison MNRL training.
5. **Retriever Fine-Tuning & Eval** (implemented) -- `finetune/` package fine-tunes `BAAI/bge-m3` on the labelled triplets with `MultipleNegativesRankingLoss` and evaluates against the off-the-shelf baseline. Train/test split by `source_chunk_id` (queries are 1:1 with source chunks); each query's k positives × m hard_negatives expand to k×m training rows, all sharing the same anchor text. `BatchSamplers.NO_DUPLICATES` enforces at-most-one-row-per-query per batch — without it, MNRL would treat a query's other positives as in-batch negatives and push them away. Eval is multi-relevant Recall@{1,5,10} + NDCG@10 over the combined cross-book corpus, with percentile bootstrap 95% CIs.

Final output: a fine-tuned `BAAI/bge-m3` checkpoint under `checkpoints/bge-m3-finetuned-<timestamp>/final/` plus `<run-dir>/results/comparison.json` reporting the lift over baseline.

## Project Structure

```
dataset-generation/
├── chunker.py          # Step 1 orchestrator + CLI (chunking)
├── pair_generator.py   # Step 2 orchestrator + CLI (positive-pair generation)
├── hybrid_search.py    # Step 3 orchestrator + CLI (hard-negative candidate mining)
├── negative_judge.py   # Step 4 orchestrator + CLI (judge labelling -> triplets)
├── retrieval/          # Step 3 building blocks (corpus, BM25, dense, RRF)
├── finetune/           # Step 5 — MNRL fine-tuning + retrieval eval
│   ├── dataset.py      # load_judge_records, explode_to_triplets, dedup feasibility check
│   ├── split.py        # train_test_split_by_chunk_id (queries are 1:1 with source chunks)
│   ├── metrics.py      # multi-relevant Recall@k, NDCG@k, percentile bootstrap CI
│   ├── train.py        # SentenceTransformerTrainer + MNRL + NO_DUPLICATES sampler
│   └── evaluate.py     # baseline vs fine-tuned over combined cross-book corpus
├── llm_client.py       # Ollama/vLLM abstraction (OpenAI-compatible API)
├── pdf_loader.py       # PDF text extraction + batching
├── propositions.py     # LLM proposition extraction
├── grouper.py          # Proposition grouping + summarization
├── checkpoint.py       # Shared checkpoint helpers (load/save/path)
├── config.py           # Configuration dataclass
├── requirements.txt    # pypdf, openai, rank_bm25, sentence-transformers>=3, torch, numpy, datasets, accelerate
├── setup_local.sh      # Ollama + gemma2:9b setup script (local)
├── serve.sh            # vLLM serving script (GCP)
├── tests/              # Judge + hybrid-search + finetune smoke test suite
├── chunks/             # Step 1 output: JSONL files (generated)
├── pairs/              # Step 2 output: pairs + _skipped + _failed JSONL (generated)
├── candidates/         # Step 3 output: per-query candidate JSONL (generated)
├── triplets/           # Step 4 output: per-query records + exploded triplets (generated)
├── .cache/embeddings/  # Cached BGE-M3 corpus embeddings keyed by (model_name, ids, contents)
├── checkpoints/        # Step 5 model checkpoints + Step 1-4 resumption state (generated)
│                       #   bge-m3-finetuned-<timestamp>/final/        — Step 5 output model
│                       #   bge-m3-finetuned-<timestamp>/test_queries.jsonl  — frozen eval split
│                       #   bge-m3-finetuned-<timestamp>/results/comparison.json — eval table
│                       #   bge-m3-finetuned-latest                    — symlink to most recent run
└── CLAUDE.md           # Full pipeline design doc
```

## Setup & Run

### Prerequisites
- Python 3.10+
- macOS (M1/M2) for local dev, or GCP NVIDIA L4 for production

### Ollama Setup (Local)

**First-time setup** (installs Ollama + pulls the model):
```bash
cd dataset-generation
bash setup_local.sh
```

**Starting Ollama in subsequent sessions:**
```bash
# Option 1: Run as a background server
ollama serve

# Option 2: Run the model directly (starts server automatically)
ollama run gemma2:9b
# Type /bye to exit the interactive chat — server keeps running
```

**Useful Ollama commands:**
```bash
ollama list              # Show downloaded models
ollama ps                # Show currently loaded/running models
ollama stop gemma2:9b    # Unload model from memory
curl http://localhost:11434  # Check if server is running
```

> Note: Ollama server must be running before you start the chunking pipeline.
> The model stays loaded in memory (~5.4GB) after the first request.
> To free memory, run `ollama stop gemma2:9b` or quit the server.

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Run (Local -- Ollama)
```bash
# Test on one PDF
python chunker.py --pdf "progression_models_in_resistance_training.pdf"

# Full corpus
python chunker.py

# Resume after interruption
python chunker.py --resume
```

### Run (GCP -- vLLM)
```bash
LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=google/gemma-2-9b-it LLM_IS_GEMMA=true python chunker.py
```

> **Note:** `LLM_IS_GEMMA=true` is required when using Gemma models (via vLLM or Ollama).
> Gemma does not support the `system` role in its chat template — setting this flag merges
> the system prompt into the first user message instead.

### Output Format
One JSONL file per PDF in `./chunks/`:
```json
{"id": "progression_models_0001", "content": "...", "summary": "..."}
```

## Step 2: Positive Pair Generation (`pair_generator.py`)

For each chunk, generate **two** questions — one **formal**, one **informal** — in
separate LLM calls, validate each with a **judge** LLM, and retry with judge feedback up
to **2 times** per style. The same Gemma 2 9B model serves both roles in this iteration.
Non-substantive chunks (TOC, author bios, copyright, references, captions, book/edition
meta-descriptions) are detected by the generator and skipped at the chunk level.

### Generator
Decides `is_content` first. If non-content, emits a `skip_reason` from:
`table_of_contents`, `author_credentials`, `copyright_notice`, `references`,
`figure_caption`, `other_metadata`. Otherwise, produces one question in the requested
style.

### Judge — five hard rules (ALL must pass)
1. **ANSWERABILITY** — chunk fully and definitively answers the question.
2. **SPECIFICITY** — question targets info unique to this chunk.
3. **NO_ATTRIBUTION** — never names institutions (NSCA, ACSM, …), people (researchers,
   authors), specific studies/journals, or specific books/resources.
4. **USER_PERSONA** — never references the source itself ("this book", "the third
   edition", "chapters 9 and 10", "what's new in…").
5. **STYLE MATCH** — register matches the requested style (formal vs informal).

On rejection, the judge's `failure_reason` is fed back to the generator in a `RETRY
FEEDBACK` block. After the retry budget is exhausted, the chunk goes to `_failed.jsonl`
(no pair is emitted).

### Run (Local -- Ollama)
```bash
# Smoke test on a small subset of NCSA chunks
cd dataset-generation
python pair_generator.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl" \
  --limit 5

# Full file
python pair_generator.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl"

# Resume after interruption
python pair_generator.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl" --resume

# Process all chunks files (later)
python pair_generator.py
```

CLI flags: `--chunks-file`, `--resume`, `--style {both,formal,informal}` (default `both`),
`--limit N`.

### Run (GCP -- vLLM)
Two shells. **Shell A** runs the vLLM server (blocks); **shell B** runs the pair
generator with the same env vars `serve.sh` sets internally.

Shell A:
```bash
cd dataset-generation
bash serve.sh
# wait for: "Uvicorn running on http://0.0.0.0:8000"
```
Sanity check from another shell:
```bash
curl -s http://localhost:8000/v1/models   # should list google/gemma-2-9b-it
```

Shell B:
```bash
cd dataset-generation
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="google/gemma-2-9b-it"
export LLM_API_KEY="dummy"
export LLM_IS_GEMMA="true"

# Smoke test
python pair_generator.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl" \
  --limit 5

# Full file
python pair_generator.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl"
```

> Note: `serve.sh` currently launches vLLM with `--gpu-memory-utilization 0.50` (the
> CLAUDE.md design doc lists 0.90). The 0.50 value is intentional for the current GCP
> setup; raise it only if VRAM is unconstrained.

### Output Layout (`./pairs/`)
- `<chunks_filename>.jsonl` — successful pairs, two lines per content chunk:
  ```json
  {"chunk_id": "NCSA_..._0042", "question": "...", "style": "formal", "attempts": 1}
  ```
- `_skipped.jsonl` — chunks classified as non-content (chunk-level skip; both styles skipped):
  ```json
  {"chunk_id": "...", "skip_reason": "table_of_contents", "summary": "...", "source_file": "..."}
  ```
- `_failed.jsonl` — chunks where the judge rejected after all retries; full attempt trail kept
  for prompt debugging.

Checkpoint: `./checkpoints/<stem>_pairs.json` stores `processed_chunk_ids` so `--resume`
picks up exactly where the last run stopped.

### Testing the Judge (`tests/test_judge.py`)
Curated cases at `tests/judge_cases.py` exercise each judge rule with a mix of valid and
invalid examples. Every test case is intentionally distinct from the prompt few-shots so
the runner measures rule-application, not memorization.

```bash
# Full suite — run before any large pair-generation job as a calibration baseline
python -m tests.test_judge --save

# Iterate on a single rule while tuning the prompt
python -m tests.test_judge --category USER_PERSONA
```

Categories: `VALID`, `ANSWERABILITY`, `SPECIFICITY`, `NO_ATTRIBUTION`, `USER_PERSONA`,
`STYLE`. The runner prints per-category accuracy + overall, and `--save` writes
`tests/judge_results.json` for cross-revision comparison. Aim for ≥85% overall before
running a long job; any single rule under ~70% means that prompt section needs work.

## Step 3: Hard Negative Candidate Mining (`hybrid_search.py`)

For each `(chunk_id, question, style)` row in `pairs/<file>.jsonl`, run a global
hybrid search over the chunk corpus and emit the top-K candidates as hard-negative
candidates. The source chunk is always excluded from the results — by definition it
is the positive for its own query.

### Retrieval pipeline
- **Corpus:** `chunks/<file>.jsonl`, with rows whose `chunk_id` appears in
  `pairs/_skipped.jsonl` (and matches `source_file`) filtered out. Failed chunks
  (`_failed.jsonl`) stay — they're real content; the judge just couldn't get a
  clean question out of them.
- **Sparse:** classic BM25 via `rank_bm25` over a lowercase + alphanumeric
  tokenisation. No stemming, no stopwording — keeps behaviour predictable.
- **Dense:** `BAAI/bge-m3` via `sentence-transformers`. Embeddings are normalised so
  cosine similarity is just a dot product. Corpus embeddings are cached to
  `.cache/embeddings/<stem>.<hash>.npy` keyed on a hash of `(model, ids, contents)`,
  so re-runs over an unchanged file skip the encode step entirely.
- **Fusion:** Reciprocal Rank Fusion (RRF, `k=60`). Each list contributes
  `1 / (k + rank)`. RRF was chosen over weighted sums because it ignores score
  magnitudes — BM25 scores are unbounded and cosine sits in `[-1, 1]`, so any
  weighted-sum tuning would be brittle.
- **Pool:** top-50 from each list, fused, source chunk dropped, top-5 emitted.

### Run (Local — Ollama / CPU embedder)
```bash
cd dataset-generation
pip install -r requirements.txt   # first run pulls ~2GB BGE-M3 weights

# Smoke test on 5 queries
python hybrid_search.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl" \
  --limit 5 --device cpu

# Full file
python hybrid_search.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl"

# Resume after interruption
python hybrid_search.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl" --resume
```

### Run (GCP — CUDA embedder)
The embedder runs on the same L4 as the LLM server. CUDA is auto-detected, no flag
needed.
```bash
cd dataset-generation
python hybrid_search.py \
  --chunks-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl"
```

CLI flags: `--chunks-file` (required), `--pairs-file` (defaults to chunks-file
basename), `--top-k` (default 5), `--limit N`, `--resume`,
`--device {auto,cpu,cuda,mps}`.

### Output Layout (`./candidates/`)
One line per query in `candidates/<chunks_filename>.jsonl`:
```json
{
  "chunk_id": "NCSA_..._0042",
  "question": "...",
  "style": "formal",
  "source_chunk_id": "NCSA_..._0042",
  "candidates": [
    {"chunk_id": "...", "content": "...", "summary": "...",
     "bm25_rank": 3, "dense_rank": 1, "rrf_score": 0.0312}
  ]
}
```

Checkpoint: `checkpoints/<stem>_candidates.json` keyed by `f"{chunk_id}::{style}"`.

## Step 4: Expert Judge Validation (`negative_judge.py`)

For each `(query, candidate)` pair from Step 3, ask a judge LLM to label the
candidate as `positive`, `hard_negative`, or `irrelevant`. The source chunk is
always a positive by construction and is never sent to the judge.

### Judge contract
- One LLM call per candidate, structured JSON output via
  `LLMClient.chat_structured`.
- **Schema (reason-first)** — JSON schema property order is significant under
  strict structured outputs; `reason` is listed first so the model writes its
  justification BEFORE committing to a label (chain-of-thought before verdict):
  ```json
  {"reason": "...", "label": "positive" | "hard_negative" | "irrelevant"}
  ```
- **Definitions:**
  - `positive` — CANDIDATE contains a complete answer to QUERY, sufficient on
    its own. Qualitative-but-domain-correct phrasing (e.g. "moderate loads"
    for a hypertrophy-intensity question) IS positive — don't penalise correct
    domain-typical phrasing for lacking numbers.
  - `hard_negative` — CANDIDATE doesn't actually answer QUERY but shares
    keyword/topical overlap. The classic shape is the **neighbouring-question
    pattern**: a complete answer to a *different but adjacent* question on the
    same topic — same exercise / nutrient / system, different goal, scope,
    population, or facet. Same vocabulary, wrong answer.
  - `irrelevant` — neither answers QUERY nor shares meaningful keyword/topical
    overlap. Off-topic.
- **Hard rules** (from the prompt): verdicts must be GROUNDED in CANDIDATE's
  actual text (quote the phrase); multi-part queries require every sub-part
  addressed for `positive`; definition-of-X chunks for a topic in QUERY are
  usually `hard_negative` when QUERY asks for a prescription; default on
  malformed verdict is `irrelevant` (conservative — keeps noise out of the MNRL
  training set).

### Run (Local — Ollama, deepseek-r1:8b)
For prompt iteration and the calibration test on a laptop. Q4_K_M (~5.2 GB on
disk, ~6-7 GB resident with 8K ctx). Tested on a 16 GB M1 MBP.

```bash
# One-time
ollama pull deepseek-r1:8b
ollama serve   # if not already running

cd dataset-generation
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="deepseek-r1:8b"
export LLM_API_KEY="ollama"
export LLM_IS_GEMMA="false"
export OLLAMA_KEEP_ALIVE="0"   # unload between calls on RAM-tight machines

# Smoke run
python negative_judge.py \
  --candidates-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl" \
  --limit 5

# Full file
python negative_judge.py \
  --candidates-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl"
```

Inference is slower than gemma2:9b (~40 s/call vs ~7 s on M1) because the
reasoning chain-of-thought goes into the `reason` field before the `label` is
committed. That's what makes the self-check rules in the prompt fire.

### Run (GCP — vLLM, DeepSeek-R1-0528-Qwen3-8B)
Stop the gemma2 vLLM server first to free VRAM (the L4 doesn't fit both at the
same precision).
```bash
# Shell A — vLLM with DeepSeek
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 8192 \
  --port 8000 \
  --gpu-memory-utilization 0.90

# Shell B — Step 4
cd dataset-generation
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
export LLM_API_KEY="dummy"
export LLM_IS_GEMMA="false"
python negative_judge.py --candidates-file "..."
```
No code changes needed — the model swap is purely env-driven.

### Output Layout (`./triplets/`)
Three artifacts per book.

`triplets/<file>.jsonl` — one record per query, easy to inspect:
```json
{"query": "...", "style": "formal",
 "source_chunk_id": "NCSA_..._0042",
 "positives": ["NCSA_..._0042", "NCSA_..._0117"],
 "hard_negatives": ["NCSA_..._0089", "NCSA_..._0288"],
 "irrelevants": ["NCSA_..._0901"]}
```

`triplets/<file>.triplets.jsonl` — exploded `(query, positive, hard_negative)`
rows ready for `MultipleNegativesRankingLoss`. Source-positive content is
hydrated from `chunks/<file>.jsonl` via `load_chunk_contents()` so the source
positive contributes triplets directly — no downstream rejoin step needed.

`triplets/<file>.judge_debug.jsonl` — one row per judge call, written
incrementally so the file is inspectable mid-run:
```json
{"query": "...", "style": "formal",
 "source_chunk_id": "NCSA_..._0042",
 "source_content": "...",
 "candidate_chunk_id": "NCSA_..._0089",
 "candidate_content": "...",
 "label": "hard_negative",
 "reason": "...",
 "raw_verdict": {"reason": "...", "label": "hard_negative"}}
```
`source_content` is recorded for the human reviewer doing prompt iteration; the
judge itself never sees it. Use this file to find verdicts you disagree with
and feed them back into the prompt.

Checkpoint: `checkpoints/<stem>_triplets.json` keyed by `f"{chunk_id}::{style}"`.

CLI flags: `--candidates-file` (required), `--limit N`, `--resume`.

### Testing
- **Hybrid-search smoke** (`tests/test_hybrid_search.py`) — synthetic 10-doc corpus
  with a hand-coded paraphrase map standing in for the dense embedder. Verifies BM25
  exact-match ranking, dense paraphrase ranking, RRF consensus, source-exclusion,
  and `top_k` capping. No model download required.
  ```bash
  python -m tests.test_hybrid_search
  ```
- **Step-4 judge calibration** (`tests/test_negative_judge.py`) — runs the live
  judge against `tests/negative_judge_cases.py` (15 hand-labeled cases across
  five categories: `POSITIVE`, `HARD_NEGATIVE_PARTIAL`, `HARD_NEGATIVE_TOPICAL`,
  `HARD_NEGATIVE_DEFINITION`, `IRRELEVANT_OFFTOPIC`). Cases are intentionally
  distinct from the prompt few-shots so the test measures rule-application, not
  memorisation. Aim for ≥85% overall AND POSITIVE = 100% before any large run
  (false hard_negatives in the POSITIVE bucket directly poison MNRL training).
  Backup result snapshots per model live next to the live results file as
  `negative_judge_results.<model-tag>.json`.
  ```bash
  # Step-4 judge env must be exported first (see "Run (Local — Ollama)").
  python -m tests.test_negative_judge --save
  ```

## Step 5: Retriever Fine-Tuning & Eval (`finetune/`)

Fine-tune `BAAI/bge-m3` on the labelled triplets and quantify the lift over the
off-the-shelf model on a held-out test split. Lives in `finetune/`; consumes
`triplets/*.jsonl` (per-query records) and `chunks/*.jsonl` (for content
hydration), produces a checkpoint under `checkpoints/bge-m3-finetuned-<timestamp>/`.

**Train/test split** (`split.py`) — partitions unique `source_chunk_id`s 80/20
with a seeded RNG. Every query is generated from exactly one source chunk in
Step 2, so chunk-level partitioning is equivalent to query-level partitioning;
the splitter raises if a query somehow ends up in both sides. Test queries
carry `relevant_chunk_ids = {source_chunk_id} ∪ judge-positives`, persisted to
`<run-dir>/test_queries.jsonl` so `evaluate.py` consumes the same split.

**Row construction** (`dataset.py`) — for each judge record with k positives
(source + judge-promoted) and m hard_negatives, emit **k × m rows** of shape
`(anchor=query, positive=P_i.content, negative=HN_j.content)`. Records with
empty `hard_negatives` are dropped — MNRL with an empty negative column would
silently inject zeros into the in-batch negative pool. Source content for both
positives and hard_negatives is hydrated from the matching `chunks/<book>.jsonl`.

**Why per-batch query-dedup matters.** MNRL's denominator pulls in every
other row's positive AND every other row's hard_negative. Two same-query rows
in one batch — `(Q, P_a, HN_a)` and `(Q, P_b, HN_b)` — would put P_b in Q's
denominator as a "negative" even though it's a true positive for Q, pushing
Q away from a chunk that actually answers it. `BatchSamplers.NO_DUPLICATES`
(sentence-transformers ≥3.0) defers same-anchor rows to later batches; since
all k×m rows of a query share anchor text, this guarantees at most one row
per query per batch. `assert_dedup_feasibility` warns at load time if any
query has more rows than batches per epoch (its tail rows would starve under
NO_DUPLICATES).

**Training** (`train.py`):
- `MultipleNegativesRankingLoss` (asymmetric).
- `BatchSamplers.NO_DUPLICATES`, batch=32, lr=2e-5, warmup_ratio=0.1, epochs=3, max_seq_length=512.
- fp16 on CUDA; disabled on MPS (unstable) and CPU.
- Effective negatives per anchor = `2N − 2` other-row positives + `N` other-row hard_negatives = 94 at N=32.
- Saves `<run-dir>/final/` (full SentenceTransformer) and refreshes the
  `checkpoints/bge-m3-finetuned-latest` symlink.

```bash
# Production run on GCP L4 (vLLM not needed for training)
python -m finetune.train --device cuda --epochs 3 --batch-size 32

# M1 smoke against real triplets, capped to 50 rows
python -m finetune.train --smoke

# M1 smoke with synthetic in-memory data — needs no real triplets, downloads
# BGE-M3 once, runs ~7 batches in ~80 s. Use this on a fresh M1 to verify the
# training path before the GCP judge run produces data.
python -m finetune.train --synthetic-smoke
```

**Evaluation** (`evaluate.py`) — concatenates every `chunks/<book>.jsonl`
through the existing `retrieval.corpus.load_corpus` (Step-3 skip filter
applies) and embeds the combined corpus with each variant via
`retrieval.dense_index.DenseIndex` (whose embedding cache is keyed on
`(model_name, ids, contents)` — baseline and fine-tuned cache to separate
`.npy` files automatically). Per test query: encode → cosine top-10 → record
where any chunk in `relevant_chunk_ids` lands.

Metrics: **Recall@1, Recall@5, Recall@10, NDCG@10**, multi-relevant binary,
macro-averaged with percentile bootstrap 95% CIs (1k resamples). Per-style
breakdown (`formal` vs `informal`) also reported. Output:
`<run-dir>/results/comparison.json` plus a printed table.

```bash
python -m finetune.evaluate --run-dir checkpoints/bge-m3-finetuned-<timestamp>
# --only baseline    — sanity check the metric infra without needing a fine-tuned model
# --only finetuned   — re-evaluate after manual checkpoint changes
```

### Testing — `tests/test_finetune_smoke.py`

Six unit cases that don't touch the network:
1. Explosion carries `query_id` and skips records with empty `hard_negatives`.
2. `train_test_split_by_chunk_id` produces query-disjoint splits and every
   test query has its source in `relevant_chunk_ids`.
3. Multi-relevant `recall_at_k` / `ndcg_at_k` behave on edge cases.
4. End-to-end `load_judge_records` round-trip against a synthetic
   `chunks/`/`triplets/` pair.
5. `assert_dedup_feasibility` warns but doesn't crash on starvation cases.
6. **The NO_DUPLICATES invariant** — synthesises a dataset where one query
   has 4 rows, runs the actual `NoDuplicatesBatchSampler`, and asserts no
   batch ever contains two same-anchor rows.

The full M1 model-fit smoke runs only when `FINETUNE_SMOKE_FULL=1` is set
(it shells out to `python -m finetune.train --synthetic-smoke`).

```bash
python -m tests.test_finetune_smoke
```
