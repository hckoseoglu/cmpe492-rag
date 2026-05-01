# Codebase Overview

## What This Does

Synthetic dataset generation pipeline for fine-tuning a RAG-based AI personal trainer's retriever. The pipeline processes fitness/exercise science PDFs through 4 stages:

1. **Agentic Chunking** (implemented) -- LLM decomposes PDF text into atomic propositions, then groups them into thematic chunks with summaries. Output: JSONL files in `./chunks/`.
2. **Positive Pair Generation** (implemented) -- For each chunk, generates one formal and one informal user query (separate LLM calls). A judge LLM validates each query against the chunk; on rejection, the failure reason is fed back to the generator (up to 2 retries). Non-substantive chunks (TOC, author bios, copyright pages, references, captions) are detected and skipped. Output: JSONL files in `./pairs/`.
3. **Hard Negative Candidate Mining** (implemented) -- For each query, hybrid search (BM25 + BGE-M3 dense, fused with RRF) over the chunk corpus produces top-K candidates with the source chunk excluded. Output: JSONL files in `./candidates/`.
4. **Expert Judge Validation** (implemented) -- A judge LLM labels each candidate as `positive` or `hard_negative`. The source chunk is treated as a positive without a judge call. Output: per-query records and exploded triplets in `./triplets/`. Currently uses the same Gemma 2 9B server as Step 2; switch to DeepSeek-R1 by restarting vLLM and updating `LLM_MODEL`.

Final output: `(Query, Positive, Hard Negative)` triplets for fine-tuning `BAAI/bge-m3` with MultipleNegativesRankingLoss.

## Project Structure

```
dataset-generation/
├── chunker.py          # Step 1 orchestrator + CLI (chunking)
├── pair_generator.py   # Step 2 orchestrator + CLI (positive-pair generation)
├── hybrid_search.py    # Step 3 orchestrator + CLI (hard-negative candidate mining)
├── negative_judge.py   # Step 4 orchestrator + CLI (judge labelling -> triplets)
├── retrieval/          # Step 3 building blocks (corpus, BM25, dense, RRF)
├── llm_client.py       # Ollama/vLLM abstraction (OpenAI-compatible API)
├── pdf_loader.py       # PDF text extraction + batching
├── propositions.py     # LLM proposition extraction
├── grouper.py          # Proposition grouping + summarization
├── checkpoint.py       # Shared checkpoint helpers (load/save/path)
├── config.py           # Configuration dataclass
├── requirements.txt    # pypdf, openai, rank_bm25, sentence-transformers, torch, numpy
├── setup_local.sh      # Ollama + gemma2:9b setup script (local)
├── serve.sh            # vLLM serving script (GCP)
├── tests/              # Judge + hybrid-search test suite (cases + runners)
├── chunks/             # Step 1 output: JSONL files (generated)
├── pairs/              # Step 2 output: pairs + _skipped + _failed JSONL (generated)
├── candidates/         # Step 3 output: per-query candidate JSONL (generated)
├── triplets/           # Step 4 output: per-query records + exploded triplets (generated)
├── .cache/embeddings/  # Cached BGE-M3 corpus embeddings keyed by content hash
├── checkpoints/        # Resumption state (generated)
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

For each `(query, candidate)` pair from Step 3, ask a judge LLM whether the
candidate fully answers the query. The source chunk is always positive and never
sent to the judge (it was the original answer by construction).

### Judge contract
- One LLM call per candidate, structured JSON output via `LLMClient.chat_structured`.
- Schema: `{"label": "positive" | "hard_negative", "reason": "..."}`.
- "positive" = the chunk contains the COMPLETE answer; partial answers and
  topically-similar definitions are `hard_negative`.

### Run (Local — Ollama)
```bash
cd dataset-generation
python negative_judge.py \
  --candidates-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl" \
  --limit 5

# Full file
python negative_judge.py \
  --candidates-file "NCSA_Essentials_of_ Strength_Training_and_Conditioning.jsonl"
```

### Switching to DeepSeek-R1 on GCP
Per `CLAUDE.md` §1, the planned judge for this step is
`DeepSeek-R1-Distill-Llama-8B`. To switch:
```bash
# Shell A — vLLM server with the new model
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --max-model-len 8192 --port 8000 --gpu-memory-utilization 0.90

# Shell B — point negative_judge.py at it (note: DeepSeek-R1 is NOT a Gemma model)
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export LLM_API_KEY="dummy"
export LLM_IS_GEMMA="false"
python negative_judge.py --candidates-file "..."
```
No code changes needed — the model swap is purely env-driven.

### Output Layout (`./triplets/`)
Two artifacts per book, side by side:

`triplets/<file>.jsonl` — one record per query, easy to inspect:
```json
{"query": "...", "style": "formal",
 "source_chunk_id": "NCSA_..._0042",
 "positives": ["NCSA_..._0042", "NCSA_..._0117"],
 "hard_negatives": ["NCSA_..._0089", "NCSA_..._0288"]}
```

`triplets/<file>.triplets.jsonl` — exploded `(query, positive, hard_negative)`
rows ready for `MultipleNegativesRankingLoss`. The source-chunk positive is
referenced by id in the per-query record only — its content lives in
`chunks/<file>.jsonl` and the trainer rejoins on `chunk_id`.

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
- **Step-4 judge calibration** (`tests/test_negative_judge.py`) — curated cases
  in `tests/negative_judge_cases.py` (intentionally distinct from the prompt
  few-shots) measure the judge's accuracy on positives vs partial / topical /
  definition-only `hard_negative` cases. Aim for ≥85% before any large run.
  ```bash
  python -m tests.test_negative_judge --save
  ```

## What's Next

### Step 5: Retriever Fine-Tuning
- Fine-tune `BAAI/bge-m3` on the triplet dataset
- Loss: MultipleNegativesRankingLoss (MNRL)
- Batch size: 32-64, on NVIDIA L4 (24GB VRAM)
