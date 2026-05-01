# Codebase Overview

## What This Does

Synthetic dataset generation pipeline for fine-tuning a RAG-based AI personal trainer's retriever. The pipeline processes fitness/exercise science PDFs through 4 stages:

1. **Agentic Chunking** (implemented) -- LLM decomposes PDF text into atomic propositions, then groups them into thematic chunks with summaries. Output: JSONL files in `./chunks/`.
2. **Positive Pair Generation** (implemented) -- For each chunk, generates one formal and one informal user query (separate LLM calls). A judge LLM validates each query against the chunk; on rejection, the failure reason is fed back to the generator (up to 2 retries). Non-substantive chunks (TOC, author bios, copyright pages, references, captions) are detected and skipped. Output: JSONL files in `./pairs/`.
3. **Hard Negative Mining** (not yet implemented) -- Hybrid search (BM25 + vector) to find hard negatives per query.
4. **Expert Judge Validation** (not yet implemented) -- DeepSeek-R1 classifies candidates as new positives or hard negatives.

Final output: `(Query, Positive, Hard Negative)` triplets for fine-tuning `BAAI/bge-m3` with MultipleNegativesRankingLoss.

## Project Structure

```
dataset-generation/
├── chunker.py          # Step 1 orchestrator + CLI (chunking)
├── pair_generator.py   # Step 2 orchestrator + CLI (positive-pair generation)
├── llm_client.py       # Ollama/vLLM abstraction (OpenAI-compatible API)
├── pdf_loader.py       # PDF text extraction + batching
├── propositions.py     # LLM proposition extraction
├── grouper.py          # Proposition grouping + summarization
├── checkpoint.py       # Shared checkpoint helpers (load/save/path)
├── config.py           # Configuration dataclass
├── requirements.txt    # pypdf, openai
├── setup_local.sh      # Ollama + gemma2:9b setup script (local)
├── serve.sh            # vLLM serving script (GCP)
├── tests/              # Judge test suite (cases + runner)
├── chunks/             # Step 1 output: JSONL files (generated)
├── pairs/              # Step 2 output: pairs + _skipped + _failed JSONL (generated)
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

## What's Next

### Step 3: Hard Negative Mining
- Build a hybrid search index (BM25 + dense vectors via `BAAI/bge-m3`) over all chunks
- For each successful pair record in `./pairs/<file>.jsonl` (i.e. one query per line, not
  a triplet yet), retrieve top-5 chunks from the global index excluding the chunk
  identified by `chunk_id`
- These top-5 become hard-negative candidates fed to Step 4 for classification

### Step 4: Expert Judge Validation
- Use DeepSeek-R1-Distill-Llama-8B to classify each candidate as relevant (new positive) or irrelevant (hard negative)
- Output final triplets: `(query, positive_chunk, hard_negative_chunk)`

### Step 5: Retriever Fine-Tuning
- Fine-tune `BAAI/bge-m3` on the triplet dataset
- Loss: MultipleNegativesRankingLoss (MNRL)
- Batch size: 32-64, on NVIDIA L4 (24GB VRAM)
