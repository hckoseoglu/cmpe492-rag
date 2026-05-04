
# Synthetic Dataset Generation & Fine-Tuning For RAG Based AI Personal Trainer

This guide outlines the technical setup for generating a synthetic fitness dataset and fine-tuning a RAG pipeline on a **GCP NVIDIA L4 (24GB VRAM)** instance.

---

## 1. Environment & Hardware Setup
* **GPU:** NVIDIA L4 (24GB GDDR6).
* **Inference Engine:** [vLLM](https://github.com/vllm-project/vllm) (Recommended for high-throughput generation)

### Model Selection
| Role | Model | Reason |
| :--- | :--- | :--- |
| **Generator** | `Gemma 2 9B` (4-bit quantized) | Optimization on GCP as strong as Llama 3.1 |
| **Judge** | `DeepSeek-R1-0528-Qwen3-8B` | Superior reasoning for factual verification and error detection |
| **Embedder** | `BAAI/bge-m3` | Multi-vector support (Dense + Sparse) and 8k context window |

### Serving the Generator — Steps 1-3 (Gemma 2 9B)
Used for chunking (Step 1), pair generation + Step-2 judge, and the embedder
isn't an LLM (Step 3 uses `BAAI/bge-m3` directly).

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-9b-it \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 8192 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Steps 1-3 env:
```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="google/gemma-2-9b-it"
export LLM_API_KEY="dummy"
export LLM_IS_GEMMA="true"
```

### Serving the Step-4 Judge (DeepSeek-R1-0528-Qwen3-8B)
The Step-4 negative judge always uses `DeepSeek-R1-0528-Qwen3-8B`. It is a
reasoning model whose chain-of-thought goes into the `reason` field of the
strict JSON schema (which is listed first specifically so this works). Calibrated
on `tests/test_negative_judge.py` at 13/15 = 87% with **POSITIVE 6/6 (100%)** —
no false-positive contamination of the MNRL training set.

**GCP / vLLM** — the L4 has 24 GB VRAM, so Steps 1-3 and Step 4 swap rather than
co-reside (stop the gemma server before starting DeepSeek):
```bash
# Stop the Gemma server first (free VRAM)
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 8192 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Step 4 env (GCP):
```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
export LLM_API_KEY="dummy"
export LLM_IS_GEMMA="false"   # DeepSeek is not a Gemma model
```

**Local / Ollama** — for prompt iteration and the Step-4 calibration test on a
laptop. Q4_K_M (~5.2 GB on disk, ~6-7 GB resident with 8K ctx). Tested working
on a 16 GB M1 MBP.
```bash
ollama pull deepseek-r1:8b
ollama serve   # if not already running
```

Step 4 env (Local):
```bash
export LLM_BASE_URL="http://localhost:11434/v1"   # Ollama default
export LLM_MODEL="deepseek-r1:8b"
export LLM_API_KEY="ollama"
export LLM_IS_GEMMA="false"
export OLLAMA_KEEP_ALIVE="0"   # unload between calls on RAM-tight machines
```

Inference is slower than gemma2:9b (~40 s/call vs ~7 s on M1) because the model
emits its chain-of-thought into the `reason` field before committing to a
`label`. That's the feature, not a bug — it's what makes the self-check rules
in the prompt actually fire.

---

## 2. Dataset Generation Pipeline

### Step 1: Agentic Chunking & Summarization
1.  **Extract:** Load fitness documents (PDF/MD).
2.  **Chunk:** Use agentic logic to split by semantic concept.
3.  **Augment:** For each chunk, generate a 1-2 sentence summary.
4.  **Store:** Create a JSONL object: `{"id": "...", "content": "...", "summary": "..."}`.

### Step 2: Positive Pair Generation (`pair_generator.py`)

For each chunk, generate two questions — one **formal**, one **informal** — in **separate**
LLM calls so styles don't bleed. Each candidate question is then validated by a **judge**
LLM; on rejection, the failure reason is fed back to the generator (up to **2 retries**
per style). Both generator and judge are served by the same **Gemma 2 9B** vLLM server in
this iteration. (DeepSeek-R1 remains the planned model for Step 4 hard-negative
validation, not for the Step 2 pair-generation judge.)

**Generator** (`generate_question`)
- First decides whether the chunk has substantive content. Non-substantive chunks are
  skipped at the chunk level (both styles) with one of six labels:
  `table_of_contents`, `author_credentials`, `copyright_notice`, `references`,
  `figure_caption`, `other_metadata` (the last covers book/edition meta-descriptions).
- If substantive, emits one question in the requested style (`formal` = textbook tone;
  `informal` = casual gym-goer voice).

**Judge** (`judge_question`) — five hard rules, ALL must pass:
1. **ANSWERABILITY** — the chunk fully and definitively answers the question.
2. **SPECIFICITY** — the question targets info unique to this chunk (no generic trivia).
3. **NO_ATTRIBUTION** — never names institutions (NSCA, ACSM, …), people (researchers,
   authors), specific studies/journals, or specific books/resources.
4. **USER_PERSONA** — never references the source itself ("this book", "the third
   edition", "chapters 9 and 10", "what's new in…").
5. **STYLE MATCH** — register matches the requested style.

**Outputs** (in `./pairs/`):
- `<chunks_filename>.jsonl` — `{chunk_id, question, style, attempts}`, two lines per
  content chunk (one formal, one informal).
- `_skipped.jsonl` — chunk-level skips with the skip_reason.
- `_failed.jsonl` — chunks the judge rejected after all retries; the full attempt trail
  is preserved for prompt debugging.

**Testing the judge** — `tests/test_judge.py` runs the live judge against curated cases
in `tests/judge_cases.py` (each case is intentionally distinct from the prompt few-shots
so we measure rule-application, not memorization). Use it as a calibration baseline
before any large run:
```bash
python -m tests.test_judge --save
```

### Step 3: Hybrid Search & Hard Negative Mining (`hybrid_search.py`)

For each `(chunk_id, question, style)` row in `pairs/<file>.jsonl`, retrieve the
top-K candidate chunks from the same book that the source chunk does NOT include.
These candidates feed the Step 4 judge. The source chunk is always excluded — by
construction it is the positive for its own query.

**Corpus filtering**
- The corpus is `chunks/<file>.jsonl`. Chunk ids that appear in
  `pairs/_skipped.jsonl` (matched on `source_file`) are dropped — they were
  classified as non-content in Step 2.
- `_failed.jsonl` chunks STAY in the corpus. They are real content; the Step 2
  judge just couldn't extract a clean question from them, but they still might
  be a relevant hit for someone else's question.

**Sparse retriever** — `rank_bm25` over a lowercase + alphanumeric tokenisation.
No stemming, no stopwords — keeps behaviour predictable and easy to reason about.

**Dense retriever** — `BAAI/bge-m3` via `sentence-transformers`. Embeddings are
L2-normalised so cosine similarity reduces to a dot product. Corpus embeddings
are cached to `.cache/embeddings/<stem>.<hash>.npy`, keyed on
`hash(model, ids, contents)` — re-runs over an unchanged file skip the encode
entirely.

**Fusion** — Reciprocal Rank Fusion with `k=60`. Each list contributes
`1 / (k + rank)`. RRF was chosen over weighted-sum because it ignores score
magnitudes — BM25 scores are unbounded and cosine sits in `[-1, 1]`, so any
weighted-sum tuning would be brittle.

**Pool / top-K** — `candidate_pool_size=50` from each list, fused, source chunk
dropped, top-`top_k=5` emitted. Both constants live in `config.py`.

**Output** (in `./candidates/`):
- `<chunks_filename>.jsonl` — one line per query:
  ```json
  {"chunk_id": "...", "question": "...", "style": "formal",
   "source_chunk_id": "...",
   "candidates": [{"chunk_id": "...", "content": "...", "summary": "...",
                   "bm25_rank": 3, "dense_rank": 1, "rrf_score": 0.0312}]}
  ```

Checkpoint: `checkpoints/<stem>_candidates.json` keyed by `f"{chunk_id}::{style}"`;
`--resume` skips processed queries.

**Testing** — `tests/test_hybrid_search.py` runs against a synthetic 10-document
corpus with a hand-coded paraphrase map standing in for the dense embedder
(no model download required). Verifies BM25 exact-match ranking, dense
paraphrase ranking, RRF consensus, source-exclusion, and `top_k` capping.
```bash
python -m tests.test_hybrid_search
```

### Step 4: Expert Judge Validation (`negative_judge.py`)

For each `(query, candidate)` from Step 3, ask a judge LLM to label the candidate
as `positive`, `hard_negative`, or `irrelevant`. Source chunks are positives by
construction and are never sent to the judge. The labelled output feeds Step 5
(retriever fine-tuning).

**Schema (reason-first)** — JSON schema property order is significant under
strict structured outputs; `reason` is listed first so the model writes its
justification BEFORE committing to a label (chain-of-thought before verdict).
```json
{"reason": "...", "label": "positive" | "hard_negative" | "irrelevant"}
```

**Label definitions**
- `positive` — CANDIDATE contains a complete answer to QUERY, sufficient on its
  own. Qualitative-but-domain-correct answers (e.g. "moderate loads" for a
  hypertrophy intensity question) ARE positives — don't penalise correct
  domain-typical phrasing for lacking numbers.
- `hard_negative` — CANDIDATE doesn't actually answer QUERY, but shares
  meaningful surface keyword overlap or topical/semantic similarity with it. The
  classic shape is the **neighbouring-question pattern**: a complete answer to a
  *different but adjacent* question on the same topic — same exercise / nutrient
  / system, but different goal, scope, population, or facet. Same vocabulary,
  wrong answer.
- `irrelevant` — neither answers QUERY nor shares meaningful keyword/topical
  overlap. Off-topic — the kind of chunk a retriever might surface by accident.

**Decision order** — fully answers → `positive`; topically related → `hard_negative`;
else → `irrelevant`.

**Hard rules in the prompt** (highlights):
- Verdicts must be GROUNDED in CANDIDATE's actual text. The prompt requires
  quoting the specific phrase being judged so contradictions ("doesn't specify
  X" when CANDIDATE literally states X) are caught at write time.
- Multi-part queries (dose AND timing, sets AND reps AND rest) require every
  sub-part addressed for `positive`; partial coverage is `hard_negative`.
- Definition-of-X chunks for a topic in QUERY are usually `hard_negative` when
  QUERY asks for a prescription, recommendation, or comparison.
- Default on malformed verdict = `irrelevant` — conservative, keeps noise out
  of the MNRL training set rather than letting it leak in as a hard negative.

**Outputs** (in `./triplets/`):
- `<file>.jsonl` — per-query records:
  ```json
  {"query": "...", "style": "formal", "source_chunk_id": "...",
   "positives": [...], "hard_negatives": [...], "irrelevants": [...]}
  ```
- `<file>.triplets.jsonl` — exploded `(query, positive, hard_negative)` rows
  ready for `MultipleNegativesRankingLoss`. Source-positive content is hydrated
  from `chunks/<file>.jsonl` via `load_chunk_contents()` so source positives
  contribute triplets directly (no downstream re-hydration step needed).
- `<file>.judge_debug.jsonl` — one row per judge call: `query`, `style`,
  `source_chunk_id`, `source_content` (recorded for the human reviewer doing
  prompt iteration; the judge itself never sees it), `candidate_chunk_id`,
  `candidate_content`, `label`, `reason`, and `raw_verdict`. Designed for prompt
  iteration — grep this file to find verdicts you disagree with.

Checkpoint: `checkpoints/<stem>_triplets.json` keyed by `f"{chunk_id}::{style}"`;
`--resume` skips processed queries.

**Testing the judge** — `tests/test_negative_judge.py` runs the live judge
against `tests/negative_judge_cases.py` — 15 hand-labeled cases across five
categories: `POSITIVE`, `HARD_NEGATIVE_PARTIAL`, `HARD_NEGATIVE_TOPICAL`,
`HARD_NEGATIVE_DEFINITION`, `IRRELEVANT_OFFTOPIC`. Each case is intentionally
distinct from the prompt few-shots so the test measures rule-application, not
memorisation. Use it as a calibration baseline before any large run; aim for
≥85% overall AND POSITIVE = 100% (false hard_negatives poison the MNRL training
set, so the POSITIVE bucket must be clean).
```bash
# Make sure the Step-4 judge env is exported (see §1: "Serving the Step-4 Judge").
python -m tests.test_negative_judge --save
```


---

## 3. Fine-Tuning the Retriever

### Training Configuration
* **Dataset:** Triplets of `(Query, Positive, Hard Negative)`.
* **Loss Function:** **MultipleNegativesRankingLoss (MNRL)**
* **Batch Size:** 32–64 

