
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
| **Judge** | `DeepSeek-R1-Distill-Llama-8B` | Superior reasoning for factual verification and error detection |
| **Embedder** | `BAAI/bge-m3` | Multi-vector support (Dense + Sparse) and 8k context window |

### Serving the Generator (vLLM)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-9b-it \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 8192 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Set env vars before running scripts:
```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="google/gemma-2-9b-it"
export LLM_API_KEY="dummy"
```

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

### Step 3: Global Hybrid Search & Hard Negative Mining
1.  **Search:** For each query, run **Hybrid Search** (BM25 + Vector)
2.  **Filter:** Exclude the original "Source Chunk".
3.  **Candidates:** Keep the top 5 ranking chunks as potential "Hard Negatives" or "Positives".

### Step 4: The Expert Judge (Validation)
1.  **Input:** `(Query, Chunk Candidate)` to DeepSeek-R1.
2.  **Classify Chunk:** 
 * If Candidate = Relevant -> Label as **New Positive**.
  * If Candidate = Irrelevant but highly ranked -> Label as **Hard Negative**


---

## 3. Fine-Tuning the Retriever

### Training Configuration
* **Dataset:** Triplets of `(Query, Positive, Hard Negative)`.
* **Loss Function:** **MultipleNegativesRankingLoss (MNRL)**
* **Batch Size:** 32–64 

