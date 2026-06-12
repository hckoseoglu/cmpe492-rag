
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

### Step 2: Positive Pair Generation
1.  **Prompt:** Input chunk content + summary to Llama-3.1.
2.  **Task:** "Generate 2 diverse user queries that are definitively answered by this text.".
3.  **Verify:** "Use LLM as judge to verify generated query is valid, otherwise feedback generator about the error until task completes sucessfuly".

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

