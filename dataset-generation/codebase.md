# Codebase Overview

## What This Does

Synthetic dataset generation pipeline for fine-tuning a RAG-based AI personal trainer's retriever. The pipeline processes fitness/exercise science PDFs through 4 stages:

1. **Agentic Chunking** (implemented) -- LLM decomposes PDF text into atomic propositions, then groups them into thematic chunks with summaries. Output: JSONL files in `./chunks/`.
2. **Positive Pair Generation** (not yet implemented) -- Generate diverse user queries per chunk, validated by an LLM judge.
3. **Hard Negative Mining** (not yet implemented) -- Hybrid search (BM25 + vector) to find hard negatives per query.
4. **Expert Judge Validation** (not yet implemented) -- DeepSeek-R1 classifies candidates as new positives or hard negatives.

Final output: `(Query, Positive, Hard Negative)` triplets for fine-tuning `BAAI/bge-m3` with MultipleNegativesRankingLoss.

## Project Structure

```
dataset-generation/
├── chunker.py          # Main pipeline orchestrator + CLI
├── llm_client.py       # Ollama/vLLM abstraction (OpenAI-compatible API)
├── pdf_loader.py       # PDF text extraction + batching
├── propositions.py     # LLM proposition extraction
├── grouper.py          # Proposition grouping + summarization
├── config.py           # Configuration dataclass
├── requirements.txt    # pypdf, openai
├── setup_local.sh      # Ollama + gemma2:9b setup script
├── chunks/             # Output JSONL files (generated)
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
LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=google/gemma-2-9b-it python chunker.py
```

### Output Format
One JSONL file per PDF in `./chunks/`:
```json
{"id": "progression_models_0001", "content": "...", "summary": "..."}
```

## What's Next

### Step 2: Positive Pair Generation
- For each chunk in `./chunks/*.jsonl`, generate 2 diverse user queries using the LLM
- Implement LLM-as-judge loop: validate each query actually maps to the chunk, retry with feedback if not
- Output: `{"chunk_id": "...", "query": "...", "positive": "..."}`

### Step 3: Hard Negative Mining
- Build a hybrid search index (BM25 + dense vectors via `BAAI/bge-m3`) over all chunks
- For each generated query, retrieve top-5 chunks excluding the source
- These become hard negative candidates

### Step 4: Expert Judge Validation
- Use DeepSeek-R1-Distill-Llama-8B to classify each candidate as relevant (new positive) or irrelevant (hard negative)
- Output final triplets: `(query, positive_chunk, hard_negative_chunk)`

### Step 5: Retriever Fine-Tuning
- Fine-tune `BAAI/bge-m3` on the triplet dataset
- Loss: MultipleNegativesRankingLoss (MNRL)
- Batch size: 32-64, on NVIDIA L4 (24GB VRAM)
