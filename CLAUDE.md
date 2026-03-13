# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval Augmented Generation) research and evaluation project** focused on AI-powered personal training systems. It contains multiple sub-projects exploring RAG implementation, chunking strategies, evaluation, and knowledge probing.

## Environment Setup

```bash
# Python virtual environment (located at .venv/)
source .venv/bin/activate

# Required environment variables
export OPENAI_API_KEY="sk-..."
export LANGSMITH_API_KEY="..."  # Only needed for rag_eval_langchain/
```

## Commands

### simple-rag (Full-Stack Personal Trainer)
```bash
# Backend (Flask)
python simple-rag/app.py

# Frontend (React/Vite)
cd simple-rag/trainer-chat
npm install
npm run dev
npm run build
npm run lint
```

### rag_eval (Ragas-based evaluation)
```bash
cd rag_eval
python evals.py   # Run evaluations
python rag.py     # Test RAG pipeline directly
```

### rag_eval_langchain (Advanced chunking + evaluation)
```bash
cd rag_eval_langchain
python eval.py          # Full evaluation run
python eval_mock.py     # Mock run (no API calls)
python create_dataset.py  # Build test dataset
```

### synthetic_data
```bash
cd synthetic_data
python knowledge_graph.py   # Build KnowledgeGraph from PDFs
python generate_questions.py  # Generate test questions
```

## Architecture

### simple-rag/
Full-stack personal trainer app. The Flask backend (`app.py`) exposes `POST /api/chat` and `GET /api/health`. On each chat request:
1. `extract_user_intent.py` — extracts a `WorkoutIntent` (target muscles, equipment, difficulty)
2. `exercise_filter.py` — queries `exercise_db.xlsx` based on intent
3. `rag_on_rules.py` / `rag_on_store.py` — retrieves relevant chunks from PDF rules
4. A LangChain agent with tools calls OpenAI to generate the response

The React frontend (`trainer-chat/`) is a Vite app that talks to the Flask API.

### rag_eval_langchain/
The core evaluation harness for comparing chunking strategies:
- **`chunking/chunking.py`** — Three strategies: `BaseChunker` (1500 char recursive split), `SemanticChunker` (embedding-based, 95th percentile breakpoint), `ParentChildChunker` (small indexed chunks, large parent chunks returned)
- **`evaluators.py`** — Four LangSmith evaluators: `correctness`, `relevance`, `groundedness`, `retrieval_relevance`
- **`knowledge_probing/`** — Tests whether the LLM uses retrieved documents vs. parametric knowledge. Applies lexical substitution at three levels: SOFT (muscle names), MEDIUM (exercises/equipment), HARD (full obfuscation). Results saved to `leakage_results_*.json`.
- **`eval.py`** — Runs all three chunkers against the test dataset and logs results to LangSmith

### rag_eval/
Simpler evaluation using the Ragas framework directly. Uses a keyword-based retriever and OpenAI LLM. Results saved to CSV in `evaluation-results/`.

### synthetic_data/
Uses Ragas `TestsetGenerator` with a `KnowledgeGraph` built from the fitness PDFs in `resources/`. Generates single-hop and multi-hop questions with fitness enthusiast personas.

## Key Data Files
- `resources/` — Source PDFs (anatomy, NSCA, ACSM, periodization, hypertrophy literature)
- `simple-rag/exercise_db.xlsx` — Exercise database with muscle/equipment/difficulty columns
- `test_dataset/` — Pre-built evaluation datasets
- `evaluation-results/` — Output CSVs from evaluation runs
