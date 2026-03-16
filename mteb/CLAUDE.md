# Fitness RAG MTEB Benchmark Project

## Project Overview

Build a custom MTEB retrieval benchmark for a fitness RAG pipeline that includes:
- A vocabulary file for query transformations (lay terms → canonical terms, abbreviations → full forms)
- 100 orthogonal queries generated from the vocabulary
- Hypothetical document chunks (HyDE-style) generated per query
- MTEB-compatible dataset (corpus, queries, qrels)
- A custom MTEB embedder that applies query transformations before retrieval
- Full MTEB evaluation run against the custom dataset

---

## Project Structure

```
fitness-rag-benchmark/
│
├── CLAUDE.md                    ← this file
├── requirements.txt
│
├── vocab/
│   └── fitness_vocab.json       ← transformation vocabulary (generated first)
│
├── data/
│   ├── queries.json             ← 100 orthogonal queries
│   ├── corpus.json              ← generated chunks (one per query + distractors)
│   └── qrels.json               ← relevance judgments
│
├── dataset/                     ← MTEB-compatible HuggingFace dataset format
│   ├── corpus/
│   │   └── test/
│   │       └── data.parquet
│   ├── queries/
│   │   └── test/
│   │       └── data.parquet
│   └── qrels/
│       └── test/
│           └── data.parquet
│
├── src/
│   ├── build_vocab.py           ← Step 1: generate vocabulary
│   ├── build_dataset.py         ← Step 2: generate queries + chunks + qrels
│   ├── build_mteb_dataset.py    ← Step 3: convert to MTEB format
│   ├── custom_embedder.py       ← Step 4: custom embedder with transformations
│   └── run_benchmark.py         ← Step 5: run MTEB evaluation
│
└── results/                     ← MTEB outputs written here
```

---

## Step 1 — Build Vocabulary (`src/build_vocab.py`)

Generate `vocab/fitness_vocab.json` with ~50 entries per transformation type.

### Vocabulary Schema

```json
{
  "lay_to_canonical": {
    "cardio": "cardiovascular exercise",
    "abs": "abdominal muscles",
    "quads": "quadriceps",
    "hammies": "hamstrings",
    "lats": "latissimus dorsi",
    "traps": "trapezius",
    "delts": "deltoids",
    "pecs": "pectoralis major",
    "glutes": "gluteus maximus",
    "calves": "gastrocnemius and soleus",
    "...": "... (target ~50 entries)"
  },
  "abbreviations": {
    "HIIT": "High Intensity Interval Training",
    "RPE": "Rate of Perceived Exertion",
    "1RM": "One Repetition Maximum",
    "DOMS": "Delayed Onset Muscle Soreness",
    "ROM": "Range of Motion",
    "CNS": "Central Nervous System",
    "BMI": "Body Mass Index",
    "HR": "Heart Rate",
    "MHR": "Maximum Heart Rate",
    "ATP": "Adenosine Triphosphate",
    "...": "... (target ~50 entries)"
  }
}
```

### Implementation Notes for `build_vocab.py`

- Use the OpenAI API to generate vocabulary entries
- Prompt the model to generate fitness-specific lay terms and their canonical equivalents
- Prompt separately for abbreviations common in fitness/exercise science
- Deduplicate and validate entries
- Save to `vocab/fitness_vocab.json`
- Print summary: how many entries per category

### Example API Call Pattern

```python
from openai import OpenAI
import json

client = OpenAI()

def generate_lay_terms():
    response = client.chat.completions.create(
        model="gpt-5-mini",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": """Generate a JSON object with 50 fitness lay term to canonical term mappings.

            Rules:
            - Lay terms: informal words people use in gyms, social media, casual conversation
            - Canonical terms: anatomically/scientifically correct terminology
            - Cover: muscle groups, exercises, training concepts, nutrition terms
            - Return ONLY valid JSON, no markdown, no explanation

            Format: {"lay_term": "canonical_term", ...}"""
        }]
    )
    return json.loads(response.choices[0].message.content)

def generate_abbreviations():
    response = client.chat.completions.create(
        model="gpt-5-mini",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": """Generate a JSON object with 50 fitness abbreviation to full form mappings.

            Rules:
            - Abbreviations: common acronyms used in fitness, exercise science, gym culture
            - Full forms: complete expanded versions
            - Cover: training methods, physiological terms, measurement units, program types
            - Return ONLY valid JSON, no markdown, no explanation

            Format: {"ABBREVIATION": "Full Form", ...}"""
        }]
    )
    return json.loads(response.choices[0].message.content)
```

---

## Step 2 — Build Dataset (`src/build_dataset.py`)

Generate 100 orthogonal queries, one HyDE chunk per query, and qrels.

### Orthogonality Requirement

Queries must be orthogonal: for any given query, no other query in the set should be answered by the same chunk. This means:
- Each query targets a DISTINCT fitness concept
- Queries should not overlap in topic (e.g. there SHOULD NOT be two queries about lat-pulldown technique)
- Use at least one vocabulary term in each query

### Query Generation Strategy

Generate queries in BATCHES to enforce orthogonality:

```python
def generate_orthogonal_queries(vocab: dict) -> list[dict]:
    """
    Generate 100 queries that:
    1. Are topically distinct from each other
    2. Contain at least one lay term OR abbreviation from vocab
       (so the transformation pipeline has something to do)
    3. Are realistic user queries for a fitness RAG system
    """
    
    # Step 1: define 100 distinct fitness topic slots first
    topics_prompt = """List 100 distinct fitness topics that a RAG system might be queried about.
    
    Topics must be:
    - Non-overlapping (each topic is clearly distinct)
    - Cover: exercise form, programming, nutrition, recovery, physiology, equipment, injury prevention
    - Specific enough that a single document chunk can fully answer a query about it
    
    Return ONLY a JSON array of topic strings."""
    
    # Step 2: for each topic, generate a query using vocab terms
    query_prompt = """Given this fitness topic: {topic}
    And these vocabulary terms available for transformation:
    Lay terms: {lay_terms_sample}
    Abbreviations: {abbrev_sample}
    
    Write a realistic user query about this topic that:
    1. Uses at least one lay term OR abbreviation from the lists above
    2. Is phrased as a real person would ask it (natural language)
    3. Can be fully answered by a single focused document chunk
    
    Return ONLY the query string, nothing else."""
```

### Query Output Schema

```json
[
  {
    "id": "q_0001",
    "text": "what's the best way to train my hammies without a leg curl machine?",
    "topic": "hamstring training without equipment",
    "transformations_applicable": {
      "lay_to_canonical": {"hammies": "hamstrings"},
      "abbreviations": {}
    },
    "transformed_text": "what's the best way to train my hamstrings without a leg curl machine?"
  }
]
```

### Chunk Generation (HyDE Style)

For each query, generate a chunk that FULLY answers it:

```python
def generate_chunk_for_query(query: dict) -> dict:
    """
    Generate a hypothetical document chunk that fully answers the query.
    This is the HyDE approach: generate the ideal answer document.
    """
    prompt = f"""Write a focused fitness guide chunk that fully answers this query:
    
    Query: {query['transformed_text']}
    
    Requirements:
    - 150-300 words (chunk-sized, not a full article)
    - Use canonical/scientific terminology (not lay terms)
    - Don't use abbreviations, instead use the full forms
    - Be specific and actionable
    - Written as if from a professional fitness guide
    - Do NOT include the query text itself
    
    Return ONLY the chunk text."""
    
    # ... API call
    
    return {
        "id": f"doc_{query['id'].replace('q_', '')}",
        "title": query['topic'],
        "text": chunk_text
    }
```

### Qrels Generation

```python
def build_qrels(queries: list, corpus: list) -> list:
    """
    Score: 2 for the chunk generated from the query
           0 for all others (not stored, MTEB assumes 0)
    """
    qrels = []
    for query in queries:
        # The chunk generated for this query gets score 2
        corresponding_doc_id = f"doc_{query['id'].replace('q_', '')}"
        qrels.append({
            "query-id": query["id"],
            "corpus-id": corresponding_doc_id,
            "score": 2
        })
    return qrels
```

### Important: Save Intermediate Files

After generation, save:
- `data/queries.json` — list of query dicts
- `data/corpus.json` — list of chunk dicts
- `data/qrels.json` — list of qrel dicts

---

## Step 3 — Convert to MTEB Format (`src/build_mteb_dataset.py`)

Convert JSON files to HuggingFace Dataset format that MTEB expects.

```python
from datasets import Dataset, DatasetDict
import pandas as pd
import json

def build_mteb_dataset():
    # Load generated data
    queries_raw = json.load(open("data/queries.json"))
    corpus_raw = json.load(open("data/corpus.json"))
    qrels_raw = json.load(open("data/qrels.json"))

    # Corpus: requires id, text, title
    corpus_df = pd.DataFrame([
        {"id": doc["id"], "text": doc["text"], "title": doc["title"]}
        for doc in corpus_raw
    ])

    # Queries: requires id, text (use ORIGINAL text, not transformed)
    # Transformation happens at encode time in the custom embedder
    queries_df = pd.DataFrame([
        {"id": q["id"], "text": q["text"]}
        for q in queries_raw
    ])

    # Qrels: requires query-id, corpus-id, score
    qrels_df = pd.DataFrame(qrels_raw)

    # Build HuggingFace DatasetDict
    corpus_dataset = DatasetDict({"test": Dataset.from_pandas(corpus_df)})
    queries_dataset = DatasetDict({"test": Dataset.from_pandas(queries_df)})
    qrels_dataset = DatasetDict({"test": Dataset.from_pandas(qrels_df)})

    # Save locally
    corpus_dataset.save_to_disk("dataset/corpus")
    queries_dataset.save_to_disk("dataset/queries")
    qrels_dataset.save_to_disk("dataset/qrels")

    print(f"Dataset built:")
    print(f"  Corpus: {len(corpus_df)} chunks")
    print(f"  Queries: {len(queries_df)} queries")
    print(f"  Qrels: {len(qrels_df)} relevance pairs")
```

---

## Step 4 — Custom Embedder (`src/custom_embedder.py`)

Wraps the OpenAI `text-embedding-3-large` model and applies vocabulary transformations to queries before encoding.

```python
import json
import re
from pathlib import Path
import numpy as np
from openai import OpenAI

DEFAULT_MODEL = "text-embedding-3-large"
_BATCH_SIZE = 100  # OpenAI embeddings API limit per request

def _embed(client, texts, model):
    all_embeddings = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend(e.embedding for e in response.data)
    return np.array(all_embeddings, dtype=np.float32)

class FitnessRAGEmbedder:
    """
    Custom MTEB-compatible embedder that:
    1. Applies lay term → canonical term transformations to queries
    2. Applies abbreviation expansion to queries
    3. Encodes using OpenAI text-embedding-3-large

    Documents are encoded WITHOUT transformations (they use canonical terms already).
    Transformations only apply to queries (encode_queries method).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 vocab_path: str = "vocab/fitness_vocab.json"):
        self.model_name = model_name
        self.client = OpenAI()
        self.vocab = json.loads(Path(vocab_path).read_text())
        self.lay_to_canonical = self.vocab["lay_to_canonical"]
        self.abbreviations = self.vocab["abbreviations"]
        self._build_patterns()

    def _build_patterns(self):
        """Pre-compile regex patterns for all vocabulary terms."""
        # Abbreviations: match whole word, case-sensitive
        self.abbrev_pattern = {
            abbr: re.compile(rf'\b{re.escape(abbr)}\b')
            for abbr in self.abbreviations
        }
        # Lay terms: match whole word, case-insensitive
        self.lay_pattern = {
            term: re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            for term in self.lay_to_canonical
        }

    def transform_query(self, query: str) -> str:
        """Apply vocabulary transformations to a single query."""
        # 1. Expand abbreviations first (before lay term matching)
        for abbr, full_form in self.abbreviations.items():
            query = self.abbrev_pattern[abbr].sub(full_form, query)

        # 2. Convert lay terms to canonical terms
        for lay_term, canonical in self.lay_to_canonical.items():
            query = self.lay_pattern[lay_term].sub(canonical, query)

        return query

    def encode_queries(self, queries: list[str], batch_size: int = _BATCH_SIZE,
                       **kwargs) -> np.ndarray:
        transformed = [self.transform_query(q) for q in queries]
        return _embed(self.client, transformed, self.model_name)

    def encode_corpus(self, corpus: list[dict], batch_size: int = _BATCH_SIZE,
                      **kwargs) -> np.ndarray:
        texts = [
            (doc.get("title", "") + " " + doc.get("text", "")).strip()
            for doc in corpus
        ]
        return _embed(self.client, texts, self.model_name)

    def encode(self, sentences: list[str], batch_size: int = _BATCH_SIZE,
               **kwargs) -> np.ndarray:
        return _embed(self.client, sentences, self.model_name)
```

### Baseline Embedder (No Transformations)

Also implement a baseline for comparison:

```python
class BaselineEmbedder:
    """Same model, no query transformations. Used as baseline."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.client = OpenAI()

    def encode_queries(self, queries, batch_size=_BATCH_SIZE, **kwargs):
        return _embed(self.client, queries, self.model_name)

    def encode_corpus(self, corpus, batch_size=_BATCH_SIZE, **kwargs):
        texts = [(doc.get("title", "") + " " + doc.get("text", "")).strip()
                 for doc in corpus]
        return _embed(self.client, texts, self.model_name)

    def encode(self, sentences, batch_size=_BATCH_SIZE, **kwargs):
        return _embed(self.client, sentences, self.model_name)
```

---

## Step 5 — Run Benchmark (`src/run_benchmark.py`)

Register the custom dataset as an MTEB task and run evaluation.

```python
import mteb
from datasets import load_from_disk, DatasetDict
from custom_embedder import FitnessRAGEmbedder, BaselineEmbedder
import json

class FitnessRetrievalTask(mteb.AbsTaskRetrieval):
    """Custom MTEB retrieval task using our fitness dataset."""

    metadata = mteb.TaskMetadata(
        name="FitnessRAGRetrieval",
        dataset={
            # Use local path since we're not pushing to HuggingFace Hub
            "path": "dataset",           # local dataset directory
            "revision": "local",
        },
        description="Custom fitness RAG retrieval benchmark with query transformations",
        reference="",
        type="Retrieval",
        category="s2p",                  # sentence to paragraph
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )

    def load_data(self, **kwargs):
        """Override to load from local disk instead of HuggingFace Hub."""
        self.corpus = load_from_disk("dataset/corpus")
        self.queries = load_from_disk("dataset/queries")
        self.relevant_docs = self._load_qrels()

    def _load_qrels(self):
        """Convert qrels to MTEB expected format: dict[query_id][doc_id] = score"""
        import pandas as pd
        qrels_dataset = load_from_disk("dataset/qrels")
        qrels_df = qrels_dataset["test"].to_pandas()
        relevant_docs = {}
        for _, row in qrels_df.iterrows():
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])
            if qid not in relevant_docs:
                relevant_docs[qid] = {}
            relevant_docs[qid][did] = score
        return {"test": relevant_docs}


def run_evaluation():
    task = FitnessRetrievalTask()

    # Run with custom embedder (with transformations)
    print("\n=== Running: Custom Embedder (with transformations) ===")
    custom_embedder = FitnessRAGEmbedder(
        model_name="text-embedding-3-large",
        vocab_path="vocab/fitness_vocab.json"
    )
    evaluation = mteb.MTEB(tasks=[task])
    custom_results = evaluation.run(
        custom_embedder,
        output_folder="results/custom_embedder"
    )

    # Run with baseline embedder (no transformations)
    print("\n=== Running: Baseline Embedder (no transformations) ===")
    baseline_embedder = BaselineEmbedder(
        model_name="text-embedding-3-large"
    )
    baseline_results = evaluation.run(
        baseline_embedder,
        output_folder="results/baseline_embedder"
    )

    # Print comparison
    print("\n=== Results Comparison ===")
    print(f"{'Metric':<20} {'Custom':>10} {'Baseline':>10} {'Delta':>10}")
    print("-" * 52)
    metrics = ["ndcg_at_10", "ndcg_at_5", "recall_at_10", "precision_at_10"]
    for metric in metrics:
        custom_score = extract_score(custom_results, metric)
        baseline_score = extract_score(baseline_results, metric)
        delta = custom_score - baseline_score
        print(f"{metric:<20} {custom_score:>10.4f} {baseline_score:>10.4f} {delta:>+10.4f}")

def extract_score(results, metric):
    """Helper to extract a specific metric from MTEB results."""
    try:
        return results[0].scores["test"][0][metric]
    except (KeyError, IndexError):
        return 0.0

if __name__ == "__main__":
    run_evaluation()
```

---

## Requirements (`requirements.txt`)

```
openai>=1.0.0
mteb>=1.0.0
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## Execution Order

Run steps in this order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build vocabulary
python src/build_vocab.py
# Output: vocab/fitness_vocab.json
# Verify: ~50 lay terms, ~50 abbreviations

# 3. Build dataset (queries + chunks + qrels)
python src/build_dataset.py
# Output: data/queries.json, data/corpus.json, data/qrels.json
# Verify: 100 queries, 100 chunks, 100 qrel pairs
# Check: no topic overlap between queries

# 4. Convert to MTEB format
python src/build_mteb_dataset.py
# Output: dataset/ directory with corpus/queries/qrels subdirs
# Verify: parquet files created correctly

# 5. Run benchmark
python src/run_benchmark.py
# Output: results/ directory with JSON score files
# Prints: comparison table of custom vs baseline embedder
```

---

## Key Design Decisions

### Why Queries Use Original (Untransformed) Text in MTEB Dataset
The queries stored in `data/queries.json` and the MTEB dataset use the RAW query text (with lay terms and abbreviations). The transformation happens inside `FitnessRAGEmbedder.encode_queries()` at evaluation time. This correctly simulates real usage where a user types informal text and the system transforms it before retrieval.

### Why Chunks Use Canonical Terms
Chunks are generated from the TRANSFORMED query (canonical terms). This creates the asymmetry we want to measure: raw query → transformation → matches canonical chunk. If transformations work, nDCG@10 should be higher for the custom embedder than the baseline.

### Why Score 2 for Generated Chunk, Not Score 1
Score 2 (highly relevant) rather than 1 (partially relevant) because the chunk was generated specifically to fully answer the query — it is the ideal document. This maximizes the signal in nDCG scoring.

### Orthogonality Enforcement
During query generation, Claude is given the list of already-generated topics and instructed not to repeat or overlap. Generate in batches of 10, passing all previous topics as context.

### Vocabulary Coverage in Queries
Each query must use at least one vocabulary term (lay term or abbreviation). This is enforced in the generation prompt. Without this, the transformation pipeline has nothing to do and the benchmark doesn't test what we want.

---

## Expected Results Interpretation

```
If transformations help:
  custom_embedder nDCG@10 > baseline nDCG@10
  → query normalization improves retrieval

If transformations are neutral:
  scores roughly equal
  → base model already handles informal terms

If transformations hurt:
  custom_embedder nDCG@10 < baseline nDCG@10
  → over-expansion changing query meaning
  → review vocabulary mappings
```

---

## Notes for Claude Code

- All API calls use `openai` library, not `anthropic`
- Use `gpt-5-mini` for all generation tasks
- **Always use OpenAI structured outputs (`response_format={"type": "json_schema", "json_schema": {...}}`) on every OpenAI call that expects JSON** — this provides schema-validated JSON and is strictly preferred over the older `json_object` mode. Use `strict: False` when the schema includes `additionalProperties` (e.g. dynamic-key dicts). Always include a `name` and `description` in the `json_schema` dict.
- Add a system message alongside structured output calls (e.g. `"Always respond with valid JSON only"`) — the API requires the word "json" to appear somewhere in the messages when using JSON output modes
- Add retry logic around API calls — generation of 100 queries will make many calls
- The `load_data` override in `FitnessRetrievalTask` may need adjustment depending on MTEB version — check `mteb.AbsTaskRetrieval` interface if errors occur
- MTEB expects `encode_corpus` to receive a list of dicts with `id`, `text`, `title` keys
- If pushing dataset to HuggingFace Hub, update the `dataset.path` in `TaskMetadata` accordingly

## Caution for Claude Code

- **Model names matter** — when the user specifies a model (e.g. `gpt-5-mini`), use that exact string as the API model ID. Do not silently substitute a different model (e.g. `gpt-4o-mini`). If unsure whether a model ID is valid, ask before proceeding.
- **Verify string replacements** — when doing find-and-replace, double-check that old and new strings are genuinely different before running the command.
