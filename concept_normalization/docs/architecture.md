# Architecture: Concept Normalization Module

## Pipeline Position

```
User Query
    ↓
[Vocabulary Extraction — ONE-TIME PREPROCESSING]
    ├── Extract text from 9 PDFs (pdfplumber)
    ├── LLM extracts structured vocabulary by category
    └── Save as vocabulary.json
    ↓ (vocabulary.json loaded at runtime)
[Concept Normalization Module]
    ├── Step 1: Abbreviation Expansion   (uses vocabulary["abbreviations"])
    ├── Step 2: Coordination Resolution  (uses full vocabulary JSON)
    └── Step 3: Term Variation Handling  (uses full vocabulary JSON)
    ↓
Normalized Query (list of sub-queries)
    ↓
RAG Retriever (vector similarity search)
```

## Execution Order Justification

The three submodules MUST execute in this order:

| Step | Submodule | Why This Order |
|------|-----------|----------------|
| 1 | Abbreviation Expansion | Must run first — downstream submodules need full terms to reason about coordination and variation. Expanding "RT" to "resistance training" before coordination resolution prevents misinterpretation. |
| 2 | Coordination Resolution | Runs second — decomposes compound queries into distinct concept units. "strength and hypertrophy training" becomes two separate concepts. This must happen before term variation so each concept normalizes independently. |
| 3 | Term Variation Handling | Runs last — operates on fully expanded, decomposed individual concepts. Maps lay terms to canonical KB forms. Running on clean single concepts produces the most reliable normalization. |

## Knowledge Base Grounding

Every normalization decision is grounded in `vocabulary.json`:

- **Vocabulary injection**: The full JSON is injected into each submodule's system prompt. Each submodule gets the entire JSON for cross-referencing.
- **Conservative normalization**: If a term is not in the vocabulary (directly or as a recognized variation), the submodule MUST preserve the original unchanged.
- **No hallucinated mappings**: Submodules never infer or guess. Every mapping traces to a vocabulary entry.
- **Confidence flagging**: Uncertain mappings get a confidence score in the log. Both original and candidate are preserved.

## Submodule Data Flow (Worked Example)

```
User Query: "What are the benefits of RT and HIIT for VO2max?"

Step 1 — Abbreviation Expansion:
  Input:  "What are the benefits of RT and HIIT for VO2max?"
  Output: "What are the benefits of resistance training and
           high-intensity interval training for VO2max?"
  Log:    RT → resistance training (vocabulary.abbreviations)
          HIIT → high-intensity interval training (vocabulary.abbreviations)
          VO2max → preserved (recognized scientific notation)

Step 2 — Coordination Resolution:
  Input:  "What are the benefits of resistance training and
           high-intensity interval training for VO2max?"
  Output: ["benefits of resistance training for VO2max",
           "benefits of high-intensity interval training for VO2max"]
  Log:    coordination "X and Y" resolved into 2 sub-queries
          shared context "benefits ... for VO2max" distributed to both

Step 3 — Term Variation Handling (applied per sub-query):
  Input:  "benefits of resistance training for VO2max"
  Output: "benefits of resistance training for maximal oxygen uptake"
  Log:    VO2max → maximal oxygen uptake (canonical form, vocabulary.physiological_concepts)
```

Final output: list of normalized sub-queries + full transformation log.

## Transformation Log Schema

Every pipeline call produces this structure:

```json
{
  "original_query": "string",
  "abbreviation_step": {
    "expanded_query": "string",
    "expansions": [
      {"original": "str", "expanded": "str", "source": "str"}
    ]
  },
  "coordination_step": {
    "sub_queries": ["string"],
    "coordination_type": "string",
    "decomposed": true
  },
  "term_variation_step": [
    {
      "normalized_query": "string",
      "normalizations": [
        {"original": "str", "normalized": "str", "category": "str", "confidence": 0.95}
      ]
    }
  ],
  "final_normalized_queries": ["string"],
  "total_latency_ms": 0.0
}
```

## Vocabulary JSON Structure

The vocabulary file (`vocabulary.json`) has this structure:

```json
{
  "exercise_types": ["resistance training", "plyometrics", "isometric exercise", "..."],
  "physiological_concepts": ["muscle hypertrophy", "VO2max", "lactate threshold", "..."],
  "training_variables": ["repetition maximum", "training volume", "progressive overload", "..."],
  "anatomy": ["quadriceps", "glenohumeral joint", "rotator cuff", "..."],
  "conditions_populations": ["type 2 diabetes", "sarcopenia", "postmenopausal women", "..."],
  "equipment": ["barbell", "resistance band", "isokinetic dynamometer", "..."],
  "abbreviations": {"RT": "resistance training", "HIIT": "high-intensity interval training", "...": "..."},
  "lay_to_technical": {"toning": "muscular endurance training", "bulking": "hypertrophy-focused training", "...": "..."}
}
```

## Prompt Placeholder Conventions

System prompts in `prompts/` use these placeholders, replaced at runtime:

- `{vocabulary_json["abbreviations"]}` — replaced with `json.dumps(vocabulary["abbreviations"])`
- `{full_vocabulary_json}` — replaced with `json.dumps(vocabulary)`
- `{text_chunk}` — (vocabulary extraction only) replaced with PDF text chunk
