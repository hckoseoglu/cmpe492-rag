import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

client = OpenAI()

VOCAB_PATH = Path(__file__).parent.parent / "vocab" / "fitness_vocab.json"
DATA_DIR = Path(__file__).parent.parent / "data"

# ── Schemas ────────────────────────────────────────────────────────────────────

TOPICS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fitness_topics",
        "description": "A batch of distinct fitness topics for RAG queries",
        "schema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["topics"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

QUERY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fitness_query",
        "description": "A realistic user query for a fitness RAG system using a vocabulary term",
        "schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "vocab_term_used": {"type": "string"},
                "vocab_term_type": {
                    "type": "string",
                    "enum": ["lay_term", "abbreviation"],
                },
                "transformed_text": {"type": "string"},
            },
            "required": [
                "text",
                "vocab_term_used",
                "vocab_term_type",
                "transformed_text",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

CHUNK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fitness_chunk",
        "description": "A focused fitness guide chunk that fully answers a query",
        "schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

DISTRACTORS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "distractor_chunks",
        "description": "Hard-negative distractor chunks that are topically adjacent but do not answer the query",
        "schema": {
            "type": "object",
            "properties": {
                "distractors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["title", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["distractors"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ── Utilities ──────────────────────────────────────────────────────────────────


def call_with_retry(fn, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def load_vocab() -> dict:
    with open(VOCAB_PATH) as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} items to {path}")


# ── Step 1: Generate 100 orthogonal topics ─────────────────────────────────────


def generate_topic_batch(existing_topics: list[str], batch_size: int = 10) -> list[str]:
    existing_str = (
        "\n".join(f"- {t}" for t in existing_topics) if existing_topics else "None yet."
    )

    def _call():
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format=TOPICS_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fitness curriculum designer. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate exactly {batch_size} NEW fitness topics for a RAG system benchmark.\n\n"
                        "Topics already chosen (DO NOT overlap or repeat):\n"
                        f"{existing_str}\n\n"
                        "Requirements:\n"
                        "- Each topic must be clearly distinct from all others listed above\n"
                        "- Topics must span a wide range: exercise form/technique, training programming, "
                        "nutrition, recovery, physiology, equipment usage, injury prevention, "
                        "flexibility/mobility, cardio training, strength training\n"
                        "- Each topic must be specific enough that a single 150-300 word chunk can fully answer it\n"
                        "- Write topics as short descriptive phrases (5-10 words each)\n\n"
                        f'Return JSON with a "topics" array of exactly {batch_size} strings.'
                    ),
                },
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return data["topics"]

    return call_with_retry(_call)


def generate_topics(target: int = 100, batch_size: int = 10) -> list[str]:
    print(f"Generating {target} orthogonal topics in batches of {batch_size}...")
    all_topics: list[str] = []
    while len(all_topics) < target:
        remaining = target - len(all_topics)
        size = min(batch_size, remaining)
        batch = generate_topic_batch(all_topics, size)
        all_topics.extend(batch)
        print(f"  {len(all_topics)}/{target} topics collected")
    return all_topics[:target]


# ── Validation helpers ─────────────────────────────────────────────────────────

MAX_VALIDATION_RETRIES = 3


def _term_present(term: str, term_type: str, text: str) -> bool:
    """Check that vocab_term_used appears verbatim in the query text."""
    flags = re.IGNORECASE if term_type == "lay_term" else 0
    return bool(re.search(r"\b" + re.escape(term) + r"\b", text, flags))


def _chunk_violations(text: str, vocab: dict) -> list[str]:
    """Return any lay terms or abbreviations found verbatim in the chunk text."""
    violations = []
    for term in vocab["lay_to_canonical"]:
        if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
            violations.append(f"lay term '{term}'")
    for abbr in vocab["abbreviations"]:
        if re.search(r"\b" + re.escape(abbr) + r"\b", text):
            violations.append(f"abbreviation '{abbr}'")
    return violations


# ── Step 2: Generate a query for each topic ────────────────────────────────────


def generate_query_for_topic(topic: str, vocab: dict, idx: int) -> dict | None:
    lay_to_canonical = vocab["lay_to_canonical"]
    abbreviations = vocab["abbreviations"]

    lay_str = "\n".join(f'  "{k}" -> "{v}"' for k, v in lay_to_canonical.items())
    abbrev_str = "\n".join(f'  "{k}" -> "{v}"' for k, v in abbreviations.items())

    def _call():
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format=QUERY_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fitness query generator. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Fitness topic: {topic}\n\n"
                        "Available vocabulary terms — pick whichever fits most naturally:\n\n"
                        f"Lay terms (informal -> canonical):\n{lay_str}\n\n"
                        f"Abbreviations (abbrev -> full form):\n{abbrev_str}\n\n"
                        "Write a realistic query a gym-goer would type about this topic.\n\n"
                        "Rules:\n"
                        "- The chosen vocab term MUST appear verbatim (exact spelling) inside 'text'\n"
                        "- Pick the term that fits the topic most naturally — do NOT force an unrelated term\n"
                        "- Phrase the query naturally, as a real person would ask it\n"
                        "- The query must be answerable by a single focused fitness document chunk\n"
                        "- 'transformed_text' is identical to 'text' except the vocab term is replaced "
                        "with its canonical/expanded form\n\n"
                        "Example (topic: hamstring injury prevention):\n"
                        '  text: "how do I avoid hurting my hammies on deadlift day?"\n'
                        '  vocab_term_used: "hammies"\n'
                        '  vocab_term_type: "lay_term"\n'
                        '  transformed_text: "how do I avoid hurting my hamstrings on deadlift day?"\n\n'
                        "Example (topic: interval cardio fatigue):\n"
                        '  text: "why am I so wrecked after HIIT — is that normal?"\n'
                        '  vocab_term_used: "HIIT"\n'
                        '  vocab_term_type: "abbreviation"\n'
                        '  transformed_text: "why am I so wrecked after High Intensity Interval Training — is that normal?"'
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    for attempt in range(MAX_VALIDATION_RETRIES):
        result = call_with_retry(_call)
        term = result["vocab_term_used"]
        term_type = result["vocab_term_type"]

        # Validate term appears verbatim in query text
        if not _term_present(term, term_type, result["text"]):
            print(
                f"    Validation failed (attempt {attempt + 1}): "
                f"'{term}' not found verbatim in text — retrying..."
            )
            continue

        # Validate term exists in our vocab
        if term_type == "lay_term" and term not in lay_to_canonical:
            print(
                f"    Validation failed (attempt {attempt + 1}): "
                f"lay term '{term}' not in vocab — retrying..."
            )
            continue
        if term_type == "abbreviation" and term not in abbreviations:
            print(
                f"    Validation failed (attempt {attempt + 1}): "
                f"abbreviation '{term}' not in vocab — retrying..."
            )
            continue

        # Valid — build and return
        transformations: dict = {"lay_to_canonical": {}, "abbreviations": {}}
        if term_type == "lay_term":
            transformations["lay_to_canonical"][term] = lay_to_canonical[term]
        else:
            transformations["abbreviations"][term] = abbreviations[term]

        return {
            "id": f"q_{idx + 1:04d}",
            "text": result["text"],
            "topic": topic,
            "transformations_applicable": transformations,
            "transformed_text": result["transformed_text"],
        }

    print(
        f"    Skipping topic '{topic}' — could not generate a valid query after "
        f"{MAX_VALIDATION_RETRIES} attempts."
    )
    return None


# ── Step 3: Generate a HyDE chunk for each query ──────────────────────────────


def generate_chunk_for_query(query: dict, vocab: dict) -> dict | None:
    def _call():
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format=CHUNK_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional fitness guide author. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Write a focused fitness guide chunk that fully answers this query:\n\n"
                        f'Query: {query["transformed_text"]}\n\n'
                        "Requirements:\n"
                        "- 150-300 words\n"
                        "- Use canonical/scientific terminology (not lay terms)\n"
                        "- Write out all terms in full — do not use abbreviations\n"
                        "- Be specific and actionable\n"
                        "- Written as if from a professional fitness guide or textbook\n"
                        "- Do NOT include or repeat the query text itself\n\n"
                        'Return JSON with a "text" field containing only the chunk text.'
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    doc_id = f"doc_{query['id'].replace('q_', '')}"

    for attempt in range(MAX_VALIDATION_RETRIES):
        result = call_with_retry(_call)
        violations = _chunk_violations(result["text"], vocab)
        if violations:
            print(
                f"    Validation failed (attempt {attempt + 1}): "
                f"chunk contains {violations} — retrying..."
            )
            continue
        return {
            "id": doc_id,
            "title": query["topic"],
            "text": result["text"],
        }

    print(
        f"    Skipping query '{query['id']}' — could not generate a clean chunk after "
        f"{MAX_VALIDATION_RETRIES} attempts."
    )
    return None


# ── Step 3b: Generate hard-negative distractor chunks ────────────────────────


def generate_distractors_for_query(
    query: dict, gold_chunk: dict, num_distractors: int = 3
) -> list[dict]:
    """
    Generate hard-negative distractor chunks that are topically adjacent to the
    query but do NOT answer it.  These share vocabulary/domain overlap with the
    gold chunk so that only a well-matched embedding can distinguish them.
    """

    def _call():
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format=DISTRACTORS_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fitness content writer creating hard-negative "
                        "retrieval distractors. Always respond with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query: {query['transformed_text']}\n"
                        f"Topic / Title: {query['topic']}\n"
                        f"Gold answer chunk (first 200 chars): {gold_chunk['text'][:200]}...\n\n"
                        f"Generate exactly {num_distractors} distractor document chunks.\n\n"
                        "CRITICAL RULES:\n"
                        f'- Every distractor MUST use the EXACT same title: "{query["topic"]}"\n'
                        "- Each distractor must be about the SAME broad topic as the gold chunk\n"
                        "- Each distractor must NOT answer the original query — it must cover a "
                        "different aspect, angle, or sub-question within that same topic\n"
                        "- Each distractor must be 150-300 words, written as a professional fitness guide\n"
                        "- Use canonical/scientific terminology (no abbreviations, no slang)\n\n"
                        "Distractor strategy — stay within the topic but shift the focus:\n"
                        "- If query asks about squat FORM CUES, write about squat PROGRAMMING, "
                        "squat VARIATIONS, or squat MOBILITY PREREQUISITES\n"
                        "- If query asks about HIIT TREADMILL SETTINGS, write about HIIT "
                        "PHYSIOLOGICAL BENEFITS, HIIT RECOVERY PROTOCOLS, or HIIT CONTRAINDICATIONS\n"
                        "- If query asks about SLEEP DURATION for recovery, write about SLEEP HYGIENE "
                        "ENVIRONMENT, SLEEP SUPPLEMENTATION, or SLEEP TRACKING METRICS\n\n"
                        "The goal: these chunks look like they belong to the same guide chapter as "
                        "the gold chunk (same title, same domain vocabulary) but a user reading them "
                        "would NOT find the answer to the original query.\n\n"
                        
                        "EXAMPLES OF CORRECT DISTRACTOR BEHAVIOR:\n\n"
                        "Example 1:\n"
                        '  Query: "What knee angle should I maintain at the bottom of a back squat?"\n'
                        '  Topic: "Back Squat Technique"\n'
                        "  Gold chunk answers: knee flexion angle, joint positioning at depth\n"
                        '  GOOD distractor (shifts to programming): "Progressive overload in the back squat '
                        "follows a periodised structure. Novice athletes typically respond to linear "
                        "progression, adding 2.5–5 kg per session across three weekly exposures. "
                        "Intermediate trainees benefit from undulating periodisation, rotating between "
                        "hypertrophy (4×8–10 at 70% 1RM), strength (5×3–5 at 82–87% 1RM), and "
                        "power-focused sessions (6×2 at 85–90% 1RM). Deload weeks every fourth "
                        'microcycle reduce accumulated fatigue without sacrificing adaptation..."\n'
                        "  WHY it works: same title, same domain vocabulary, but zero information "
                        "about knee angle at the bottom of the squat.\n\n"
                        "Example 2:\n"
                        '  Query: "How many hours of sleep do athletes need for optimal muscle recovery?"\n'
                        '  Topic: "Sleep and Athletic Recovery"\n'
                        "  Gold chunk answers: recommended sleep duration (e.g. 8–10 hours for athletes)\n"
                        '  GOOD distractor (shifts to environment): "The sleep environment exerts a '
                        "measurable influence on polysomnographic sleep architecture. Ambient room "
                        "temperature between 16–19°C promotes the distal vasodilation necessary for "
                        "core body temperature reduction, a prerequisite for slow-wave sleep initiation. "
                        "Blackout curtains or sleep masks eliminate photonic stimulation of the "
                        "suprachiasmatic nucleus, preserving endogenous melatonin secretion. Acoustic "
                        "environments below 30 dB minimise cortical arousal events and reduce "
                        'fragmentation of restorative deep-sleep stages..."\n'
                        "  WHY it works: same title, same recovery vocabulary, but says nothing about "
                        "how many hours of sleep an athlete needs.\n\n"
                        
                        
                        f'Return JSON with a "distractors" array of {num_distractors} objects, '
                        f'each with "title" set to exactly "{query["topic"]}" and a "text" field.'
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    result = call_with_retry(_call)
    return result["distractors"]


def _generate_distractors_task(
    query: dict, gold: dict, num_distractors: int
) -> tuple[str, list[dict] | None]:
    """Worker function for parallel distractor generation. Returns (query_id, distractors)."""
    try:
        distractors = generate_distractors_for_query(query, gold, num_distractors)
        return query["id"], distractors
    except Exception as e:
        print(f"    Failed for {query['id']}: {e}")
        return query["id"], None


def generate_all_distractors(
    queries: list[dict],
    corpus: list[dict],
    num_distractors: int = 3,
    max_workers: int = 10,
) -> list[dict]:
    """Generate distractor chunks for every query in parallel."""
    gold_by_qid = {doc["id"].replace("doc_", "q_"): doc for doc in corpus}

    # Build work items, skipping queries without a gold chunk
    work = []
    for query in queries:
        gold = gold_by_qid.get(query["id"])
        if gold is None:
            print(f"    Skipping — no gold chunk found for {query['id']}")
            continue
        work.append((query, gold))

    # Submit all tasks to thread pool
    results: dict[str, list[dict]] = {}
    completed = 0
    total = len(work)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_distractors_task, query, gold, num_distractors): query["id"]
            for query, gold in work
        }
        for future in as_completed(futures):
            completed += 1
            qid, distractors = future.result()
            if distractors is not None:
                results[qid] = distractors
            print(f"  [{completed}/{total}] Done: {qid}")

    # Build output in deterministic order (sorted by query ID)
    distractor_docs = []
    for query in queries:
        distractors = results.get(query["id"])
        if distractors is None:
            continue
        q_num = query["id"].replace("q_", "")
        for j, d in enumerate(distractors):
            distractor_docs.append(
                {
                    "id": f"distractor_{q_num}_{j + 1}",
                    "title": d["title"],
                    "text": d["text"],
                }
            )

    return distractor_docs


# ── Step 4: Build qrels ────────────────────────────────────────────────────────


def build_qrels(queries: list) -> list:
    return [
        {
            "query-id": q["id"],
            "corpus-id": f"doc_{q['id'].replace('q_', '')}",
            "score": 2,
        }
        for q in queries
    ]


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab()

    # Step 1: Topics
    topics = generate_topics(target=100, batch_size=10)
    save_json(topics, DATA_DIR / "topics.json")

    # Step 2: Queries
    print("\nGenerating queries...")
    queries = []
    for i, topic in enumerate(topics):
        print(f"  [{i + 1}/100] {topic}")
        query = generate_query_for_topic(topic, vocab, i)
        if query is not None:
            queries.append(query)
    print(f"  -> {len(queries)} valid queries")
    save_json(queries, DATA_DIR / "queries.json")

    # Step 3: Chunks — drop query+chunk pair if chunk fails validation
    print("\nGenerating HyDE chunks...")
    valid_queries = []
    corpus = []
    for i, query in enumerate(queries):
        print(f"  [{i + 1}/{len(queries)}] {query['topic']}")
        chunk = generate_chunk_for_query(query, vocab)
        if chunk is not None:
            valid_queries.append(query)
            corpus.append(chunk)
    if len(valid_queries) < len(queries):
        print(
            f"  -> {len(queries) - len(valid_queries)} queries dropped due to invalid chunks"
        )
        queries = valid_queries
        save_json(queries, DATA_DIR / "queries.json")
    save_json(corpus, DATA_DIR / "corpus.json")

    # Step 3b: Generate hard-negative distractors
    print("\nGenerating hard-negative distractor chunks...")
    distractor_docs = generate_all_distractors(queries, corpus, num_distractors=3)
    corpus.extend(distractor_docs)
    save_json(corpus, DATA_DIR / "corpus.json")

    # Step 4: Qrels (only positive pairs — MTEB assumes score=0 for unlisted)
    qrels = build_qrels(queries)
    save_json(qrels, DATA_DIR / "qrels.json")

    print("\nDone!")
    print(f"  Queries     : {len(queries)}")
    print(
        f"  Corpus      : {len(corpus)} ({len(corpus) - len(distractor_docs)} gold + {len(distractor_docs)} distractors)"
    )
    print(f"  Qrels       : {len(qrels)}")


if __name__ == "__main__":
    main()
