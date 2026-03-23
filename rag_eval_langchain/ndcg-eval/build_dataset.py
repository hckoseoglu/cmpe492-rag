"""
Dataset generation for nDCG retrieval evaluation.

Generates 100 orthogonal fitness queries, gold answer chunks, hard-negative
distractors (same topic, wrong angle), and partly-relevant documents (helpful
but incomplete answers).

Pipeline:
  1. Generate 100 distinct topics (batched, with overlap checking) — sequential
  2. For each topic IN PARALLEL:
     a. Generate one query
     b. Generate one gold HyDE chunk (score 2)
     c. Generate 2 hard-negative distractors (score 0)
     d. Generate 1 partly-relevant document (score 1)
  3. Assemble corpus and qrels
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

client = OpenAI()

DATA_DIR = Path(__file__).parent / "data"
MODEL = "gpt-4o-mini"
NUM_DISTRACTORS = 2
MAX_WORKERS = 10

# ── Schemas ───────────────────────────────────────────────────────────────────

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
        "description": "A realistic user query for a fitness RAG system",
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
        "description": "Hard-negative distractor chunks — topically adjacent but do not answer the query",
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

PARTIAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "partial_chunk",
        "description": "A partly-relevant document that helps but does not fully answer the query",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["title", "text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ── Utilities ─────────────────────────────────────────────────────────────────


def call_with_retry(fn, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} items to {path}")


# ── Step 1: Generate orthogonal topics ────────────────────────────────────────


def generate_topic_batch(existing_topics: list[str], batch_size: int = 10) -> list[str]:
    existing_str = (
        "\n".join(f"- {t}" for t in existing_topics) if existing_topics else "None yet."
    )

    def _call():
        response = client.chat.completions.create(
            model=MODEL,
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
                        "Examples of good topics:\n"
                        '- "Proper grip width for conventional deadlift"\n'
                        '- "Creatine monohydrate loading and maintenance dosing"\n'
                        '- "Hip hinge cueing for Romanian deadlift"\n'
                        '- "Central nervous system fatigue from maximal lifting"\n\n'
                        "Examples of BAD topics (too broad):\n"
                        '- "Strength training" (too vague)\n'
                        '- "Nutrition for athletes" (too broad)\n\n'
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


# ── Per-topic generation functions ────────────────────────────────────────────


def generate_query(topic: str, idx: int) -> dict | None:
    def _call():
        response = client.chat.completions.create(
            model=MODEL,
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
                        "Write a realistic query that a gym-goer or fitness enthusiast would type "
                        "into a fitness Q&A system about this topic.\n\n"
                        "Rules:\n"
                        "- Phrase the query naturally, as a real person would ask it\n"
                        "- The query should be specific enough to have a single clear answer\n"
                        "- The query must be answerable by a single focused 150-300 word document chunk\n"
                        "- Use natural language — mix of formal and informal is fine\n"
                        "- Do NOT make the query too short (at least 10 words) or too long (at most 40 words)\n\n"
                        "Examples of good queries:\n"
                        '- "What\'s the correct grip width for a conventional deadlift and does it change based on body proportions?"\n'
                        '- "How should I load creatine monohydrate and what\'s the right daily maintenance dose?"\n'
                        '- "What are the main cues for hinging at the hips during a Romanian deadlift to avoid lower back rounding?"\n\n'
                        "Examples of BAD queries:\n"
                        '- "deadlift grip?" (too short, too vague)\n'
                        '- "Tell me everything about creatine including history, chemistry, dosing, timing, side effects, and interactions" (too broad for one chunk)\n\n'
                        'Return JSON with a "text" field containing only the query string.'
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    result = call_with_retry(_call)
    if result is None:
        return None

    return {
        "id": f"q_{idx + 1:04d}",
        "text": result["text"],
        "topic": topic,
    }


def generate_gold_chunk(query: dict) -> dict | None:
    def _call():
        response = client.chat.completions.create(
            model=MODEL,
            response_format=CHUNK_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional fitness guide author. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        "Write a focused fitness guide chunk that FULLY and DIRECTLY answers "
                        "this query:\n\n"
                        f'Query: {query["text"]}\n\n'
                        "Requirements:\n"
                        "- 150-300 words\n"
                        "- Use proper scientific and anatomical terminology\n"
                        "- Be specific, actionable, and evidence-informed\n"
                        "- Written as if from a professional fitness textbook or guide\n"
                        "- The chunk must contain ALL the information needed to fully answer the query\n"
                        "- Do NOT repeat or include the query text itself in the chunk\n\n"
                        "Example — Query: 'What grip width should I use for conventional deadlift?'\n"
                        "Good chunk: 'For the conventional deadlift, position the hands on the barbell "
                        "at approximately shoulder width, with arms hanging vertically from the "
                        "glenohumeral joint when viewed from the front. This typically translates to "
                        "a grip placed just outside the knees when the lifter is in the start position "
                        "with hips hinged and shins near-vertical. Individuals with longer torsos "
                        "relative to arm length may benefit from a marginally wider grip to accommodate "
                        "the increased forward lean required to reach the bar. A narrower grip increases "
                        "the effective range of motion at lockout but may cause the hands to contact "
                        "the lateral thighs during the pull, disrupting bar path. Mixed grip "
                        "(one supinated, one pronated) or hook grip can be employed at heavier loads "
                        "without altering width. Verify grip symmetry by measuring from the barbell's "
                        "center knurl mark to each hand...'\n\n"
                        'Return JSON with a "text" field containing only the chunk text.'
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    result = call_with_retry(_call)
    if result is None:
        return None

    doc_id = f"doc_{query['id'].replace('q_', '')}"
    return {
        "id": doc_id,
        "title": query["topic"],
        "text": result["text"],
    }


def generate_distractors(query: dict, gold_chunk: dict, num_distractors: int = 2) -> list[dict]:
    def _call():
        response = client.chat.completions.create(
            model=MODEL,
            response_format=DISTRACTORS_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fitness content writer creating hard-negative "
                        "retrieval distractors for an information retrieval benchmark. "
                        "Always respond with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query: {query['text']}\n"
                        f"Topic: {query['topic']}\n"
                        f"Gold answer chunk (first 300 chars): {gold_chunk['text'][:300]}...\n\n"
                        f"Generate exactly {num_distractors} HARD-NEGATIVE distractor document chunks.\n\n"
                        "WHAT IS A HARD NEGATIVE?\n"
                        "A hard negative is a document that looks extremely relevant at first glance — "
                        "it shares the same domain vocabulary, the same topic area, and might even "
                        "discuss the same exercise or concept — but it does NOT answer the specific "
                        "question asked. It covers a DIFFERENT aspect or angle of the same topic.\n\n"
                        "CRITICAL RULES:\n"
                        f'- Every distractor MUST use the EXACT same title: "{query["topic"]}"\n'
                        "- Each distractor must be about the SAME broad topic as the gold chunk\n"
                        "- Each distractor must NOT answer the original query — it covers a "
                        "different aspect, angle, or different question within that same topic\n"
                        "- Each distractor must be 150-300 words\n"
                        "- Use proper scientific/anatomical terminology\n"
                        "- The distractor must share significant vocabulary overlap with the gold chunk "
                        "(same muscle names, same exercise names, same domain terms) to make it "
                        "hard for an embedding model to distinguish\n\n"
                        "STRATEGY — stay within the topic but shift the focus:\n"
                        "- If query asks about FORM/TECHNIQUE → write about PROGRAMMING or VARIATIONS\n"
                        "- If query asks about DOSING/AMOUNTS → write about MECHANISMS or TIMING\n"
                        "- If query asks about BENEFITS → write about RISKS or CONTRAINDICATIONS\n"
                        "- If query asks about ONE EXERCISE → write about a RELATED but DIFFERENT exercise\n\n"
                        "EXAMPLE 1:\n"
                        '  Query: "What knee angle should I hit at the bottom of a back squat?"\n'
                        '  Topic: "Back squat depth and knee mechanics"\n'
                        "  Gold chunk: discusses knee flexion angle, joint positioning at depth\n"
                        "  GOOD hard negative (shifts to programming):\n"
                        '    "Progressive overload in the back squat follows a periodised structure. '
                        "Novice athletes typically respond to linear progression, adding 2.5-5 kg per "
                        "session across three weekly exposures. Intermediate trainees benefit from "
                        "undulating periodisation, rotating between hypertrophy sets of 8-10 at 70% "
                        "of one-repetition maximum, strength sets of 3-5 at 82-87%, and power-focused "
                        "doubles at 85-90%. The quadriceps, gluteal complex, and spinal erectors all "
                        'contribute to force production during the squat..."\n'
                        "  WHY it works: same title, same muscle names (quadriceps, gluteals), same "
                        "exercise (back squat), but ZERO information about knee angle at depth.\n\n"
                        "EXAMPLE 2:\n"
                        '  Query: "How many hours of sleep do athletes need for optimal muscle recovery?"\n'
                        '  Topic: "Sleep duration for athletic recovery"\n'
                        "  GOOD hard negative (shifts to sleep environment):\n"
                        '    "The sleep environment exerts a measurable influence on sleep architecture. '
                        "Ambient room temperature between 16-19 degrees Celsius promotes the distal "
                        "vasodilation necessary for core body temperature reduction, a prerequisite "
                        "for slow-wave sleep initiation. Blackout curtains eliminate photonic stimulation "
                        "of the suprachiasmatic nucleus, preserving endogenous melatonin secretion. "
                        "Acoustic environments below 30 decibels minimise cortical arousal events "
                        'and reduce fragmentation of restorative deep-sleep stages..."\n'
                        "  WHY it works: same title, same recovery vocabulary, but says NOTHING about "
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


def generate_partial(query: dict, gold_chunk: dict) -> dict | None:
    def _call():
        response = client.chat.completions.create(
            model=MODEL,
            response_format=PARTIAL_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fitness content writer creating partly-relevant documents "
                        "for an information retrieval benchmark. Always respond with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query: {query['text']}\n"
                        f"Topic: {query['topic']}\n"
                        f"Gold answer chunk (first 300 chars): {gold_chunk['text'][:300]}...\n\n"
                        "Generate exactly 1 PARTLY-RELEVANT document chunk.\n\n"
                        "WHAT IS A PARTLY-RELEVANT DOCUMENT?\n"
                        "A partly-relevant document provides SOME useful information toward answering "
                        "the query, but it is INCOMPLETE — it covers only a portion of what the user "
                        "asked, or gives general background without the specific details needed for a "
                        "full answer. A user reading it would say: 'This is helpful background, but it "
                        "doesn't fully answer my question.'\n\n"
                        "CRITICAL RULES:\n"
                        f'- The title MUST be exactly: "{query["topic"]}"\n'
                        "- The document must contain information that is GENUINELY USEFUL for the query "
                        "— not tangential, not a different angle, but actually relevant\n"
                        "- However, it must be MISSING key details that would make it a complete answer\n"
                        "- 150-300 words\n"
                        "- Use proper scientific/anatomical terminology\n\n"
                        "STRATEGIES for partial relevance:\n"
                        "- Give the general principle but omit the specific numbers, sets, reps, or doses\n"
                        "- Explain the WHY without the HOW (or vice versa)\n"
                        "- Cover prerequisite knowledge without answering the actual question\n"
                        "- Discuss the concept at a surface level without the depth the query demands\n"
                        "- Answer part of a multi-part question but not all parts\n\n"
                        "EXAMPLE 1:\n"
                        '  Query: "What grip width should I use for conventional deadlift and does it '
                        'change based on body proportions?"\n'
                        '  Topic: "Grip width for conventional deadlift"\n'
                        "  Gold chunk: specifies shoulder-width grip, just outside knees, adjustments "
                        "for torso-to-arm ratios, mixed vs hook grip options\n"
                        "  GOOD partly-relevant document:\n"
                        '    "Grip selection in barbell pulling movements is influenced by the interplay '
                        "between the glenohumeral joint position, forearm pronation-supination axis, "
                        "and the intended line of force application. In pulling exercises, the grip "
                        "must allow the arms to hang in a mechanically efficient position that minimises "
                        "lateral displacement of the barbell from the lifter's centre of mass. Factors "
                        "such as hand size, forearm length, and shoulder mobility all contribute to "
                        "individual grip preferences. The choice between pronated, supinated, mixed, "
                        "and hook grip affects force transmission through the kinetic chain and can "
                        "influence maximal strength expression at near-maximal loads. Chalk and straps "
                        'are commonly used accessories to mitigate grip-limiting fatigue..."\n'
                        "  WHY it's partly relevant: discusses grip biomechanics generally and mentions "
                        "relevant factors, but never specifies the actual recommended grip WIDTH for "
                        "conventional deadlift or how body proportions change it.\n\n"
                        "EXAMPLE 2:\n"
                        "  Query: \"How should I load creatine monohydrate and what's the right daily "
                        'maintenance dose?"\n'
                        '  Topic: "Creatine monohydrate loading and maintenance dosing"\n'
                        "  Gold chunk: specifies 20g/day loading for 5-7 days, then 3-5g/day maintenance\n"
                        "  GOOD partly-relevant document:\n"
                        '    "Creatine monohydrate is one of the most extensively studied ergogenic '
                        "supplements in exercise science. It functions by increasing intramuscular "
                        "phosphocreatine stores, which serve as a rapid-reserve substrate for adenosine "
                        "triphosphate resynthesis during high-intensity, short-duration efforts such as "
                        "sprinting and resistance training. Meta-analyses consistently demonstrate that "
                        "creatine supplementation improves maximal strength, power output, and lean body "
                        "mass accretion when combined with resistance training. It is well-tolerated "
                        "in healthy populations, with the most common side effect being transient water "
                        "retention during the initial supplementation phase. The International Society "
                        "of Sports Nutrition recognises creatine monohydrate as the most effective "
                        'form among commercially available creatine variants..."\n'
                        "  WHY it's partly relevant: gives solid background on what creatine does and "
                        "that it works, but never mentions specific loading or maintenance doses — "
                        "the user would still need the gold chunk to know HOW MUCH to take.\n\n"
                        "BAD examples (these are NOT partly relevant — they are distractors):\n"
                        "- Writing about creatine TIMING instead of dosing (different angle = distractor)\n"
                        "- Writing about a DIFFERENT supplement entirely (irrelevant)\n"
                        "- Writing about creatine safety concerns (different aspect = distractor)\n\n"
                        f'Return JSON with "title" set to exactly "{query["topic"]}" and a "text" field.'
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    result = call_with_retry(_call)
    if result is None:
        return None
    return result


# ── Per-topic pipeline (query → gold → distractors + partial) ─────────────────


def generate_all_for_topic(topic: str, idx: int) -> dict | None:
    """Generate query, gold chunk, distractors, and partial doc for a single topic.

    Returns a dict with keys: query, gold, distractors, partial — or None on failure.
    """
    q_num = f"{idx + 1:04d}"

    # 1. Query
    query = generate_query(topic, idx)
    if query is None:
        print(f"    [{q_num}] Failed to generate query — skipping topic")
        return None

    # 2. Gold chunk
    gold = generate_gold_chunk(query)
    if gold is None:
        print(f"    [{q_num}] Failed to generate gold chunk — skipping topic")
        return None

    # 3. Distractors + partial (these depend on gold but are independent of each other)
    distractor_list = []
    try:
        raw_distractors = generate_distractors(query, gold, NUM_DISTRACTORS)
        for j, d in enumerate(raw_distractors):
            distractor_list.append({
                "id": f"dist_{q_num}#{j + 1}",
                "title": d["title"],
                "text": d["text"],
            })
    except Exception as e:
        print(f"    [{q_num}] Failed to generate distractors: {e}")

    partial = None
    try:
        raw_partial = generate_partial(query, gold)
        if raw_partial is not None:
            partial = {
                "id": f"part_{q_num}#1",
                "title": raw_partial["title"],
                "text": raw_partial["text"],
            }
    except Exception as e:
        print(f"    [{q_num}] Failed to generate partial: {e}")

    return {
        "query": query,
        "gold": gold,
        "distractors": distractor_list,
        "partial": partial,
    }


# ── Qrels ─────────────────────────────────────────────────────────────────────


def build_qrels(queries: list[dict], partial_docs: list[dict]) -> list[dict]:
    """Gold docs get score 2, partly-relevant docs get score 1, distractors are implicit 0."""
    qrels = [
        {
            "query-id": q["id"],
            "corpus-id": f"doc_{q['id'].replace('q_', '')}",
            "score": 2,
        }
        for q in queries
    ]
    for pdoc in partial_docs:
        q_num = pdoc["id"].split("#")[0].replace("part_", "")
        qrels.append({
            "query-id": f"q_{q_num}",
            "corpus-id": pdoc["id"],
            "score": 1,
        })
    return qrels


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Topics (sequential — each batch needs prior topics for dedup)
    topics = generate_topics(target=100, batch_size=10)
    save_json(topics, DATA_DIR / "topics.json")

    # Step 2: Generate everything per topic in parallel
    print(f"\nGenerating queries, chunks, distractors & partials ({MAX_WORKERS} workers)...")

    results_by_idx: dict[int, dict] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_all_for_topic, topic, i): i
            for i, topic in enumerate(topics)
        }
        for future in as_completed(futures):
            idx = futures[future]
            completed += 1
            try:
                result = future.result()
                if result is not None:
                    results_by_idx[idx] = result
                print(f"  [{completed}/100] Done: q_{idx + 1:04d} — {topics[idx]}")
            except Exception as e:
                print(f"  [{completed}/100] FAILED: q_{idx + 1:04d} — {e}")

    # Step 3: Assemble in deterministic order
    queries = []
    corpus = []
    partial_docs = []

    for idx in sorted(results_by_idx.keys()):
        r = results_by_idx[idx]
        queries.append(r["query"])
        corpus.append(r["gold"])
        corpus.extend(r["distractors"])
        if r["partial"] is not None:
            corpus.append(r["partial"])
            partial_docs.append(r["partial"])

    # Save
    save_json(queries, DATA_DIR / "queries.json")
    save_json(corpus, DATA_DIR / "corpus.json")

    qrels = build_qrels(queries, partial_docs)
    save_json(qrels, DATA_DIR / "qrels.json")

    # Summary
    n_gold = len(queries)
    n_dist = sum(len(r["distractors"]) for r in results_by_idx.values())
    n_part = len(partial_docs)
    print("\nDone!")
    print(f"  Queries                 : {n_gold}")
    print(f"  Corpus total            : {len(corpus)}")
    print(f"    Gold chunks (s=2)     : {n_gold}")
    print(f"    Hard negatives (s=0)  : {n_dist}")
    print(f"    Partly relevant (s=1) : {n_part}")
    print(f"  Qrels                   : {len(qrels)}")


if __name__ == "__main__":
    main()
