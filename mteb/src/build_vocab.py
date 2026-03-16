import json
import time
from pathlib import Path
from openai import OpenAI

client = OpenAI()

VOCAB_PATH = Path(__file__).parent.parent / "vocab" / "fitness_vocab.json"


def call_with_retry(fn, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


_MAPPINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "mappings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["mappings"],
    "additionalProperties": False,
}

LAY_TERMS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "lay_to_canonical_mappings",
        "description": "Mapping of informal fitness lay terms to their canonical anatomical/scientific equivalents",
        "schema": _MAPPINGS_SCHEMA,
        "strict": True,
    },
}

ABBREVIATIONS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "abbreviation_mappings",
        "description": "Mapping of fitness abbreviations/acronyms to their full expanded forms",
        "schema": _MAPPINGS_SCHEMA,
        "strict": True,
    },
}


def _parse_mappings(response) -> dict:
    """Convert enforced [{key, value}] envelope to a plain dict."""
    data = json.loads(response.choices[0].message.content)
    return {item["key"]: item["value"] for item in data["mappings"]}


def generate_lay_terms() -> dict:
    print("Generating lay term -> canonical term mappings...")

    def _call():
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format=LAY_TERMS_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fitness terminology expert. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        "Generate exactly 50 fitness lay term to canonical term mappings.\n\n"
                        "Rules:\n"
                        "- Lay terms: informal words people use in gyms, social media, casual conversation\n"
                        "- Canonical terms: anatomically/scientifically correct terminology\n"
                        "- Cover: muscle groups, exercises, training concepts, nutrition terms\n"
                        "- All keys must be lowercase\n\n"
                        "Return a JSON object with a 'mappings' array where each item has 'key' (lay term) "
                        "and 'value' (canonical term).\n\n"
                        "Example:\n"
                        '{"mappings": ['
                        '{"key": "cardio", "value": "cardiovascular exercise"}, '
                        '{"key": "abs", "value": "rectus abdominis"}, '
                        '{"key": "hammies", "value": "hamstrings"}'
                        "]}"
                    ),
                },
            ],
        )
        return _parse_mappings(response)

    return call_with_retry(_call)


def generate_abbreviations() -> dict:
    print("Generating abbreviation -> full form mappings...")

    def _call():
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format=ABBREVIATIONS_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fitness terminology expert. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        "Generate exactly 50 fitness abbreviation to full form mappings.\n\n"
                        "Rules:\n"
                        "- Abbreviations: common acronyms used in fitness, exercise science, gym culture\n"
                        "- Full forms: complete expanded versions\n"
                        "- Cover: training methods, physiological terms, measurement units, program types\n"
                        "- All keys must be uppercase acronyms\n\n"
                        "Return a JSON object with a 'mappings' array where each item has 'key' (abbreviation) "
                        "and 'value' (full form).\n\n"
                        "Example:\n"
                        '{"mappings": ['
                        '{"key": "HIIT", "value": "High Intensity Interval Training"}, '
                        '{"key": "RPE", "value": "Rate of Perceived Exertion"}, '
                        '{"key": "1RM", "value": "One Repetition Maximum"}'
                        "]}"
                    ),
                },
            ],
        )
        return _parse_mappings(response)

    return call_with_retry(_call)


def deduplicate(d: dict) -> dict:
    seen_values = {}
    result = {}
    for k, v in d.items():
        k = k.strip()
        v = v.strip()
        if k and v and v.lower() not in seen_values:
            result[k] = v
            seen_values[v.lower()] = k
    return result


def main():
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    lay_terms = generate_lay_terms()
    lay_terms = deduplicate(lay_terms)
    print(f"  -> {len(lay_terms)} unique lay term entries")

    abbreviations = generate_abbreviations()
    abbreviations = deduplicate(abbreviations)
    print(f"  -> {len(abbreviations)} unique abbreviation entries")

    vocab = {
        "lay_to_canonical": lay_terms,
        "abbreviations": abbreviations,
    }

    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved to {VOCAB_PATH}")
    print(f"Summary:")
    print(f"  lay_to_canonical : {len(lay_terms)} entries")
    print(f"  abbreviations    : {len(abbreviations)} entries")


if __name__ == "__main__":
    main()
