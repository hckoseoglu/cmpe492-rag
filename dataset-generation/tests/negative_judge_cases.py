"""Hand-labeled (query, candidate, expected_label) cases for the Step 4 judge.

These cases are intentionally distinct from the few-shot examples inside
JUDGE_SYSTEM_PROMPT in negative_judge.py so the test measures rule-application,
not memorization.

Each case:
    style    — "positive" (expected label) or "hard_negative"
    query    — user query
    candidate— candidate chunk text
    expected — "positive" or "hard_negative"
    note     — short reason explaining the test
"""

CASES = [
    # ─────────────────── POSITIVE — chunk fully answers ───────────────────
    {
        "category": "POSITIVE",
        "query": "What rest interval is generally recommended between sets when training for maximum strength with heavy loads?",
        "candidate": "When training for maximal strength with loads at or above 85% of 1RM, rest intervals of 2 to 5 minutes between sets are recommended to allow the phosphocreatine system to recover.",
        "expected": "positive",
        "note": "explicit numerical rest range answers the query in full",
    },
    {
        "category": "POSITIVE",
        "query": "how many grams of protein per kg of bodyweight should i eat to build muscle?",
        "candidate": "For athletes seeking to maximize muscle hypertrophy, daily protein intakes in the range of 1.6 to 2.2 grams per kilogram of body mass are commonly recommended.",
        "expected": "positive",
        "note": "covers the protein/bodyweight ratio with a numeric range",
    },
    {
        "category": "POSITIVE",
        "query": "What approximate work-to-rest ratio is typically used during anaerobic interval training to develop lactic capacity?",
        "candidate": "Anaerobic interval training aimed at developing lactic capacity commonly employs work-to-rest ratios of about 1:3 to 1:4, with high-intensity work bouts of 30 to 90 seconds.",
        "expected": "positive",
        "note": "ratio and bout durations are both present",
    },

    # ─────────────────── HARD_NEGATIVE — partial / topical ───────────────────
    {
        "category": "HARD_NEGATIVE_PARTIAL",
        "query": "What set, repetition, and rest-interval ranges are recommended for hypertrophy in advanced lifters?",
        "candidate": "Hypertrophy training typically uses moderate loads and emphasizes accumulating sufficient mechanical tension across working sets to drive muscle protein synthesis.",
        "expected": "hard_negative",
        "note": "discusses hypertrophy training but gives no numbers for sets/reps/rest",
    },
    {
        "category": "HARD_NEGATIVE_TOPICAL",
        "query": "At what percentage of VO2 max does the lactate threshold occur in well-trained endurance athletes?",
        "candidate": "Lactate is produced as a byproduct of anaerobic glycolysis when the rate of pyruvate production exceeds mitochondrial uptake.",
        "expected": "hard_negative",
        "note": "lactate physiology, but no VO2 max percentage given",
    },
    {
        "category": "HARD_NEGATIVE_DEFINITION",
        "query": "What box height range is recommended for depth jumps performed by advanced athletes?",
        "candidate": "A depth jump is a plyometric drill in which the athlete steps off a raised platform and immediately rebounds upward upon landing.",
        "expected": "hard_negative",
        "note": "definition without box height numbers",
    },
    {
        "category": "HARD_NEGATIVE_PARTIAL",
        "query": "how much caffeine should i take before training and how long beforehand?",
        "candidate": "Caffeine has been shown to improve endurance performance by lowering perceived exertion and increasing fat oxidation during prolonged exercise.",
        "expected": "hard_negative",
        "note": "describes effects but gives neither dose nor timing",
    },
    {
        "category": "HARD_NEGATIVE_TOPICAL",
        "query": "What is the recommended weekly training frequency per muscle group for hypertrophy?",
        "candidate": "Resistance training programs are commonly organized into split routines, full-body routines, or upper/lower splits depending on schedule and recovery needs.",
        "expected": "hard_negative",
        "note": "talks about split structure, no per-muscle frequency recommendation",
    },
    {
        "category": "POSITIVE",
        "query": "What is the typical loading intensity range for hypertrophy expressed as a percentage of 1RM?",
        "candidate": "For hypertrophy, working loads in the range of approximately 67 to 85 percent of one-repetition maximum are commonly prescribed.",
        "expected": "positive",
        "note": "single concise answer to a numerical query",
    },
    {
        "category": "HARD_NEGATIVE_TOPICAL",
        "query": "What is the typical loading intensity range for hypertrophy expressed as a percentage of 1RM?",
        "candidate": "Maximal strength training generally uses loads at or above 85 percent of one-repetition maximum with low repetition counts.",
        "expected": "hard_negative",
        "note": "neighbouring goal (max strength) — same units, different prescription",
    },
]


def by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for c in CASES:
        out.setdefault(c["category"], []).append(c)
    return out
