"""Hand-labeled (query, candidate, expected_label) cases for the Step 4 judge.

These cases are intentionally distinct from the few-shot examples inside
JUDGE_SYSTEM_PROMPT in negative_judge.py so the test measures rule-application,
not memorization.

Each case:
    category — coarse grouping shown in the summary
    query    — user query
    candidate— candidate chunk text
    expected — "positive", "hard_negative", or "irrelevant"
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
        "category": "POSITIVE",
        "query": "What are the primary mechanisms by which different bodily systems, such as the endocrine and nervous systems, communicate with each other?",
        "candidate": (
            "The functions of the endocrine system, nervous system, immune "
            "system, and musculoskeletal system are integrated. Communication "
            "among systems is accomplished with hormones and other peptides."
        ),
        "expected": "positive",
        "note": "qualitative-mechanism answer: 'hormones and other peptides' IS the mechanism — must not be flagged hard_negative for lacking further detail",
    },
    {
        "category": "POSITIVE",
        "query": "What limits the rise in cardiac stroke volume during submaximal aerobic exercise in untrained adults?",
        "candidate": "In untrained adults, stroke volume rises with submaximal aerobic exercise but plateaus around 40 to 60 percent of VO2 max because increasing heart rate shortens diastolic filling time and limits ventricular filling.",
        "expected": "positive",
        "note": "qualitative-mechanism answer: 'shortened diastolic filling time' IS the limit — same shape as the endocrine-communication case but a different domain",
    },
    {
        "category": "HARD_NEGATIVE_TOPICAL",
        "query": "What is the typical loading intensity range for hypertrophy expressed as a percentage of 1RM?",
        "candidate": "Maximal strength training generally uses loads at or above 85 percent of one-repetition maximum with low repetition counts.",
        "expected": "hard_negative",
        "note": "neighbouring goal (max strength) — same units, different prescription",
    },

    # ─────────────────── IRRELEVANT — no topical or surface overlap ───────────────────
    {
        "category": "IRRELEVANT_OFFTOPIC",
        "query": "What box height range is recommended for depth jumps performed by advanced athletes?",
        "candidate": "The rotator cuff comprises the supraspinatus, infraspinatus, teres minor, and subscapularis muscles, which together stabilize the glenohumeral joint during overhead motion.",
        "expected": "irrelevant",
        "note": "shoulder anatomy has no overlap with depth-jump box height",
    },
    {
        "category": "IRRELEVANT_OFFTOPIC",
        "query": "how much caffeine should i take before training and how long beforehand?",
        "candidate": "During a knee extension, the quadriceps generate torque about the tibiofemoral joint that increases as the knee approaches full extension due to the changing moment arm of the patellar tendon.",
        "expected": "irrelevant",
        "note": "knee-extension biomechanics is unrelated to caffeine dose/timing",
    },
    {
        "category": "IRRELEVANT_OFFTOPIC",
        "query": "What is the recommended weekly training frequency per muscle group for hypertrophy?",
        "candidate": "Sweat rate during prolonged endurance exercise can be estimated from pre- and post-exercise body mass changes, and athletes are advised to monitor urine color as a practical hydration indicator.",
        "expected": "irrelevant",
        "note": "hydration monitoring shares no terms or topic with hypertrophy frequency",
    },
]


def by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for c in CASES:
        out.setdefault(c["category"], []).append(c)
    return out
