"""Test cases for the judge LLM in pair_generator.py.

IMPORTANT: every case here is intentionally distinct from the few-shot examples
inside the generator and judge prompts. Avoiding overlap is critical — if a test
case duplicates a prompt example, the judge can pattern-match the answer instead
of applying the rule, and we'd be measuring memorization not generalization.

Topics deliberately AVOIDED because they appear in the prompts: the back squat,
the NSCA hypertrophy 3-6×6-12 prescription, Schoenfeld's squat-depth study,
"the third edition" / "chapters 9 and 10", "what is a compound exercise?",
the 67-85% 1RM hypertrophy reps chunk, "Dr. Jane Smith".

Each case is a dict with:
    category   — which judge rule the case targets (or "VALID")
    style      — "formal" or "informal" (the requested style)
    chunk      — source text (CHUNK in the judge prompt)
    question   — candidate question to evaluate
    expected   — True if the judge SHOULD accept it, False if it SHOULD reject it
    note       — short reason explaining what's being tested

Categories:
    VALID            — well-formed questions the judge must accept
    ANSWERABILITY    — chunk does not contain a full answer
    SPECIFICITY      — generic question that any chunk could answer
    NO_ATTRIBUTION   — names institution / person / study / book
    USER_PERSONA     — references the source itself (book, chapter, edition)
    STYLE            — wrong register for the requested style
"""

CASES = [
    # ───────────────────────── VALID ─────────────────────────
    {
        "category": "VALID",
        "style": "formal",
        "chunk": "The lactate threshold is the exercise intensity at which blood lactate begins to accumulate above resting levels, typically occurring at 50-60% of VO2 max in untrained individuals and 70-80% in well-trained endurance athletes.",
        "question": "At approximately what percentage of VO2 max does the lactate threshold typically occur in untrained individuals versus well-trained endurance athletes?",
        "expected": True,
        "note": "specific, fully answerable from the numerical ranges given, formal tone",
    },
    {
        "category": "VALID",
        "style": "informal",
        "chunk": "Caffeine taken 30-60 minutes before exercise at a dose of 3-6 mg per kilogram of body weight has been shown to improve endurance performance and reduce perceived exertion.",
        "question": "how much caffeine should i take before a workout and how long beforehand for it to actually help?",
        "expected": True,
        "note": "casual gym-goer voice, fully answerable from the dosage and timing in the chunk",
    },
    {
        "category": "VALID",
        "style": "formal",
        "chunk": "Depth jumps for advanced athletes are typically performed from boxes 30-50 cm in height; jumps from boxes higher than 75 cm tend to increase ground contact time and reduce reactive strength index.",
        "question": "What box height range is typically recommended for depth jumps performed by advanced athletes, and what occurs at heights greater than 75 cm?",
        "expected": True,
        "note": "specific to chunk's numerical content, fully answerable, formal third-person tone",
    },

    # ───────────────────── ANSWERABILITY ─────────────────────
    {
        "category": "ANSWERABILITY",
        "style": "formal",
        "chunk": "Delayed-onset muscle soreness typically peaks 24 to 72 hours after unaccustomed eccentric exercise.",
        "question": "What dietary or recovery interventions are most effective at preventing delayed-onset muscle soreness?",
        "expected": False,
        "note": "chunk only states the timing of DOMS; says nothing about prevention",
    },
    {
        "category": "ANSWERABILITY",
        "style": "informal",
        "chunk": "VO2 max is defined as the maximum rate of oxygen consumption attainable during incremental exercise.",
        "question": "what kind of vo2 max numbers do elite endurance athletes hit?",
        "expected": False,
        "note": "chunk only defines VO2 max; provides no numerical values",
    },
    {
        "category": "ANSWERABILITY",
        "style": "formal",
        "chunk": "The conventional deadlift engages the posterior chain, including the hamstrings, gluteus maximus, and erector spinae.",
        "question": "What hand grip width is recommended for the conventional deadlift?",
        "expected": False,
        "note": "chunk lists muscles only; nothing about grip width",
    },

    # ────────────────────── SPECIFICITY ──────────────────────
    {
        "category": "SPECIFICITY",
        "style": "formal",
        "chunk": "The Olympic snatch requires explosive triple extension of the ankles, knees, and hips followed by a rapid pull under the bar into an overhead squat receiving position.",
        "question": "What is weightlifting?",
        "expected": False,
        "note": "generic — any weightlifting chunk could 'answer' this",
    },
    {
        "category": "SPECIFICITY",
        "style": "informal",
        "chunk": "Beta-alanine supplementation increases intramuscular carnosine, which buffers hydrogen-ion accumulation during high-intensity efforts lasting 1-4 minutes.",
        "question": "what are supplements?",
        "expected": False,
        "note": "generic question; not specific to beta-alanine or its mechanism",
    },

    # ───────────────────── NO_ATTRIBUTION ────────────────────
    {
        "category": "NO_ATTRIBUTION",
        "style": "formal",
        "chunk": "The American Heart Association recommends at least 150 minutes of moderate-intensity aerobic activity per week for cardiovascular health in adults.",
        "question": "What is the American Heart Association's recommendation for weekly moderate-intensity aerobic activity in adults?",
        "expected": False,
        "note": "names the institution (American Heart Association) in the question",
    },
    {
        "category": "NO_ATTRIBUTION",
        "style": "informal",
        "chunk": "Research by Stuart McGill shows that maintaining a neutral spine during heavy lifting reduces shear forces on the lumbar discs.",
        "question": "what did stuart mcgill find about neutral spine and shear forces during heavy lifting?",
        "expected": False,
        "note": "names a specific researcher",
    },
    {
        "category": "NO_ATTRIBUTION",
        "style": "formal",
        "chunk": "A 2017 meta-analysis published in Sports Medicine concluded that protein intake of approximately 1.6 grams per kilogram of body weight per day maximizes muscle protein synthesis in resistance-trained individuals.",
        "question": "What protein intake did the 2017 Sports Medicine meta-analysis identify as maximizing muscle protein synthesis in resistance-trained individuals?",
        "expected": False,
        "note": "names a specific paper, year, and journal",
    },
    {
        "category": "NO_ATTRIBUTION",
        "style": "formal",
        "chunk": "World Health Organization physical activity guidelines suggest 75 minutes of vigorous-intensity aerobic activity per week as an alternative to moderate-intensity recommendations.",
        "question": "What does the World Health Organization recommend for weekly vigorous-intensity aerobic activity?",
        "expected": False,
        "note": "names the WHO institution",
    },

    # ───────────────────── USER_PERSONA ──────────────────────
    {
        "category": "USER_PERSONA",
        "style": "formal",
        "chunk": "The second edition of this manual now includes expanded coverage of plyometric training and post-injury rehabilitation protocols.",
        "question": "What new content has been added in the second edition of this manual?",
        "expected": False,
        "note": "references 'the second edition of this manual'",
    },
    {
        "category": "USER_PERSONA",
        "style": "informal",
        "chunk": "Section 4.2 outlines five categories of sport-specific warm-up: general aerobic activation, dynamic stretching, movement preparation, sport-specific drills, and progressive intensity rehearsal.",
        "question": "what warm-up categories are listed in section 4.2?",
        "expected": False,
        "note": "references a numbered section",
    },
    {
        "category": "USER_PERSONA",
        "style": "informal",
        "chunk": "Chapter 12 explains that the rotator cuff stabilizes the glenohumeral joint by compressing the humeral head into the glenoid fossa during overhead movements.",
        "question": "what does chapter 12 say about how the rotator cuff stabilizes the shoulder during overhead movements?",
        "expected": False,
        "note": "uses 'chapter 12' meta-framing",
    },
    {
        "category": "USER_PERSONA",
        "style": "formal",
        "chunk": "This textbook is organized into six parts: exercise physiology, biomechanics, nutrition, program design, recovery, and special populations.",
        "question": "How is this textbook organized?",
        "expected": False,
        "note": "references 'this textbook' and asks about source structure",
    },

    # ───────────────────────── STYLE ─────────────────────────
    {
        "category": "STYLE",
        "style": "informal",
        "chunk": "Static stretching held for 30-60 seconds immediately prior to a maximal-effort task can transiently reduce force production by 5-10%.",
        "question": "What is the magnitude of the transient force-production reduction induced by static stretching held for 30-60 seconds prior to maximal-effort tasks?",
        "expected": False,
        "note": "formal academic tone delivered when informal was requested",
    },
    {
        "category": "STYLE",
        "style": "formal",
        "chunk": "Concurrent training that combines endurance and resistance work can attenuate strength and hypertrophy adaptations, a phenomenon termed the interference effect.",
        "question": "yo why does mixing in cardio mess up my gym gains?",
        "expected": False,
        "note": "casual slang ('yo', 'gym gains') when formal was requested",
    },
]


def by_category() -> dict:
    """Group cases by category for reporting."""
    out: dict = {}
    for c in CASES:
        out.setdefault(c["category"], []).append(c)
    return out
