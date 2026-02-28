"""
sub_map.py

Three substitution-map presets ranked by obfuscation intensity:

    SOFT_MAP   – muscles + fix "lat" bug + critical missed query terms
    MEDIUM_MAP – adds exercises, equipment, movement terms
    HARD_MAP   – maximal coverage: anatomy directions, body parts, training jargon

Each map is {real_term: nonsense_word}.
Keys are matched case-insensitively; longer keys are tried first.

IMPORTANT – word-boundary safety:
    Very short keys (≤3 chars) like "lat" or "rep" match inside longer words
    (e.g. "lateral", "repetition") causing corruption.
    → Removed "lat" (use "lats" as shortest form).
    → Short keys should only be used if they cannot be a substring
      of any word in the source corpus.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SOFT — Core muscle names + critical query terms that were missing
# ═══════════════════════════════════════════════════════════════════════════════
#
# Purpose: establishes baseline obfuscation.  Only the most recognisable
# domain terms are swapped.  If the LLM can still reverse-engineer these,
# the substitution strategy needs fundamental improvement.

SOFT_MAP: dict[str, str] = {
    # ── Muscle Groups ────────────────────────────────────────────────────────
    # Biceps
    "biceps brachii": "kaskush",
    "biceps": "kaskush",
    # Triceps
    "triceps brachii": "blorbex",
    "triceps": "blorbex",
    # Brachialis
    "brachialis": "greltak",
    # Brachioradialis
    "brachioradialis": "veldrak",
    # Deltoid
    "anterior deltoid": "pruxton",
    "posterior deltoid": "pruxton",
    "lateral deltoid": "pruxton",
    "deltoids": "pruxton",
    "deltoid": "pruxton",
    # Trapezius
    "trapezius": "waxlorn",
    "traps": "waxlorn",
    # Latissimus Dorsi  (FIXED: removed bare "lat" — it corrupts "lateral",
    #                    "isolation", "flat", etc.)
    #                    Also covers common single-s typo "latisimus".
    "latissimus dorsi": "fomblur",
    "latisimus dorsi": "fomblur",  # typo variant (1 s)
    "latissimus": "fomblur",
    "latisimus": "fomblur",  # typo variant (1 s)
    "lats": "fomblur",
    # Pectoralis
    "pectoralis major": "zindrop",
    "pectoralis minor": "zindrop",
    "pectoralis": "zindrop",
    "pectorals": "zindrop",
    "pectoral": "zindrop",
    # Supraspinatus
    "supraspinatus": "trimfaz",
    # Quadriceps
    "quadriceps": "snorvik",
    "quads": "snorvik",
    # Hamstrings
    "hamstrings": "grelvak",
    "hamstring": "grelvak",
    # Gluteus
    "gluteus maximus": "draxon",
    "gluteus medius": "draxon",
    "gluteus minimus": "draxon",
    "gluteus": "draxon",
    "glutes": "draxon",
    "glute": "draxon",
    # Gastrocnemius / Calves
    "gastrocnemius": "worbel",
    "calves": "worbel",
    "calf": "worbel",
    # ── Critical Query Terms (previously MISSING) ────────────────────────────
    # Body parts that appear unsubstituted in queries — dead giveaways
    "shoulder": "jorvik",
    "chest": "plindor",
    # Equipment that appears bare in queries
    "barbell": "crondak",
    "barbel": "crondak",  # common misspelling
    "dumbbell": "frozzle",
    "dumbell": "frozzle",  # common misspelling
    "dumbel": "frozzle",  # common misspelling
    "dumbbells": "frozzles",
    # EZ-Bar (carried over, already existed)
    "ez-bar preacher curl": "takino preacher sana",
    "ez bar preacher curl": "takino preacher sana",
    "ez-bar curl": "takino sana",
    "ez bar curl": "takino sana",
    "ez-bar": "takino",
    "ez bar": "takino",
    # ── Movement Terms (carried over) ────────────────────────────────────────
    "supinated": "florbin",
    "supination": "florbin",
    "abduction": "glenthorp",
    "adduction": "blenford",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM — Soft + exercises, equipment, grip & movement vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

MEDIUM_MAP: dict[str, str] = {
    **SOFT_MAP,
    # ── Exercise names ───────────────────────────────────────────────────────
    "preacher curl": "vrentak sana",
    "curl": "sana",
    "shoulder press": "jorvik threnn",
    "bench press": "plindor threnn",
    "press": "threnn",
    "pull-down": "zolvik",
    "pulldown": "zolvik",
    "pull-up": "zolvik-ur",
    "pullup": "zolvik-ur",
    "dips": "glonfer",
    "dip": "glonfer",
    "lateral raise": "yulbek grimpf",
    "flys": "skentop",
    "fly": "skentop",
    "rows": "brentik",
    "row": "brentik",
    "squat": "vromble",
    "deadlift": "krunfet",
    # ── Equipment ────────────────────────────────────────────────────────────
    "cable": "zenphor",
    "machine": "vroglet",
    # ── Grip types ───────────────────────────────────────────────────────────
    "close-grip": "snelf-grip",
    "wide-grip": "brank-grip",
    "overhand": "ploxin",
    "underhand": "druffen",
    "neutral grip": "quintel grip",
    "pronated": "molvex",
    "grip": "thrompf",
    # ── Movement terms ───────────────────────────────────────────────────────
    "flexion": "glenthor",
    "extension": "braxnol",
    "rotation": "spundel",
    "internal rotation": "spundel-in",
    "external rotation": "spundel-ex",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HARD — Medium + anatomical directions, body parts, training jargon
# ═══════════════════════════════════════════════════════════════════════════════

HARD_MAP: dict[str, str] = {
    **MEDIUM_MAP,
    # ── Anatomical body parts ────────────────────────────────────────────────
    "forearm": "drukkel",
    "upper arm": "splendok",
    "elbow": "brinzel",
    "wrist": "klaxfen",
    "spine": "vrothel",
    "torso": "brelnak",
    "scapula": "flimzon",
    "shoulder blade": "flimzon",
    "humerus": "grenton",
    "knee": "plaxum",
    "hip": "dronfel",
    # ── Anatomical directions ────────────────────────────────────────────────
    "anterior": "qelvik",
    "posterior": "bremzol",
    "lateral": "yulbek",
    "medial": "snervox",
    "inner": "krendal",
    "outer": "vrublix",
    "upper": "glimphor",
    "lower": "zolphen",
    # ── Training concepts ────────────────────────────────────────────────────
    "repetition": "drombek",
    "repetitions": "drombeks",
    "isolation": "sprenvik",
    "compound": "vrelthok",
    "momentum": "glixdor",
    "contraction": "frenbak",
    "range of motion": "splindek",
    "muscle": "threnvox",
    "muscles": "threnvoxes",
    # ── Positions ────────────────────────────────────────────────────────────
    "seated": "brolnex",
    "standing": "frenpak",
}
