"""Inspect rank-1-flip candidates so we can hand-pick intuitive demo queries.

Reads _all_wins.json and prints, per model, the most compelling wins:
the baseline missed rank-1 and the correct passage was buried deep, the query
reads naturally, and the passages are substantive (not book metadata).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = json.load(open(HERE / "_all_wins.json"))
chunks = W["chunks"]

META_RE = re.compile(
    r"\b(the book is titled|isbn|edition|chapter \d|table of contents|copyright|"
    r"published by|human kinetics|figure \d|^p\. ?\d)", re.I)


def is_substantive(cid):
    c = chunks.get(cid, {}).get("content", "")
    if len(c) < 160:
        return False
    if META_RE.search(c[:160]):
        return False
    return True


def score(r):
    s = 0
    gr = r["baseline"]["gold_rank"] or 99
    s += min(gr, 20)                       # bigger baseline miss = better story
    if r["style"] == "informal":
        s += 4                             # informal reads more intuitively
    ql = len(r["query"])
    if 25 <= ql <= 130:
        s += 3
    gid = r["relevant_chunk_ids"][0]
    bid = r["baseline"]["top"][0]["chunk_id"]
    if is_substantive(gid) and is_substantive(bid):
        s += 6
    return s


tag = sys.argv[1] if len(sys.argv) > 1 else "micro"
wins = sorted(W["wins"][tag], key=score, reverse=True)
print(f"\n##### {tag}: {len(wins)} wins, top 18 by demo-score #####\n")
for r in wins[:18]:
    gid = r["relevant_chunk_ids"][0]
    bid = r["baseline"]["top"][0]["chunk_id"]
    print(f"[score {score(r)}] style={r['style']}  baseline_gold_rank=#{r['baseline']['gold_rank']}")
    print(f"  Q ({r['source_chunk_id'].split('_')[-1]}): {r['query']}")
    print(f"  ✗ baseline#1 [{bid.split('_')[-1]}]: {chunks[bid]['content'][:150].strip()}")
    print(f"  ✓ gold       [{gid.split('_')[-1]}]: {chunks[gid]['content'][:150].strip()}")
    print()
