"""Emit website/data.js (window.DEMO_DATA) from the hand-picked demo queries.

Selections below are source_chunk_id numbers chosen from _all_wins.json: each is
a held-out query where the off-the-shelf baseline ranked a topically-related but
WRONG passage at #1, and the fine-tuned model ranked a labeled-correct passage at
#1. Picked for intuitive phrasing and readable, on-point passages.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = json.load(open(HERE / "_all_wins.json"))
chunks = W["chunks"]

PICKS = {
    "micro":    ["0380", "0416", "0215", "0301", "2672"],
}
ORDER = ["micro"]  # only the bge-micro-v2 retriever is shown


def compact_side(side):
    t = side["top"][0]
    return {"chunk_id": t["chunk_id"], "score": t["score"],
            "gold_rank": side["gold_rank"], "rank1_ok": side["rank1_ok"]}


examples = {}
needed = set()
for tag, nums in PICKS.items():
    bynum = {r["source_chunk_id"].split("_")[-1]: r for r in W["wins"][tag]}
    rows = []
    for n in nums:
        r = bynum[n]
        rows.append({
            "query": r["query"],
            "style": r["style"],
            "baseline": compact_side(r["baseline"]),
            "finetuned": compact_side(r["finetuned"]),
        })
        needed.add(r["baseline"]["top"][0]["chunk_id"])
        needed.add(r["finetuned"]["top"][0]["chunk_id"])
    examples[tag] = rows

chunk_text = {cid: chunks[cid] for cid in needed if cid in chunks}

payload = {
    "order": ORDER,
    "models": {t: W["models"][t] for t in ORDER},
    "examples": examples,
    "chunks": chunk_text,
}

out = HERE / "data.js"
with open(out, "w") as f:
    f.write("window.DEMO_DATA = ")
    json.dump(payload, f, ensure_ascii=False, indent=1)
    f.write(";\n")
print(f"wrote {out}  ({sum(len(v) for v in examples.values())} examples, {len(chunk_text)} chunks)")
