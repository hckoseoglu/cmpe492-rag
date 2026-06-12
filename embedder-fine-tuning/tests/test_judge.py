"""Run the judge LLM against curated cases and report accuracy.

Usage:
    cd dataset-generation
    python -m tests.test_judge                    # run all categories
    python -m tests.test_judge --category VALID   # run only one category
    python -m tests.test_judge --save             # also write tests/judge_results.json

Requires the same env as pair_generator.py (defaults to local Ollama gemma2:9b).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the parent dataset-generation/ importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from pair_generator import judge_question  # noqa: E402
from tests.judge_cases import CASES, by_category  # noqa: E402


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def evaluate_case(llm: LLMClient, case: dict) -> dict:
    chunk = {"content": case["chunk"], "summary": ""}
    t0 = time.time()
    try:
        verdict = judge_question(llm, chunk, case["question"], case["style"])
        elapsed = time.time() - t0
        actual = bool(verdict["valid"])
        reason = verdict.get("failure_reason", "")
        error = None
    except Exception as e:
        elapsed = time.time() - t0
        actual = None
        reason = ""
        error = str(e)

    correct = (actual is not None) and (actual == case["expected"])
    return {
        "category": case["category"],
        "style": case["style"],
        "expected": case["expected"],
        "actual": actual,
        "correct": correct,
        "judge_reason": reason,
        "error": error,
        "elapsed_s": round(elapsed, 2),
        "note": case["note"],
        "question": case["question"],
    }


def fmt_bool(v):
    if v is None:
        return "ERR"
    return "ACCEPT" if v else "REJECT"


def print_result(idx: int, total: int, result: dict):
    tag = f"{GREEN}PASS{RESET}" if result["correct"] else f"{RED}FAIL{RESET}"
    cat = result["category"]
    style = result["style"]
    expected = fmt_bool(result["expected"])
    actual = fmt_bool(result["actual"])
    print(
        f"[{idx:>2}/{total}] {tag} {cat:<14} style={style:<8} "
        f"expected={expected} actual={actual}  ({result['elapsed_s']}s)"
    )
    print(f"     {DIM}note:{RESET} {result['note']}")
    print(f"     {DIM}Q:   {RESET} {result['question']}")
    if result["error"]:
        print(f"     {RED}error:{RESET} {result['error']}")
    elif not result["correct"]:
        # The interesting part on failures: what did the judge actually say?
        reason = result["judge_reason"] or "(empty)"
        print(f"     {DIM}judge_reason:{RESET} {reason}")
    print()


def print_summary(results: list[dict]):
    print("=" * 60)
    print("Summary by category")
    print("=" * 60)

    grouped: dict = {}
    for r in results:
        grouped.setdefault(r["category"], []).append(r)

    for cat, rs in grouped.items():
        passed = sum(1 for r in rs if r["correct"])
        total = len(rs)
        pct = (passed / total * 100) if total else 0
        color = GREEN if passed == total else RED
        print(f"  {cat:<16} {color}{passed}/{total}{RESET}  ({pct:.0f}%)")

    overall_pass = sum(1 for r in results if r["correct"])
    overall_total = len(results)
    overall_pct = (overall_pass / overall_total * 100) if overall_total else 0
    color = GREEN if overall_pass == overall_total else RED
    print("-" * 60)
    print(f"  {'OVERALL':<16} {color}{overall_pass}/{overall_total}{RESET}  ({overall_pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test the judge LLM against curated cases.")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Run only one category (VALID, ANSWERABILITY, SPECIFICITY, NO_ATTRIBUTION, USER_PERSONA, STYLE)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write detailed results to tests/judge_results.json",
    )
    args = parser.parse_args()

    cases = CASES
    if args.category:
        cat = args.category.upper()
        cases = [c for c in CASES if c["category"] == cat]
        if not cases:
            print(f"No cases for category '{cat}'. Known: {sorted(by_category().keys())}")
            sys.exit(1)

    config = Config()
    llm = LLMClient(config)

    print(f"Running {len(cases)} judge case(s) against model={config.model} @ {config.base_url}\n")

    results = []
    for i, case in enumerate(cases, 1):
        result = evaluate_case(llm, case)
        results.append(result)
        print_result(i, len(cases), result)

    print_summary(results)

    if args.save:
        out_path = Path(__file__).resolve().parent / "judge_results.json"
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": config.model,
            "base_url": config.base_url,
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
