"""
Step 6 - Batch evaluation runner.

Runs a list of questions through one architecture (single_agent or
multi_agent), logs every turn (via chat_interface/logger.py), and writes an
Excel file matching the exam guide's required submission format exactly:
one row per question, columns Question / Answer / Context retrieved.

For the OFFICIAL submission: once the hidden 20 questions are released,
replace EVAL_QUESTIONS' source (or point --questions at a JSON file with the
same [{"id":..., "question":...}, ...] shape) and re-run for both systems.

Usage:
    python batch_run.py --system single_agent
    python batch_run.py --system multi_agent
    python batch_run.py --system single_agent --questions my_questions.json --out custom.xlsx
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "single_agent"))
sys.path.insert(0, str(REPO_ROOT / "src" / "multi_agent"))
sys.path.insert(0, str(REPO_ROOT / "src" / "chat_interface"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

from logger import log_turn  # noqa: E402
from questions import EVAL_QUESTIONS  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "results_exam_june_2026"


def format_context(trace) -> str:
    """Turn a trace into the 'documents/passages retrieved' string the
    guide asks for -- case numbers actually observed, in order of first
    appearance, deduped. Used as a fallback when result['sources'] is
    empty; recurses into multi-agent 'delegate' steps' nested sub_trace."""
    seen = []

    def scan(steps):
        for step in steps or []:
            if step.get("type") == "delegate":
                scan(step.get("sub_trace"))
            preview = step.get("observation_preview", "")
            for token in preview.replace('"', ' ').split():
                cleaned = token.rstrip(",;.")
                if cleaned.startswith("C-") and "/" in cleaned and cleaned not in seen:
                    seen.append(cleaned)

    scan(trace)
    return "; ".join(seen)


def run_batch(system: str, questions: list, out_path: Path):
    if system == "single_agent":
        from agent import SingleAgent
        agent = SingleAgent()
    elif system == "multi_agent":
        from supervisor import Supervisor
        agent = Supervisor()
    else:
        raise ValueError("system must be 'single_agent' or 'multi_agent'")

    rows = []
    for q in questions:
        print(f"[{system}] {q['id']}: {q['question'][:80]}...")
        start = time.time()
        result = agent.answer(q["question"])
        latency = time.time() - start
        log_turn(system, q["question"], result, latency)

        rows.append({
            "Question": f"{q['id']} - {q['question']}",
            "Answer": result.get("answer", ""),
            "Context retrieved": "; ".join(result.get("sources") or []) or format_context(result.get("trace")),
        })
        print(f"    -> {latency:.1f}s, sources: {result.get('sources')}")

    df = pd.DataFrame(rows, columns=["Question", "Answer", "Context retrieved"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=["single_agent", "multi_agent"])
    parser.add_argument("--questions", default=None, help="Path to a JSON file [{'id':..,'question':..}, ...]. Defaults to the guide's 20 example questions.")
    parser.add_argument("--out", default=None, help="Output .xlsx path. Defaults to results_exam_june_2026/results_<system>.xlsx")
    args = parser.parse_args()

    questions = EVAL_QUESTIONS
    if args.questions:
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"results_{args.system}.xlsx"
    run_batch(args.system, questions, out_path)
