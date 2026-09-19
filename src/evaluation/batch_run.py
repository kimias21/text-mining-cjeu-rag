"""
Step 6 - Batch evaluation runner.

Runs a list of questions through one architecture (single_agent or
multi_agent), logs every turn (via chat_interface/logger.py), and writes an
Excel file matching the exam guide's required submission format exactly:
one row per question, columns Question / Answer / Context retrieved.

Resilient and resumable (fixed 2026-09-19, mirroring the same fix already
applied to run_kg_ablation.py after it happened live twice during the KG
ablation run): a per-question failure (an unrecoverable 503, an unexpected
API error, anything generate_with_retry couldn't ride out) used to crash
the WHOLE batch with an unhandled exception, and since the Excel file was
only ever written once at the very end, that meant a single bad question
threw away every already-answered question too -- the exported file ended
up with zero rows despite real answers having been produced. Now:
  - every question is wrapped in try/except; a failure is logged and
    written as an "[ERROR ...]" row rather than crashing the run,
  - the Excel file is rewritten after EVERY question, not just at the end,
    so a crash/kill/lost connection loses at most the one question in
    flight, not the whole batch,
  - re-running the same command skips questions that already have a real
    (non-error) answer in the existing output file, and retries only the
    ones that are missing or previously failed.

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
ERROR_PREFIX = "[ERROR -- system failed to answer: "


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


def _load_existing_rows(out_path: Path) -> dict:
    """Map 'QID - question text' -> already-written row, from a prior
    (possibly partial or failed) run's own output file, so a re-run skips
    real work already done instead of re-spending API calls -- and, on the
    free tier, re-risking the exact rate limit/503 that may have killed the
    previous attempt. A row whose Answer is one of our own ERROR_PREFIX
    markers is deliberately NOT included here, so a previously-failed
    question gets retried rather than permanently skipped."""
    if not out_path.exists():
        return {}
    try:
        df = pd.read_excel(out_path)
    except Exception as e:
        print(f"NOTE: couldn't read existing {out_path} to resume from ({e}) -- starting fresh.")
        return {}
    existing = {}
    for _, row in df.iterrows():
        answer = row.get("Answer")
        if isinstance(answer, str) and answer.startswith(ERROR_PREFIX):
            continue
        existing[str(row["Question"])] = row.to_dict()
    return existing


def run_batch(system: str, questions: list, out_path: Path):
    if system == "single_agent":
        from agent import SingleAgent
        agent = SingleAgent()
    elif system == "multi_agent":
        from supervisor import Supervisor
        agent = Supervisor()
    else:
        raise ValueError("system must be 'single_agent' or 'multi_agent'")

    existing = _load_existing_rows(out_path)
    if existing:
        print(f"Resuming from {out_path} -- {len(existing)}/{len(questions)} already answered, skipping those.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    for q in questions:
        key = f"{q['id']} - {q['question']}"
        if key in existing:
            rows.append(existing[key])
            continue

        print(f"[{system}] {q['id']}: {q['question'][:80]}...")
        start = time.time()
        try:
            result = agent.answer(q["question"])
            latency = time.time() - start
            log_turn(system, q["question"], result, latency)
            answer = result.get("answer", "")
            context = "; ".join(result.get("sources") or []) or format_context(result.get("trace"))
            print(f"    -> {latency:.1f}s, sources: {result.get('sources')}")
        except Exception as e:
            latency = time.time() - start
            print(f"    !! FAILED after {latency:.1f}s: {e}")
            log_turn(system, q["question"],
                     {"answer": None, "sources": [], "trace": [{"type": "error", "content": str(e)}]},
                     latency)
            answer = f"{ERROR_PREFIX}{e}]"
            context = ""
            failures.append(q["id"])

        rows.append({"Question": key, "Answer": answer, "Context retrieved": context})

        # Rewrite after every question (cheap for 20 rows) so the file on
        # disk always reflects real progress -- see the module docstring.
        pd.DataFrame(rows, columns=["Question", "Answer", "Context retrieved"]).to_excel(out_path, index=False)

    if failures:
        print(f"\nWrote {len(rows)} rows -> {out_path} "
              f"({len(failures)} FAILED: {', '.join(failures)} -- re-run the same command to retry just those)")
    else:
        print(f"\nWrote {len(rows)} rows -> {out_path} (complete)")


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
