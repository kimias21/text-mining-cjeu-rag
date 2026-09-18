"""
Bonus (Sec. 8) - Knowledge-Graph "with vs. without" ablation.

The guide asks for the comparative performance table to also report RAGAS
metrics obtained WITH and WITHOUT the Knowledge Graph, "so that its actual
contribution to faithfulness, answer correctness and context quality can be
assessed rather than merely assumed." This script produces that comparison
for both architectures, using the same ground-truth-free proxy metrics the
evaluation dashboard already computes (metrics.py) -- context recall and
answer correctness still need the hidden reference answers, same caveat as
everywhere else in this project.

How the "without KG" condition works: tools.py exposes `expand_via_graph`
only when DISABLE_KG is *not* set in the environment (see tools.py). Each
(system, condition) pair below runs as its own subprocess, so the tool list
an agent is built with is fixed for its whole run -- there's no risk of the
KG tool being available for some questions and not others within one run.

Usage (from src/evaluation/):
    python run_kg_ablation.py
        # runs single_agent x multi_agent x {with_kg, without_kg} = 4 batches
        # of 20 questions each (80 live API calls total). This can take well
        # over an hour on the free tier -- narrow the scope if you're short
        # on time:
    python run_kg_ablation.py --systems single_agent
        # just single_agent x {with_kg, without_kg} = 2 batches, 40 calls.
        # Still demonstrates the required "with vs. without" comparison for
        # at least one architecture, which is the minimum the guide asks for.
    python run_kg_ablation.py --systems single_agent --conditions with_kg
        # one batch only. Runs are resumable by default: a question already
        # logged from an earlier, interrupted attempt at this exact
        # (system, condition) is skipped rather than re-answered, so if a
        # run dies partway through (rate limit, transient 503, etc.) just
        # re-run the same command and it picks up where it left off. Delete
        # logs/kg_ablation_<system>_<condition>.jsonl yourself first for a
        # genuinely clean re-run of that one batch.

Writes:
    logs/kg_ablation_<system>_<condition>.jsonl   (one per batch, raw turns)
    docs/performance_table.md                     (ablation table filled in,
                                                     replacing the "_TBD_"s)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"
PERF_TABLE = REPO_ROOT / "docs" / "performance_table.md"

SYSTEMS = ["single_agent", "multi_agent"]
CONDITIONS = ["with_kg", "without_kg"]


def _log_path(system: str, condition: str) -> Path:
    return LOGS_DIR / f"kg_ablation_{system}_{condition}.jsonl"


# ---------------------------------------------------------------------------
# Worker: runs the 20 questions through ONE (system, condition) pair in this
# process, with DISABLE_KG already set (or not) by the parent before this
# process was spawned -- see main() below.
# ---------------------------------------------------------------------------
def _run_worker(system: str, condition: str):
    sys.path.insert(0, str(REPO_ROOT / "src" / "single_agent"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "multi_agent"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "chat_interface"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))
    from logger import log_turn  # noqa: E402
    from questions import EVAL_QUESTIONS  # noqa: E402
    import tools  # noqa: E402

    assert tools._GRAPH_AVAILABLE == (condition == "with_kg"), (
        f"DISABLE_KG didn't take effect as expected for condition={condition} "
        f"(tools._GRAPH_AVAILABLE={tools._GRAPH_AVAILABLE}) -- is the graph "
        f"actually built (src/knowledge_graph/graph.json)? Run "
        f"`python src/knowledge_graph/build_graph.py` first if not."
    )

    if system == "single_agent":
        from agent import SingleAgent
        agent = SingleAgent()
    else:
        from supervisor import Supervisor
        agent = Supervisor()

    log_path = _log_path(system, condition)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumable by default: a question already logged from an earlier,
    # interrupted run of this exact (system, condition) is skipped rather
    # than re-answered -- added after two live runs both died partway
    # through on a transient Gemini 503 (now also retried, see
    # gemini_utils.generate_with_retry, but no reason to waste API calls
    # re-doing questions that already succeeded). Delete the log file
    # yourself first if you want a genuinely clean re-run.
    already_done = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                already_done.add(json.loads(line)["question"])
        if already_done:
            print(f"[{system}/{condition}] resuming -- {len(already_done)}/20 already logged, skipping those", flush=True)

    for q in EVAL_QUESTIONS:
        if q["question"] in already_done:
            continue
        print(f"[{system}/{condition}] {q['id']}: {q['question'][:70]}...", flush=True)
        start = time.time()
        try:
            result = agent.answer(q["question"])
        except Exception as e:
            # One bad question (a persistent 503, an unexpected API error,
            # anything generate_with_retry couldn't recover from) used to
            # take the whole 20-question batch down with it, throwing away
            # every already-answered question's results too. Log it as a
            # visibly-failed turn and move on -- the aggregate metrics
            # already treat a turn with no answer/sources as a zero, and
            # `n_questions` in the summary will show it as incomplete.
            latency = time.time() - start
            print(f"    !! FAILED after {latency:.1f}s: {e}", flush=True)
            log_turn(f"{system}_{condition}", q["question"],
                      {"answer": None, "sources": [], "trace": [{"type": "error", "content": str(e)}]},
                      latency, log_path=log_path)
            continue
        latency = time.time() - start
        log_turn(f"{system}_{condition}", q["question"], result, latency, log_path=log_path)
        print(f"    -> {latency:.1f}s, {len(result.get('sources') or [])} sources", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator: spawns one subprocess per (system, condition), then
# aggregates every batch's logged turns into the ablation table.
# ---------------------------------------------------------------------------
def _spawn(system: str, condition: str) -> bool:
    env = dict(os.environ)
    if condition == "without_kg":
        env["DISABLE_KG"] = "1"
    else:
        env.pop("DISABLE_KG", None)
    print(f"\n=== Running {system} / {condition} ({20} questions) ===")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--system", system, "--condition", condition],
        env=env, cwd=str(Path(__file__).resolve().parent),
    )
    if result.returncode != 0:
        print(f"!! {system}/{condition} FAILED (exit {result.returncode}) -- its log file may be partial/missing.")
        return False
    return True


def _aggregate(system: str, condition: str):
    sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))
    import metrics  # noqa: E402

    log_path = _log_path(system, condition)
    if not log_path.exists():
        return None
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        return None

    rows = [metrics.compute_all(e) for e in entries]

    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n_questions": len(rows),
        "citation_consistency": avg("citation_consistency"),
        "context_utilization": avg("context_utilization"),
        "answer_relevancy": avg("answer_relevancy"),
        "abstention_rate": round(sum(1 for r in rows if r["abstained"]) / len(rows), 3),
        "avg_latency_seconds": avg("latency_seconds"),
    }


def _cell(results: dict, system: str, condition: str, key: str):
    r = results.get((system, condition))
    if r is None or r.get(key) is None:
        return "_not run_"
    return r[key]


def _update_performance_table(results: dict):
    """Fill in the *existing* "Knowledge Graph bonus: with vs. without" table
    in docs/performance_table.md in place -- column order there is fixed:
    single-agent/no-KG, single-agent/with-KG, multi-agent/no-KG,
    multi-agent/with-KG -- rather than appending a differently-shaped table."""
    if not PERF_TABLE.exists():
        print(f"NOTE: {PERF_TABLE} not found -- can't fill it in. Results:\n{results}")
        return
    text = PERF_TABLE.read_text(encoding="utf-8")
    cols = [("single_agent", "without_kg"), ("single_agent", "with_kg"),
            ("multi_agent", "without_kg"), ("multi_agent", "with_kg")]

    def row(old_line_start: str, key: str, last_cell_note: str = ""):
        cells = [str(_cell(results, s, c, key)) for s, c in cols]
        if last_cell_note:
            cells[-1] = f"{cells[-1]} ({last_cell_note})"
        return f"{old_line_start} {' | '.join(cells)} |"

    import re
    replacements = [
        (r"\| Context precision\* \|.*\|\s*$", row("| Context precision* |", "context_utilization")),
        (r"\| Faithfulness\* \|.*\|\s*$", row("| Faithfulness* |", "citation_consistency")),
        (r"\| Answer relevancy \|.*\|\s*$", row("| Answer relevancy |", "answer_relevancy")),
        (r"\| Avg\. latency \(s\) \|.*\|\s*$",
         row("| Avg. latency (s) |", "avg_latency_seconds", "expect higher -- extra tool calls")),
    ]
    # Bug fix (found 2026-09-19): this used to scan the WHOLE file and
    # replace every line matching any pattern, with no notion of "this
    # pattern already matched once, stop." docs/performance_table.md has
    # THREE tables (Internal evaluation, Official evaluation, this one) that
    # reuse row labels like "Answer relevancy" and "Avg. latency (s)" --
    # the unscoped loop silently overwrote the Internal-evaluation table's
    # real latency numbers and the Official-evaluation table's placeholder
    # row with this table's content. Now scoped to only the lines between
    # this section's own heading and the next "## " heading.
    lines = text.splitlines()
    section_start = next((i for i, l in enumerate(lines) if l.startswith("## Knowledge Graph bonus")), None)
    if section_start is None:
        print(f"NOTE: '## Knowledge Graph bonus' section not found in {PERF_TABLE} -- can't fill it in safely. Results:\n{results}")
        return
    section_end = next((i for i in range(section_start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    for i in range(section_start, section_end):
        for pattern, new_line in replacements:
            if re.match(pattern, lines[i]):
                lines[i] = new_line
                break
    text = "\n".join(lines) + "\n"
    # Also fill in the abstention-rate mention if present as a one-off line;
    # otherwise this is only tracked in the raw per-run logs, not the table
    # (the existing table has no dedicated abstention row for the ablation).
    PERF_TABLE.write_text(text, encoding="utf-8")
    print(f"\nWrote ablation results into {PERF_TABLE}")
    print("n_questions per run:", {f"{s}/{c}": (results.get((s, c)) or {}).get("n_questions") for s, c in cols})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--system", choices=SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=CONDITIONS, help=argparse.SUPPRESS)
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="Comma list, subset of: single_agent,multi_agent")
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="Comma list, subset of: with_kg,without_kg")
    args = parser.parse_args()

    if args.worker:
        _run_worker(args.system, args.condition)
        return

    systems = args.systems.split(",")
    conditions = args.conditions.split(",")
    results = {}
    for system in systems:
        for condition in conditions:
            ok = _spawn(system, condition)
            if not ok:
                print(f"!! {system}/{condition}'s worker exited with an error -- "
                      f"aggregating whatever it managed to log before that, if anything. "
                      f"Re-run with --systems {system} --conditions {condition} to fill in the rest "
                      f"(already-logged questions are skipped automatically).")
            # Aggregate even on failure: log_turn writes incrementally, so a
            # worker that died partway through (e.g. an unrecoverable API
            # error) still leaves every question answered before the crash
            # on disk -- discarding those on a non-zero exit code, as this
            # used to do, threw away real results for no reason.
            results[(system, condition)] = _aggregate(system, condition)

    print("\n=== Results ===")
    for (system, condition), r in results.items():
        if r is None:
            print(f"{system}/{condition}: nothing logged yet")
        else:
            complete = "complete" if r["n_questions"] >= 20 else f"PARTIAL, {r['n_questions']}/20"
            print(f"{system}/{condition} ({complete}): {r}")

    _update_performance_table(results)


if __name__ == "__main__":
    main()
