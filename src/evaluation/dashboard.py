"""
Step 6 - Evaluation dashboard.

Reads logs/conversation_log.jsonl (written by chat_interface/logger.py from
either the chat UI or batch_run.py), computes the ground-truth-free metrics
in metrics.py per turn, and shows:
  - an aggregate comparison table (single_agent vs multi_agent averages),
  - per-turn detail with an optional LLM-judge faithfulness button,
  - a one-click "run the 20 example questions" trigger for both systems
    (wraps batch_run.py), producing results_exam_june_2026/results_*.xlsx.

Run with:
    streamlit run dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "chat_interface"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

from logger import read_logs  # noqa: E402
import metrics  # noqa: E402

st.set_page_config(page_title="RAGAS-inspired Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")
st.caption(
    "Ground-truth-free metrics computed from logged conversations. "
    "Context recall and answer correctness need a reference answer -- see the sidebar."
)

logs = read_logs()

if not logs:
    st.info(
        "No conversations logged yet. Chat with the app (`streamlit run ../chat_interface/app.py`), "
        "or run the batch evaluator below, to populate `logs/conversation_log.jsonl`."
    )

with st.sidebar:
    st.header("Batch-run the 20 example questions")
    st.caption(
        "Runs the guide's Appendix example questions (NOT the official hidden "
        "set) through one architecture, logs every turn, and writes "
        "results_exam_june_2026/results_<system>.xlsx in the required submission format."
    )
    batch_system = st.radio("System", ["single_agent", "multi_agent"], key="batch_system")
    if st.button("Run batch (20 questions)"):
        with st.spinner(f"Running 20 questions through {batch_system}... this can take a few minutes."):
            import subprocess
            result = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "batch_run.py"), "--system", batch_system],
                capture_output=True, text=True,
            )
        if result.returncode == 0:
            st.success("Done. Reloading logs and metrics below.")
            st.text(result.stdout[-2000:])
            st.rerun()
        else:
            st.error("Batch run failed -- see output below.")
            st.text(result.stdout[-1000:] + "\n" + result.stderr[-2000:])

    st.divider()
    st.caption(
        "**Note on metrics shown below:** context precision & context recall & "
        "answer correctness (RAGAS's reference-based metrics) need a ground-truth "
        "reference answer per question, which the exam guide says is only held "
        "by the evaluator. What's shown here are ground-truth-free proxies: "
        "citation consistency (faithfulness proxy), context utilization "
        "(precision proxy), and answer relevancy (embedding similarity)."
    )

if logs:
    rows = []
    for entry in logs:
        m = metrics.compute_all(entry)
        rows.append({
            "timestamp": entry["timestamp"], "system": entry["system"],
            "question": entry["question"][:80] + ("..." if len(entry["question"]) > 80 else ""),
            "citation_consistency": m["citation_consistency"],
            "context_utilization": m["context_utilization"],
            # answer_relevancy is None for a turn whose agent call itself
            # failed (see metrics.compute_all) -- round() would crash on
            # None, so only round a real value.
            "answer_relevancy": round(m["answer_relevancy"], 3) if m["answer_relevancy"] is not None else None,
            # A failed turn isn't a real "didn't abstain" -- leave it as a
            # missing value so groupby(...).mean() skips it below instead
            # of silently diluting the abstention rate.
            "abstained": None if m["failed"] else m["abstained"],
            "num_sources": m["num_sources"],
            "latency_seconds": m["latency_seconds"],
            "failed": m["failed"],
        })
    df = pd.DataFrame(rows)

    st.subheader("Aggregate comparison: single-agent vs multi-agent")
    agg = df.groupby("system").agg(
        n_questions=("question", "count"),
        avg_citation_consistency=("citation_consistency", "mean"),
        avg_context_utilization=("context_utilization", "mean"),
        avg_answer_relevancy=("answer_relevancy", "mean"),
        avg_num_sources=("num_sources", "mean"),
        avg_latency_seconds=("latency_seconds", "mean"),
        abstention_rate=("abstained", "mean"),
    ).round(3)
    st.dataframe(agg, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df.groupby("system")["latency_seconds"].mean(), horizontal=True)
        st.caption("Average latency (s) by system")
    with col2:
        st.bar_chart(df.groupby("system")["answer_relevancy"].mean(), horizontal=True)
        st.caption("Average answer relevancy by system")

    st.subheader("All logged turns")
    st.dataframe(df, use_container_width=True)

    st.subheader("Turn detail")
    idx = st.selectbox("Select a turn", options=range(len(logs)),
                        format_func=lambda i: f"[{logs[i]['system']}] {logs[i]['question'][:60]}")
    entry = logs[idx]
    st.markdown(f"**Question:** {entry['question']}")
    st.markdown(f"**Answer:** {entry['answer']}")
    st.markdown(f"**Sources:** {', '.join(entry.get('sources') or [])}")
    with st.expander("Full trace"):
        st.json(entry.get("trace"))

    if st.button("Run LLM-judge faithfulness on this turn (1 API call)"):
        with st.spinner("Scoring..."):
            from batch_run import format_context
            ctx = format_context(entry.get("trace")) or ", ".join(entry.get("sources") or [])
            score = metrics.faithfulness_llm_judge(entry["question"], entry["answer"], ctx)
        st.metric("Faithfulness (LLM-judge, 1-5)", score)
