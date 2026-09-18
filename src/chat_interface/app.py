"""
Step 5 - Chat interface.

A Streamlit chat window over both architectures: pick Single Agent (Task A)
or Multi-Agent (Task B) in the sidebar, ask questions, see the cited case
numbers -- with their legal domain, thematic area, EU instrument and
Member State (guide Sec. 5: "the chat should indicate ... their case
number and referring Member State, the legal domain and thematic area,
the EU instrument concerned") -- and (optionally) the full step-by-step
trace for each answer. Every turn is logged to logs/conversation_log.jsonl
via chat_interface/logger.py, which Step 6's evaluation dashboard reads.

Run with:
    streamlit run app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "single_agent"))
sys.path.insert(0, str(REPO_ROOT / "src" / "multi_agent"))
sys.path.insert(0, str(REPO_ROOT / "src" / "chat_interface"))

from logger import log_turn, get_source_details  # noqa: E402

st.set_page_config(page_title="CJEU Env/Agri RAG", page_icon="⚖️")
st.title("⚖️ CJEU Preliminary Rulings — Env & Agri Law")
st.caption("RAG over 279 CJEU preliminary rulings (environmental + agricultural law)")


@st.cache_resource(show_spinner="Loading single agent...")
def get_single_agent():
    from agent import SingleAgent
    return SingleAgent()


@st.cache_resource(show_spinner="Loading multi-agent supervisor (may take a bit longer)...")
def get_supervisor():
    from supervisor import Supervisor
    return Supervisor()


def _render_sources(source_details: list):
    """Sec. 5 asks the chat to show, for each retrieved judgment, its case
    number and referring Member State, legal domain, thematic area, and EU
    instrument -- not just a bare case-number list. `source_details` comes
    from logger.get_source_details(), which joins the retrieved case
    numbers against data/json/manifest.json."""
    if not source_details:
        return
    case_numbers = [d["case_number"] for d in source_details]
    st.caption("Sources: " + ", ".join(case_numbers))
    with st.expander(f"Source details ({len(source_details)})"):
        for d in source_details:
            instruments = ", ".join(d.get("eu_instruments") or []) or "—"
            st.markdown(
                f"**{d['case_number']}** — "
                f"domain: {d.get('legal_domain') or '—'} · "
                f"area: {d.get('thematic_area') or '—'} · "
                f"instrument: {instruments} · "
                f"Member State: {d.get('member_state') or '—'}"
            )


with st.sidebar:
    st.header("Settings")
    system_choice = st.radio(
        "Architecture", ["Single agent (Task A)", "Multi-agent (Task B)"],
        help="Task A: one agent, full corpus. Task B: supervisor routes to domain specialists.",
    )
    show_trace = st.checkbox("Show step-by-step trace", value=False)
    st.divider()
    st.caption(
        "Every question/answer is logged to `logs/conversation_log.jsonl` "
        "for the evaluation dashboard (Step 6)."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("source_details"):
            _render_sources(msg["source_details"])
        if msg["role"] == "assistant" and msg.get("trace") and show_trace:
            with st.expander("Trace"):
                st.json(msg["trace"])

question = st.chat_input("Ask a question about CJEU environmental or agricultural rulings...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            is_multi = system_choice.startswith("Multi-agent")
            system_name = "multi_agent" if is_multi else "single_agent"
            agent = get_supervisor() if is_multi else get_single_agent()

            start = time.time()
            result = agent.answer(question)
            latency = time.time() - start

            logged_entry = log_turn(system_name, question, result, latency)

        st.markdown(result["answer"])
        source_details = logged_entry.get("source_details") or get_source_details(result.get("sources"))
        if source_details:
            _render_sources(source_details)
        if is_multi and result.get("consulted_domains"):
            st.caption("Specialists consulted: " + ", ".join(sorted(set(result["consulted_domains"]))))
        if show_trace and result.get("trace"):
            with st.expander("Trace"):
                st.json(result["trace"])

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"],
        "source_details": source_details, "trace": result.get("trace"),
    })
