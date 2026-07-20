# Chat interface & logging (Step 5)

- `app.py` — Streamlit chat window. Pick **Single agent (Task A)** or
  **Multi-agent (Task B)** in the sidebar, ask questions, see cited case
  numbers and (optionally) the full step-by-step trace per answer.
- `logger.py` — `log_turn(system, question, result, latency_seconds)` appends
  one JSON line per turn to `logs/conversation_log.jsonl` in a schema common
  to both architectures (question, answer, sources, trace, consulted_domains
  [multi-agent only], latency). `read_logs()` reads it back — Step 6's
  evaluation dashboard uses this directly.

## Setup

```bash
pip install streamlit
export GEMINI_API_KEY=...
```

Requires both Step 2's FAISS indices and Steps 3/4's agent code already in
place.

## Run

```bash
streamlit run app.py
```

Opens in your browser (usually http://localhost:8501). First question on
each architecture takes a little longer (models/agents are loaded lazily and
cached via `st.cache_resource`).

## Log format

```json
{
  "timestamp": "2026-07-20T13:00:00+00:00",
  "system": "single_agent",
  "question": "...",
  "answer": "...",
  "sources": ["C-116/20"],
  "consulted_domains": null,
  "trace": [...],
  "latency_seconds": 4.21
}
```
