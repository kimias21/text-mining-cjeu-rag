# Task A — Single-agent ReAct system (Step 3)

- `tools.py` — the agent's toolbox:
  - `search_corpus` — semantic search over the full corpus (global FAISS index),
    with optional `legal_domain` / `thematic_area` / `eu_instrument` / `member_state` filters
  - `get_case_by_number` — exact lookup of one judgment's full text + operative part + metadata
  - `list_cases_by_filter` — pure metadata browsing/counting, no embedding call
- `agent.py` — `SingleAgent`: one LLM in a loop (OpenAI native tool-calling)
  repeatedly choosing a tool, observing its result, until it has enough
  evidence to give a final answer. Every step (tool + args + observation
  preview) is recorded in `result["trace"]` and every case number actually
  touched is recorded in `result["sources"]` — feed both straight into the
  chat interface's logging (Step 5) and evaluation dashboard (Step 6).

Loop shape (maps directly onto a flowchart for the write-up):

```
question -> [LLM: choose tool or answer] -> tool call? --yes--> [run tool] -> observation -> back to LLM
                                                |no
                                                v
                                          final answer (+ cited case numbers)
```

## Setup

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

## Run

```bash
python agent.py "Can mitigation measures be taken into account in a Habitats Directive assessment?"
```

Requires the FAISS indices to already exist (`src/embeddings/index/`, built by
`python src/embeddings/build_index.py` — see Step 2).

Swap models via `SINGLE_AGENT_MODEL` env var (default `gpt-4o-mini`), or pass a
different `model_name=` to `SingleAgent(...)`.
