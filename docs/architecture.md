# Architecture description

## Overview

The system is a retrieval-augmented question-answering pipeline over 279
CJEU preliminary-ruling judgments (Article 267 TFEU), split into an
environmental corpus (135 judgments) and an agricultural corpus (144
judgments), with 6 judgments present in both. Two independent
answer-generation architectures are implemented against the same underlying
data and tools, so they are directly comparable.

### Modules

| Module | Path | Responsibility |
|---|---|---|
| Ingestion | `src/ingestion/parse_judgments.py` | Parses the raw EUR-Lex/Curia HTML (three distinct HTML templates were found across the corpus) into one JSON document per judgment: full text, numbered paragraphs, operative part (the numbered rulings), and metadata (case number, date, chamber, referring court, Member State, language, legal domain, thematic area, EU instruments cited, keywords). Detects and tags the 6 cross-domain judgments. |
| Embeddings & vector DB | `src/embeddings/` | `chunker.py` splits each judgment into a header chunk, several body chunks (paragraphs greedily merged to ~1000 chars), and an operative-part chunk. `build_index.py` embeds every chunk and builds three FAISS `IndexFlatIP` (cosine) indices: `global` (all 279 judgments, used by Task A) and `agricultural`/`environmental` (used by Task B's specialists; cross-domain judgments are in both). `retriever.py` is the shared query interface, supporting metadata post-filtering and over-fetch-then-filter re-ranking. |
| Single-agent pipeline (Task A) | `src/single_agent/` | `tools.py` defines three tools (`search_corpus`, `get_case_by_number`, `list_cases_by_filter`) as both plain Python functions and Gemini function-calling schemas. `agent.py`'s `SingleAgent` runs one LLM in a loop, calling tools and observing results until it has enough evidence to answer, citing case numbers. |
| Multi-agent pipeline (Task B) | `src/multi_agent/` | `domain_agent.py`'s `DomainAgent` is structurally identical to `SingleAgent` but every tool call is pinned to one legal domain. `supervisor.py`'s `Supervisor` has no direct corpus access; its only tools are `consult_agricultural_agent` / `consult_environmental_agent`, each running a full `DomainAgent` internally. The supervisor is instructed to consult both specialists whenever a question is ambiguous or cross-domain, then synthesizes one final answer. |
| Chat interface & logging | `src/chat_interface/` | `app.py` is a Streamlit chat window over both architectures. `logger.py` appends every turn (question, answer, sources, full step-by-step trace, latency) to `logs/conversation_log.jsonl` in a schema common to both architectures. |
| Evaluation | `src/evaluation/` | `metrics.py` computes RAGAS-inspired metrics that don't require a ground-truth reference (citation consistency as a faithfulness proxy, context utilization as a context-precision proxy, embedding-similarity answer relevancy, abstention detection), plus optional LLM-judge scoring functions. `dashboard.py` is a Streamlit dashboard over the logs. `batch_run.py` runs a question list through either architecture and exports the exam's required `results.xlsx` submission format directly. |

### Data flow (both architectures)

```
raw HTML (2 corpora, 3 HTML templates)
  -> parse_judgments.py -> data/json/{domain}/*.json + manifest.json
  -> chunker.py -> data/embeddings/chunks.jsonl
  -> build_index.py -> FAISS indices (global / agricultural / environmental)
  -> agent.py (Task A) or supervisor.py + domain_agent.py (Task B)
  -> chat_interface/app.py (or batch_run.py)
  -> logger.py -> logs/conversation_log.jsonl
  -> evaluation/dashboard.py + metrics.py
```

### Design choice: cross-domain judgments

The 6 judgments present in both sub-corpora (C-24/21, C-251/21, C-392/23,
C-528/16, C-634/17, and the joined C-364/24 & C-393/24) are **kept as
duplicated entries** in both the agricultural and environmental JSON/chunk
stores and both FAISS sub-indices, rather than deduplicated into a single
domain-agnostic store. This means either routing path (a domain-scoped
search in Task A, or either specialist in Task B) can reach them without
needing a third "both" index or special-casing in the retrieval code — at
the cost of the two domain indices' total vector count exceeding the global
index's count by the (small) overlap.

## Models used

### Embedding model

- **Name:** `sentence-transformers/all-MiniLM-L6-v2`
- **Type:** open-source (downloaded once via Hugging Face, run locally — no
  API calls or per-query cost for embeddings)
- **Motivation:** small (≈90 MB, 384-dim output) and fast, giving strong
  general-purpose semantic search performance for its size, which matters
  since the corpus is chunked into ~14,500 pieces that all need embedding
  at index-build time. The *indexed text* is uniformly English (every
  judgment is stored as its EN CELEX/Curia translation, regardless of the
  case's original procedural language), so multilingual embedding capacity
  was not required for the corpus side. If cross-lingual *querying* is a
  priority (a user typing a question in French or Romanian, say), swapping
  in a multilingual model such as `paraphrase-multilingual-mpnet-base-v2`
  is a one-line change (`MODEL_NAME` in `build_index.py`), at the cost of
  a larger, slower model.

### Generative model

- **Provider:** Google (Gemini API)
- **Model:** `gemini-3.1-flash-lite`
- **API parameters:** function/tool calling via `google-genai`'s
  `GenerateContentConfig(tools=..., automatic_function_calling=disabled)`,
  so the agent loop is driven explicitly (see below) rather than by the
  SDK's automatic tool-execution wrapper, for full trace/logging control.
- **Motivation:** the deciding factor was **cost/access**: Gemini's Flash
  tier has a genuinely free API tier (no credit card, no billing setup
  required), unlike OpenAI or Anthropic, which both require a funded
  account from the first request. This matters concretely for a student
  project run entirely from a personal machine. `gemini-3.1-flash-lite` was
  chosen over the heavier `gemini-3.5-flash` because it is fast and
  inexpensive while still reliably supporting multi-step function calling,
  which is all the agent loop needs (it does not require frontier-level
  reasoning, since the actual legal analysis is grounded in retrieved text,
  not derived from the model's own knowledge).
  **Caveat:** Google iterates on model names/availability faster than this
  document can track — `gemini-2.5-flash` was retired for new accounts
  mid-project. Check https://ai.google.dev/gemini-api/docs/models for the
  current free-tier model list if `SINGLE_AGENT_MODEL`/`MODEL_NAME` stops
  resolving.

### Agent loop implementation

Both `SingleAgent` and `DomainAgent`/`Supervisor` implement the same
explicit ReAct-style loop rather than relying on a framework
(LangChain/LlamaIndex): call the model with the running message history and
available tools; if it returns tool call(s), execute them, append the
observation(s) to the history, and loop; if it returns plain text with no
tool calls, that is the final answer. Every step (tool name, arguments, and
a preview of the observation) is recorded in a `trace` list returned
alongside the answer, which is what both the chat interface's logging and
the evaluation dashboard's metrics consume.
