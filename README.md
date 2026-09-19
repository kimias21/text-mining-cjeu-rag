# Text Mining Exam Project — RAG over CJEU Preliminary Rulings (Environmental & Agricultural Law)

RAG system(s) over 279 CJEU preliminary-ruling judgments (Article 267 TFEU), split into
an environmental corpus (135 judgments) and an agricultural corpus (144 judgments), with
6 cross-domain judgments present in both. Built for the Text Mining exam project comparing
a single ReAct-style agent against a multi-agent system with a supervisor.

Reference guide: see `Text_Mining_Exam_Guide_Env_Agri_Preliminary_Rulings.docx` (not tracked
in this repo — keep your local copy for reference).

Starter code baseline: [RAG_4_Scratch](https://github.com/Al-Moccardi/RAG_4_Scratch) by Alberto Moccardi.

## Status

- [x] **Step 1 — HTML → JSON conversion & metadata extraction** (`src/ingestion/`)
- [x] **Step 2 — Embeddings & vector database construction** (`src/embeddings/`)
- [x] **Step 3 — Task A: single-agent ReAct system** (`src/single_agent/`)
- [x] **Step 4 — Task B: multi-agent system with supervisor** (`src/multi_agent/`)
- [x] **Step 5 — Chat interface & conversation logging** (`src/chat_interface/`)
- [x] **Step 6 — Evaluation dashboard (RAGAS-style metrics)** (`src/evaluation/`)
- [x] **Step 7 — Documentation: flowcharts, architecture write-up, performance table, slides** (`docs/`) — results still TBD pending batch run + official evaluation
- [x] **Bonus — Knowledge-Graph-augmented generation** (`src/knowledge_graph/`) — `expand_via_graph` wired into both agent systems as an optional 4th tool; RAGAS with-vs-without comparison still TBD (see `docs/performance_table.md`)

## Repository structure

```
data/
  html/{environmental,agricultural}/   raw judgment HTML files (input)
  json/{environmental,agricultural}/   one JSON per judgment: {text, paragraphs, operative_part, metadata}
  json/manifest.json                    flat index of all 279 judgments' metadata
src/
  ingestion/        HTML -> JSON parsing (parse_judgments.py)
  embeddings/        chunking, embedding computation, vector DB (FAISS) construction
  single_agent/       Task A: ReAct-style single agent
  multi_agent/         Task B: supervisor + specialized agents
  chat_interface/       user-facing chat app + structured conversation logging
  evaluation/            RAGAS-inspired metrics & dashboard
docs/
  flowcharts/         process diagrams for Task A and Task B
  architecture.md      architectural description, models used
  performance_table.md RAGAS metrics comparison across trials
notebooks/            exploratory / experimental notebooks
results_exam_june_2026/   final Excel submission (20 evaluation questions)
```

## Dataset notes

- Each judgment is identified by case number (e.g. `C-100/21`); joined cases share one file
  (e.g. `C-364.24 and C-393.24.html`).
- Cross-domain judgments (currently present in **both** sub-corpora, tagged
  `"legal_domain": "both"` in their metadata): `C-24/21`, `C-251/21`, `C-392/23`, `C-528/16`,
  `C-634/17`, and the joined cases `C-364/24 & C-393/24`.
  **Design choice:** kept as duplicated copies in both `data/json/{environmental,agricultural}`
  sub-corpora (rather than deduplicated into a single multi-valued-domain store), so either
  routing path can reach them. Document this choice in `docs/architecture.md`.
- `thematic_area` in the metadata is a **first-pass, rule-based heuristic** (keyword matching),
  not ground truth — refine it (e.g. with an LLM classification pass) before relying on it for
  routing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Re-run the ingestion step (only needed if the raw HTML changes):

```bash
python src/ingestion/parse_judgments.py
```

Build the chunk store and FAISS indices (needed once, or whenever the JSON
corpus / chunking strategy / embedding model changes -- this downloads the
embedding model weights on first run, so it needs internet access and may
take a few minutes):

```bash
cd src/embeddings
python chunker.py       # data/embeddings/chunks.jsonl
python build_index.py   # src/embeddings/index/{global,agricultural,environmental}.faiss + .meta.json
python retriever.py "your test query here"   # quick sanity check
```

Build the Knowledge Graph (bonus, optional -- both agent systems work fine
without it; running this just adds a 4th `expand_via_graph` tool):

```bash
cd src/knowledge_graph
python build_graph.py
```

## Evaluation

Official evaluation uses RAGAS against 20 hidden-ground-truth questions (see the exam guide's
Appendix). Submit results as `results_exam_june_2026/results_single_agent.xlsx` and
`results_multi_agent.xlsx` with columns: Question, Answer, Context retrieved.
