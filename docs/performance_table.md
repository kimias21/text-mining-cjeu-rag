# Comparative performance table

Per the exam guide (Sec. 7.2), this table reports RAGAS metrics for the two
architectures. **Fill in the numeric columns after running `batch_run.py`
for both systems and reviewing the dashboard** (`src/evaluation/dashboard.py`);
placeholders below show the intended shape and are NOT results.

## Internal evaluation (guide's 20 example questions, no disclosed ground truth)

Context precision, context recall, and answer correctness need a reference
answer per question (see `src/evaluation/metrics.py`'s docstring) and are
genuinely **not computable** without one — the guide is explicit that even
these 20 example questions' "correct" answers are not disclosed to
students, only the official hidden set's are (to the evaluator, not to
us). The columns below marked * are therefore the ground-truth-free proxies
implemented in `metrics.py`, not the literal RAGAS metric.

| Metric | Single-agent (Task A) | Multi-agent (Task B) | Notes |
|---|---|---|---|
| Context precision* (proxy: context utilization — fraction of retrieved case numbers actually cited) | _fill in from dashboard_ | _fill in from dashboard_ | Proxy only |
| Context recall | not computable without reference answers | not computable without reference answers | Needs ground truth |
| Faithfulness* (proxy: citation consistency — fraction of cited case numbers that were genuinely retrieved) | _fill in from dashboard_ | _fill in from dashboard_ | Proxy only; optional LLM-judge score available via `metrics.faithfulness_llm_judge` |
| Answer relevancy (embedding similarity, question vs. answer) | _fill in from dashboard_ | _fill in from dashboard_ | |
| Answer correctness | not computable without reference answers | not computable without reference answers | Needs ground truth |
| Avg. latency (s) | _fill in from dashboard_ | _fill in from dashboard_ | Multi-agent is expected to be slower (sequential specialist consultations) |
| Abstention rate (Q20-style questions) | _fill in from dashboard_ | _fill in from dashboard_ | |

**How to fill this in:**
```bash
cd src/evaluation
python batch_run.py --system single_agent
python batch_run.py --system multi_agent
streamlit run dashboard.py   # read off the aggregate comparison table
```

## Official evaluation (hidden 20-question set, RAGAS against undisclosed reference answers)

To be filled in once the evaluator's report is returned. This is the
authoritative comparison for the report — the internal metrics above are a
proxy for iterating during development, not a substitute.

| Metric | Single-agent (Task A) | Multi-agent (Task B) |
|---|---|---|
| Context precision | _pending official evaluation_ | _pending official evaluation_ |
| Context recall | _pending official evaluation_ | _pending official evaluation_ |
| Faithfulness | _pending official evaluation_ | _pending official evaluation_ |
| Answer relevancy | _pending official evaluation_ | _pending official evaluation_ |
| Answer correctness | _pending official evaluation_ | _pending official evaluation_ |

## Knowledge Graph bonus: with vs. without (guide Sec. 8 requirement)

The guide asks for RAGAS metrics obtained **with and without** the
Knowledge Graph, so its actual contribution can be assessed rather than
assumed. To produce this: run `batch_run.py` once with
`src/knowledge_graph/graph.json` absent/unbuilt (the `expand_via_graph`
tool is simply not offered to the agents in that case) and once with it
built (`python src/knowledge_graph/build_graph.py` first), then compare.

| Metric | Single-agent, no KG | Single-agent, with KG | Multi-agent, no KG | Multi-agent, with KG |
|---|---|---|---|---|
| Context precision* | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Faithfulness* | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Answer relevancy | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Answer correctness | not computable without reference | not computable without reference | not computable without reference | not computable without reference |
| Avg. latency (s) | _TBD_ | _TBD_ | _TBD_ | _TBD_ (expect higher: extra tool calls) |

## Discussion (fill in after results are available)

- **Strengths/weaknesses of each approach:** _e.g. does the multi-agent
  system's "consult both when ambiguous" rule measurably help on
  cross-domain questions (guide's Q02, Q04, Q09 style) at the cost of
  latency? Does the single agent's unified index ever mis-route a
  domain-specific question that a forced specialist wouldn't?_
- **Abstention behavior (Q20-style):** _did either system hallucinate a
  case citation instead of admitting the corpus lacks a direct answer?_
- **Scaling assessment:** _which routing strategy seems more tractable if
  more legal domains were added beyond environmental/agricultural — does
  the multi-agent supervisor pattern scale more cleanly by just adding
  another `DomainAgent` + one more `consult_*` tool, versus the
  single-agent's filters becoming an ever-larger enum?_
