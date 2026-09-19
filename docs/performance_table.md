# Comparative performance table

Per the exam guide (Sec. 7.2), this table reports RAGAS metrics for the two
architectures. **Internal numbers below are real**, from running
`batch_run.py` for both systems against the guide's 20 example questions on
2026-07-23 (`results_exam_june_2026/results_single_agent.xlsx` and
`results_multi_agent.xlsx`) and reading the aggregate comparison off
`src/evaluation/dashboard.py`.

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
| Context precision* (proxy: context utilization — fraction of retrieved case numbers actually cited) | 0.740 | 0.716 | Proxy only |
| Context recall | not computable without reference answers | not computable without reference answers | Needs ground truth |
| Faithfulness* (proxy: citation consistency — fraction of cited case numbers that were genuinely retrieved) | 0.974 | 0.958 | Proxy only; optional LLM-judge score available via `metrics.faithfulness_llm_judge` |
| Answer relevancy (embedding similarity, question vs. answer) | 0.803 | 0.785 | |
| Answer correctness | not computable without reference answers | not computable without reference answers | Needs ground truth |
| Avg. latency (s) | 44.9 | 27.3 | **Confounded** — see note below; both include Gemini free-tier rate-limit wait time, and the single-agent run includes one outlier question (Q18, "what is the latest judgment") whose retrieval strategy returned a very large source list |
| Abstention rate (Q20-style questions) | 0.20 (4/20) | 0.10 (2/20) | Detected via phrase matching (`looks_like_abstention`), not manually verified per-question |

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
assumed. To produce this: build the graph once (`python
src/knowledge_graph/build_graph.py`), then, from `src/evaluation/`, run
`python run_kg_ablation.py` (add `--systems single_agent` to run just one
architecture instead of both). The script runs each system once with
`DISABLE_KG=1` (the `expand_via_graph` tool withheld) and once with the KG
available, computes the proxy metrics for each, and writes the results
directly into this table's section below.

| Metric | Single-agent, no KG | Single-agent, with KG | Multi-agent, no KG | Multi-agent, with KG |
|---|---|---|---|---|
| Context precision* | 0.753 | 0.767 | _not run_ | _not run_ |
| Faithfulness* | 1.0 | 1.0 | _not run_ | _not run_ |
| Answer relevancy | 0.782 | 0.803 | _not run_ | _not run_ |
| Answer correctness | not computable without reference | not computable without reference | not computable without reference | not computable without reference |
| Avg. latency (s) | 9.051 | 9.982 | _not run_ | _not run_ (expect higher -- extra tool calls) |

## Discussion

- **Both systems score similarly on faithfulness/relevancy proxies** —
  single-agent slightly ahead on citation consistency (0.974 vs 0.958) and
  answer relevancy (0.803 vs 0.785), but the gap is small enough that I
  wouldn't call one architecture clearly more faithful than the other on
  this sample.
- **Average latency looks like a big single-agent penalty (44.9s vs
  27.3s) but this number is confounded and shouldn't be read as "multi-
  agent is faster."** Both runs hit Gemini's free-tier rate limit multiple
  times, and the retry waits (tens of seconds each) are counted inside the
  logged latency — which questions happened to trigger a rate-limit pause
  was essentially down to timing, not architecture. On top of that, the
  single-agent run has one clear outlier: Q18 ("what is the latest
  judgment in this collection") took 660s because the agent called
  `list_cases_by_filter` with no filters and got back nearly the entire
  279-judgment manifest as "sources," which also explains single-agent's
  much higher `avg_num_sources` (15.75 vs multi-agent's 2.4) — that's not
  15 genuinely relevant documents per question on average, it's this one
  question dragging the average up. **Before citing the latency numbers in
  the final report, I should exclude Q18 or note it explicitly, and ideally
  re-run outside of a rate-limited period to get a cleaner comparison.**
- **Abstention rate**: single-agent abstained (or gave a hedged
  "not addressed by the corpus"-style answer) on 4/20 questions, multi-
  agent on 2/20. This is measured by simple phrase matching, not manually
  checked — worth spot-checking a few of these by hand before trusting the
  number, since a false negative (an answer that hedges without using one
  of the listed phrases) or false positive (a legitimately partial answer
  mis-flagged as an abstention) are both plausible with a phrase-matching
  approach.
- **Scaling assessment**: not something this internal run can answer
  directly — it only exercises two domains. The structural argument still
  stands from the design: the multi-agent supervisor pattern scales by
  adding one `DomainAgent` + one `consult_*` tool per new domain, versus
  the single agent's filter parameters becoming an ever-larger enum. The
  real test would be adding a third domain and observing whether the
  supervisor's routing logic degrades.
- **What I'd actually want before the final submission**: a rerun outside
  a period where the free-tier rate limit is being hit repeatedly (e.g.
  spread across a longer window, or with a paid tier), and either
  excluding or specifically discussing the Q18 outlier rather than letting
  it dominate the latency/source-count averages.
