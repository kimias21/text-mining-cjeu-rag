# Task B — Multi-agent system with supervisor (Step 4)

- `domain_agent.py` — `DomainAgent(domain)`: same ReAct loop as Task A's
  `SingleAgent`, but every tool call is pinned to one legal domain
  (`agricultural` or `environmental`). Uses the domain-specific FAISS index
  built in Step 2, which already includes the 6 cross-domain judgments, so
  neither specialist misses them.
- `../knowledge_graph/graph_tools.py` (bonus, optional) — both DomainAgents auto-pick up a 4th `expand_via_graph` tool if the Knowledge Graph has been built, deliberately not domain-restricted so it can surface the environmental↔agricultural bridge a domain-scoped search would miss; see `src/knowledge_graph/README.md`.
- `supervisor.py` — `Supervisor`: has no direct corpus access. Its only
  tools are `consult_agricultural_agent` / `consult_environmental_agent`,
  each of which runs a full `DomainAgent.answer()` internally. The
  supervisor is instructed to consult **both** specialists whenever a
  question is ambiguous or could plausibly touch both domains (e.g.
  pesticide residue limits in organic farming), rather than guessing one.
  Once it has the specialist response(s), it synthesizes one final answer.

Loop shape:

```
question -> [Supervisor: route] -> consult agri? --yes--> [DomainAgent(agricultural).answer()] --\
                                  -> consult env?  --yes--> [DomainAgent(environmental).answer()] --> [Supervisor: synthesize] -> final answer
```

`Supervisor.answer()` returns the same `{"answer", "trace", "sources"}`
contract as `SingleAgent.answer()` (plus `consulted_domains`), with `trace`
additionally nesting each consulted specialist's own step-by-step trace —
so both architectures plug into the same chat-logging (Step 5) and
evaluation (Step 6) code without special-casing.

## Setup

Same as Task A — needs `google-genai` installed and `GEMINI_API_KEY` set,
plus both domain FAISS indices built (Step 2).

## Run

```bash
python supervisor.py "What residue limits apply to pesticides used in organic farming, and are they subject to environmental impact assessment?"
```

This example question is deliberately cross-domain, to exercise the "consult
both specialists" path.
