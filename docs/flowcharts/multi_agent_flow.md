# Multi-agent (Task B) — flowchart

Mermaid source below renders automatically on GitHub. To export as an image
for the slide deck: paste this block into https://mermaid.live and export
PNG/SVG, or run `npx @mermaid-js/mermaid-cli -i multi_agent_flow.md -o multi_agent_flow.png`
locally (requires Node.js).

```mermaid
flowchart TD
    A[User question] --> B[Supervisor LLM reads\nthe question]
    B --> C{Which domain does\nthis plausibly touch?}

    C -- clearly agricultural only --> D1[consult_agricultural_agent]
    C -- clearly environmental only --> D2[consult_environmental_agent]
    C -- ambiguous / cross-domain\ne.g. organic-farming pesticide\nresidues, CAP + conditionality --> D3[consult BOTH agents]

    D1 --> E1[Agricultural specialist:\nfull ReAct loop against the\nagricultural FAISS index]
    D2 --> E2[Environmental specialist:\nfull ReAct loop against the\nenvironmental FAISS index]
    D3 --> E1
    D3 --> E2

    E1 --> F1[search_corpus / get_case_by_number /\nlist_cases_by_filter, scoped to\nthis agent's domain only]
    E2 --> F2[search_corpus / get_case_by_number /\nlist_cases_by_filter, scoped to\nthis agent's domain only]

    F1 --> G1[Specialist's own grounded\nanswer + cited sources]
    F2 --> G2[Specialist's own grounded\nanswer + cited sources]

    G1 --> H[Supervisor receives\npartial answer-s-\nas tool observation-s-]
    G2 --> H

    H --> I{More specialists\nto consult?}
    I -- yes --> B
    I -- no, have enough --> J[Supervisor synthesizes:\nmerge partial answers,\nnote agreement/differences,\ncite every case number used]

    J --> K[Log turn: question, answer,\nsources, consulted_domains,\nnested trace, latency]
    K --> L[Returned to chat interface]
```

**Where routing happens:** step C — entirely inside the supervisor's own
reasoning (no separate classifier model); it is explicitly instructed to
consult **both** specialists rather than guess a single domain whenever the
question is ambiguous.

**How specialized agents are described/selected:** each `DomainAgent` is
handed to the supervisor only as a callable tool (`consult_agricultural_agent`
/ `consult_environmental_agent`) with a one-line description; the supervisor
never sees the specialists' internal tools or FAISS indices directly — full
separation of concerns.

**How partial answers return and merge:** each specialist runs its *entire*
own retrieval/reasoning loop (steps E→G) before returning one finished,
cited answer plus its `sources` list back to the supervisor as a single tool
observation (step H); the supervisor's final synthesis step (J) is a plain
LLM call over both specialists' text, not a second retrieval pass — the
supervisor itself never touches the corpus.
