# Single-agent (Task A) — flowchart

Mermaid source below renders automatically on GitHub. To export as an image
for the slide deck: paste this block into https://mermaid.live and export
PNG/SVG, or run `npx @mermaid-js/mermaid-cli -i single_agent_flow.md -o single_agent_flow.png`
locally (requires Node.js).

```mermaid
flowchart TD
    A[User question] --> B{Does this need\nretrieval, or is it\ngeneric/off-topic?}
    B -- generic, no retrieval needed --> Z1[Answer from model's\ninternal knowledge only\n- exam-setting fallback -]
    B -- yes, legal question --> C[LLM reasons: which tool\nto call next?]

    C --> D{Tool choice}
    D -- search_corpus --> E[Select legal_domain\nenvironmental / agricultural / global]
    E --> E2[Select thematic_area,\neu_instrument, member_state\nfilters, if identifiable]
    E2 --> F[Embed query -\nsame model as corpus -]
    F --> G[FAISS similarity search\non selected index\nover-fetch fetch_k, then\nmetadata-filter down to k]
    G --> H[Observation: retrieved\nchunks + case numbers\nreturned to LLM]

    D -- get_case_by_number --> I[Exact lookup: full text +\noperative part + metadata\nfor one case]
    I --> H

    D -- list_cases_by_filter --> J[Pure metadata browse/count,\nno embedding call]
    J --> H

    H --> C
    C -- enough evidence --> K[Final answer: synthesize\nretrieved context + question,\ncite case numbers explicitly]
    K --> L[Log turn: question, answer,\nsources, full trace, latency]
    Z1 --> L
    L --> M[Returned to chat interface]
```

**Where routing happens:** step E/E2 (domain, thematic area, instrument,
Member State selection) and step G (FAISS index choice + metadata filter +
re-ranking by cosine similarity). The LLM decides *which* filters to apply
based on its own reading of the question — there is no separate classifier
model.

**Where context is constructed:** step H accumulates every tool
observation across the loop into the running conversation; step K is where
the LLM combines all accumulated observations with the original question to
produce the final, cited answer.
