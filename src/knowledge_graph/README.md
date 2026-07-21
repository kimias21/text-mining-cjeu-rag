# Bonus — Knowledge-Graph-augmented generation (Sec. 8)

Moves from pure vector-based RAG towards a lightweight GraphRAG: alongside
semantic search, both agent systems can traverse an explicit graph of how
judgments, provisions, instruments, and legal principles relate to each
other — including bridges between the environmental and agricultural
corpora that vector similarity alone won't surface.

## Schema

Built exactly to the guide's suggested schema (Sec. 8):

**Nodes:** `Judgment` (case number), `Provision` (an article of a
Directive/Regulation, approximated — see caveat below), `Instrument`
(the Directive/Regulation as a whole), `LegalDomain`, `ThematicArea`,
`MemberState`, `Chamber`, `LegalPrinciple`.

**Edges:** `cites` (Judgment→Judgment), `interprets` (Judgment→Provision),
`belongs_to` (Provision→Instrument), `concerns` (Judgment→ThematicArea),
`in_domain` (Judgment→LegalDomain), `referred_by` (Judgment→MemberState),
`decided_by` (Judgment→Chamber), `invokes` (Judgment→LegalPrinciple).

## Construction

`build_graph.py` populates the graph **entirely with rule-based
extraction** (regex over the already-parsed JSON from Step 1) rather than
an LLM-extraction prompt, to avoid ~279 extra API calls for a bonus
feature:
- Domain/thematic area/Member State/Chamber/instruments: straight from
  Step 1's metadata fields.
- Citations (`cites`): regex-matches other case numbers in each
  judgment's text (note: CJEU documents use a Unicode hyphen `‐` U+2010 in
  citations like "Case C‐115/17", not a plain ASCII hyphen — the regex
  accounts for this).
- Provisions (`interprets`/`belongs_to`): regex-matches "Article N of
  [Directive/Regulation ...]" patterns co-occurring in the text. **Caveat:**
  this is "article mentioned near an instrument name in this judgment," not
  a verified canonical provision registry — treat `Provision` nodes as a
  approximation, not ground truth.
- Legal principles (`invokes`): keyword matching against a fixed list
  (precautionary principle, polluter-pays principle, proportionality,
  effectiveness, legal certainty, equivalence, non-discrimination, sincere
  cooperation, subsidiarity, equal treatment, legitimate expectations,
  sound financial management, effective judicial protection, sustainable
  development), calibrated by sampling the corpus for actual "principle of
  X" phrasings before finalizing the list.

Stored in-memory with NetworkX (`MultiDiGraph`) rather than a dedicated
graph database — appropriate for a 279-judgment, ~2,900-node graph and
avoids Neo4j infrastructure for a bonus feature. Swap in Neo4j later by
walking the same node/edge lists if needed.

Regenerate with:
```bash
python build_graph.py
```
(Not committed to the repo — it's fully regeneratable from `data/json/`, like the FAISS indices.)

## Use at generation time

Implemented as **(i) a retrieval expander**: `graph_tools.py`'s
`expand_via_graph(case_number, relation=None)` is exposed as a 4th tool
(`expand_via_graph`) to both the single agent and each multi-agent
specialist — given a case already found relevant, it returns judgments the
graph says are related (cited by / citing it, sharing a provision, sharing
a legal principle), **deliberately not domain-restricted even inside a
domain specialist**, since the whole point is surfacing the
environmental↔agricultural bridge a domain-scoped vector search would miss.

The other two use-cases the guide suggests — (ii) serializing a subgraph
directly into the prompt as structured context, and (iii) using the graph
as a supervisor routing aid — are natural next extensions of
`graph_tools.py` but are not implemented here; `expand_via_graph` as an
agent-callable tool was prioritized since it required no changes to the
existing, tested agent loop code (both `agent.py` and `domain_agent.py`
detect the graph's availability and add the tool automatically — the
systems still run fine without a graph built at all).

## Reporting requirement

The guide asks for RAGAS metrics **with and without** the Knowledge Graph
in the comparative performance table — see `docs/performance_table.md`'s
KG row. To produce that comparison: run `batch_run.py` once with the graph
absent/unbuilt (baseline — `expand_via_graph` just won't be offered as a
tool) and once with it built, and compare the dashboard's aggregate metrics
between the two runs.
