"""
Bonus - Knowledge Graph construction.

Builds an in-memory graph (NetworkX -- simpler infra than a dedicated graph
DB, appropriate for a student project; swap in Neo4j later by walking this
same node/edge list if desired) over the 279-judgment corpus, following the
schema suggested in the exam guide Sec. 8:

Node types:
    Judgment(case_number), Instrument(name), LegalDomain(name),
    ThematicArea(name), MemberState(name), Chamber(name),
    LegalPrinciple(name)
    [Provision is approximated -- see extract_provisions below]

Edge types:
    cites(Judgment->Judgment), interprets(Judgment->Provision),
    belongs_to(Provision->Instrument), concerns(Judgment->ThematicArea),
    in_domain(Judgment->LegalDomain), referred_by(Judgment->MemberState),
    decided_by(Judgment->Chamber), invokes(Judgment->LegalPrinciple)

Construction is entirely rule-based (regex over the already-parsed JSON from
Step 1), per the guide's "rule-based patterns" option -- chosen over an
LLM-extraction prompt to avoid ~279 extra API calls for a bonus feature.

Usage:
    python build_graph.py    # writes graph.gpickle + graph_summary.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_ROOT = REPO_ROOT / "data" / "json"
OUT_DIR = Path(__file__).resolve().parent

CASE_NUMBER_RE = re.compile(r"C[\-\u2010\u2011\u2012\u2013\u2014]\d+/\d+")
PROVISION_RE = re.compile(
    r"Article\s+(\d+[a-z]?(?:\(\d+\))?(?:\([a-z]\))?)\s+of\s+"
    r"(?:the\s+)?((?:Council\s+)?(?:Directive|Regulation)\s*\(?[A-Z]{0,3}\)?\s*"
    r"(?:No\s*)?[\d/]+(?:/(?:EEC|EC|EU))?)",
    re.IGNORECASE,
)
PRINCIPLES = [
    "precautionary principle", "polluter-pays principle", "polluter pays principle",
    "principle of proportionality", "principle of effectiveness", "principle of legal certainty",
    "principle of equivalence", "principle of non-discrimination", "principle of sincere cooperation",
    "principle of subsidiarity", "principle of equal treatment", "principle of legitimate expectations",
    "protection of legitimate expectations", "principle of sound financial management",
    "principle of effective judicial protection", "principle of sustainable development",
]


def normalize_principle(p: str) -> str:
    p = p.lower().strip()
    if "polluter" in p:
        return "polluter-pays principle"
    if "legitimate expectations" in p:
        return "principle of legitimate expectations"
    return p


def extract_citations(text: str, own_case_numbers: set) -> set:
    """Other case numbers mentioned in this judgment's text (naive citation
    extraction -- does not distinguish 'the Court held in Case C-x/y that...'
    from an unrelated mention, but is a reasonable rule-based first pass)."""
    found = {re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", c) for c in CASE_NUMBER_RE.findall(text)}
    return found - own_case_numbers


def extract_provisions(text: str) -> list:
    """('Article 6(3)', 'Directive 92/43/EEC') style pairs. Approximates the
    guide's Provision node (a specific article of a Directive/Regulation) --
    granularity is 'article mentioned near an instrument name in this
    judgment', not a verified canonical provision registry."""
    return [(m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()) for m in PROVISION_RE.finditer(text)]


def extract_principles(text: str) -> set:
    text_lower = text.lower()
    return {normalize_principle(p) for p in PRINCIPLES if p in text_lower}


def build_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    docs = {}
    for sub in ("agricultural", "environmental"):
        for f in (JSON_ROOT / sub).glob("*.json"):
            doc = json.loads(f.read_text(encoding="utf-8"))
            case_number = doc["metadata"].get("case_number")
            if case_number:
                docs[case_number] = doc

    all_case_numbers = set(docs.keys())

    for case_number, doc in docs.items():
        meta = doc["metadata"]
        G.add_node(case_number, type="Judgment",
                    date=meta.get("date_of_judgment"), celex=meta.get("celex_number"))

        domain = meta.get("legal_domain")
        if domain and domain != "both":
            G.add_node(domain, type="LegalDomain")
            G.add_edge(case_number, domain, relation="in_domain")
        elif domain == "both":
            for d in ("agricultural", "environmental"):
                G.add_node(d, type="LegalDomain")
                G.add_edge(case_number, d, relation="in_domain")

        for area in meta.get("thematic_area") or []:
            G.add_node(area, type="ThematicArea")
            G.add_edge(case_number, area, relation="concerns")

        state = meta.get("member_state")
        if state:
            G.add_node(state, type="MemberState")
            G.add_edge(case_number, state, relation="referred_by")

        chamber = meta.get("chamber")
        if chamber:
            G.add_node(chamber, type="Chamber")
            G.add_edge(case_number, chamber, relation="decided_by")

        for instrument in meta.get("eu_instruments") or []:
            G.add_node(instrument, type="Instrument")
            G.add_edge(case_number, instrument, relation="interprets_instrument")

        for article, instrument in extract_provisions(doc["text"]):
            provision_id = f"{instrument} Art.{article}"
            G.add_node(provision_id, type="Provision", article=article, instrument=instrument)
            G.add_node(instrument, type="Instrument")
            G.add_edge(provision_id, instrument, relation="belongs_to")
            G.add_edge(case_number, provision_id, relation="interprets")

        for principle in extract_principles(doc["text"]):
            G.add_node(principle, type="LegalPrinciple")
            G.add_edge(case_number, principle, relation="invokes")

        for cited in extract_citations(doc["text"], {case_number}):
            if cited in all_case_numbers:  # only link citations we can resolve within our corpus
                G.add_node(cited, type="Judgment")
                G.add_edge(case_number, cited, relation="cites")

    return G


def summarize(G: nx.MultiDiGraph) -> dict:
    node_types = Counter(d.get("type", "?") for _, d in G.nodes(data=True))
    edge_types = Counter(d.get("relation", "?") for _, _, d in G.edges(data=True))
    return {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "node_types": dict(node_types), "edge_types": dict(edge_types),
    }


if __name__ == "__main__":
    G = build_graph()
    summary = summarize(G)
    print(json.dumps(summary, indent=2))

    nx.write_gpickle(G, OUT_DIR / "graph.gpickle") if hasattr(nx, "write_gpickle") else \
        json.dump(nx.node_link_data(G), (OUT_DIR / "graph.json").open("w"), default=str)
    (OUT_DIR / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nGraph saved to {OUT_DIR}")
