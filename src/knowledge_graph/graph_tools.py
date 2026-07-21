"""
Bonus - Knowledge Graph traversal, used as (i) a retrieval expander: given a
case already found by vector search, what else does the graph say is
relevant (cited/citing judgments, judgments interpreting the same
provision, judgments invoking the same legal principle) -- including,
notably, judgments in the OTHER legal domain, which is exactly the
environmental-agricultural bridge the guide highlights.

Exposes `expand_via_graph` as both a plain function and an agent tool
schema, in the same shape as single_agent/tools.py's other tools, so it can
be added to either agent's toolset without changing the agent loop code.
"""
import json
from pathlib import Path

import networkx as nx

GRAPH_DIR = Path(__file__).resolve().parent
_graph = None


def load_graph():
    global _graph
    if _graph is None:
        gpickle = GRAPH_DIR / "graph.gpickle"
        gjson = GRAPH_DIR / "graph.json"
        if gpickle.exists() and hasattr(nx, "read_gpickle"):
            _graph = nx.read_gpickle(gpickle)
        elif gjson.exists():
            _graph = nx.node_link_graph(json.loads(gjson.read_text(encoding="utf-8")))
        else:
            raise FileNotFoundError(
                "No graph found -- run `python build_graph.py` in src/knowledge_graph/ first."
            )
    return _graph


def expand_via_graph(case_number: str, relation: str = None, max_results: int = 10) -> list:
    """Given a case number, return related judgments via the Knowledge
    Graph: judgments it cites, judgments citing it, and judgments sharing a
    provision or legal principle with it (which can cross the
    environmental/agricultural domain boundary). Optionally restrict to one
    `relation` ("cites", "cited_by", "shares_provision", "shares_principle").
    """
    G = load_graph()
    if case_number not in G:
        return [{"error": f"'{case_number}' not found in the knowledge graph."}]

    results = []

    def add(other, rel, via=None):
        if other == case_number or G.nodes.get(other, {}).get("type") != "Judgment":
            return
        entry = {"case_number": other, "relation": rel}
        if via:
            entry["via"] = via
        results.append(entry)

    if relation in (None, "cites"):
        for _, tgt, d in G.out_edges(case_number, data=True):
            if d.get("relation") == "cites":
                add(tgt, "cites")

    if relation in (None, "cited_by"):
        for src, _, d in G.in_edges(case_number, data=True):
            if d.get("relation") == "cites":
                add(src, "cited_by")

    if relation in (None, "shares_provision"):
        for _, provision, d in G.out_edges(case_number, data=True):
            if d.get("relation") == "interprets":
                for other, _, d2 in G.in_edges(provision, data=True):
                    if d2.get("relation") == "interprets":
                        add(other, "shares_provision", via=provision)

    if relation in (None, "shares_principle"):
        for _, principle, d in G.out_edges(case_number, data=True):
            if d.get("relation") == "invokes":
                for other, _, d2 in G.in_edges(principle, data=True):
                    if d2.get("relation") == "invokes":
                        add(other, "shares_principle", via=principle)

    # de-dupe, keep first relation found per case, cap results
    seen, deduped = set(), []
    for r in results:
        if r["case_number"] not in seen:
            seen.add(r["case_number"])
            deduped.append(r)
    return deduped[:max_results]


EXPAND_VIA_GRAPH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "expand_via_graph",
        "description": (
            "Given a case number already found relevant, use the Knowledge Graph to find related "
            "judgments: cases it cites, cases that cite it, cases interpreting the same provision, "
            "or cases invoking the same legal principle (e.g. precautionary principle, proportionality) "
            "-- this can surface relevant cases in the OTHER legal domain that vector search alone might miss."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "case_number": {"type": "string", "description": "e.g. 'C-116/20'"},
                "relation": {
                    "type": "string",
                    "enum": ["cites", "cited_by", "shares_provision", "shares_principle"],
                    "description": "Restrict to one relation type. Omit to check all types.",
                },
                "max_results": {"type": "integer", "description": "Default 10."},
            },
            "required": ["case_number"],
        },
    },
}


if __name__ == "__main__":
    import sys
    case = sys.argv[1] if len(sys.argv) > 1 else "C-411/17"
    for r in expand_via_graph(case):
        print(r)
