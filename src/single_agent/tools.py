"""
Step 3 - Tools available to the single ReAct-style agent.

Each tool is a plain Python function plus an OpenAI-style JSON schema
describing it, so the same definitions can be handed straight to
`client.chat.completions.create(tools=...)`.

Tools:
  - search_corpus        semantic search over the FULL corpus (global FAISS
                          index), with optional metadata filters
  - get_case_by_number    exact lookup of one judgment by its case number
  - list_cases_by_filter   pure-metadata browsing (no embedding needed) --
                          e.g. "how many environmental cases from Poland",
                          answered without hitting the vector index at all
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "embeddings"))
from retriever import Retriever  # noqa: E402

# Optional bonus tool (Sec. 8 of the exam guide) -- only wired in if the
# Knowledge Graph has actually been built (src/knowledge_graph/build_graph.py).
# Falls back to no-op cleanly if it hasn't, so this module still imports and
# both agent systems still work without the bonus.
#
# DISABLE_KG=1 forces this off even when the graph *is* built -- this is the
# switch src/evaluation/run_kg_ablation.py flips between runs to produce the
# guide's required "with vs. without the Knowledge Graph" RAGAS comparison
# (Sec. 8): each condition is a separate process, so the tool list an agent
# is built with is fixed for its whole run, never toggled mid-loop.
sys.path.insert(0, str(REPO_ROOT / "src" / "knowledge_graph"))
if os.environ.get("DISABLE_KG"):
    _GRAPH_AVAILABLE = False
else:
    try:
        from graph_tools import expand_via_graph, EXPAND_VIA_GRAPH_SCHEMA
        _GRAPH_AVAILABLE = True
    except Exception:
        _GRAPH_AVAILABLE = False

MANIFEST_PATH = REPO_ROOT / "data" / "json" / "manifest.json"
JSON_ROOT = REPO_ROOT / "data" / "json"

_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
_by_case_number = {}
for entry in _manifest:
    for num in entry.get("case_numbers", [entry.get("case_number")]):
        _by_case_number.setdefault(num, entry)

_retriever_cache = {}


def _get_retriever(index_name="global"):
    if index_name not in _retriever_cache:
        _retriever_cache[index_name] = Retriever(index_name=index_name)
    return _retriever_cache[index_name]


def search_corpus(query: str, k: int = 5, legal_domain: str = None,
                   thematic_area: str = None, eu_instrument: str = None,
                   member_state: str = None) -> list:
    """Semantic search over the judgment corpus. Returns the top-k most
    relevant chunks (with case number, chunk type, preview text, score)."""
    index_name = legal_domain if legal_domain in ("agricultural", "environmental") else "global"
    retriever = _get_retriever(index_name)

    def filter_fn(meta):
        if thematic_area and thematic_area not in (meta.get("thematic_area") or []):
            return False
        if eu_instrument and not any(eu_instrument.lower() in i.lower()
                                      for i in (meta.get("eu_instruments") or [])):
            return False
        if member_state and (meta.get("member_state") or "").lower() != member_state.lower():
            return False
        return True

    needs_filter = thematic_area or eu_instrument or member_state
    return retriever.search(query, k=k, filter_fn=filter_fn if needs_filter else None)


def get_case_by_number(case_number: str) -> dict:
    """Exact lookup of a single judgment's full text, operative part and
    metadata by its case number (e.g. 'C-116/20')."""
    entry = _by_case_number.get(case_number)
    if not entry:
        return {"error": f"No judgment found with case number '{case_number}'."}
    doc = json.loads((JSON_ROOT / entry["json_file"]).read_text(encoding="utf-8"))
    return {
        "metadata": doc["metadata"],
        "operative_part": doc.get("operative_part"),
        "text": doc["text"][:6000],  # cap to keep the tool result a reasonable size
    }


def list_cases_by_filter(legal_domain: str = None, thematic_area: str = None,
                          eu_instrument: str = None, member_state: str = None,
                          limit: int = 20) -> list:
    """Browse judgments by metadata only (no semantic search) -- useful for
    counting/listing questions, e.g. 'which cases came from Poland'."""
    results = []
    for entry in _manifest:
        if legal_domain and entry.get("legal_domain") not in (legal_domain, "both"):
            continue
        if thematic_area and thematic_area not in (entry.get("thematic_area") or []):
            continue
        if eu_instrument and not any(eu_instrument.lower() in i.lower()
                                      for i in (entry.get("eu_instruments") or [])):
            continue
        if member_state and (entry.get("member_state") or "").lower() != member_state.lower():
            continue
        results.append({
            "case_number": entry.get("case_number"),
            "date_of_judgment": entry.get("date_of_judgment"),
            "member_state": entry.get("member_state"),
            "thematic_area": entry.get("thematic_area"),
        })
        if len(results) >= limit:
            break
    return results


TOOLS = {
    "search_corpus": search_corpus,
    "get_case_by_number": get_case_by_number,
    "list_cases_by_filter": list_cases_by_filter,
}
if _GRAPH_AVAILABLE:
    TOOLS["expand_via_graph"] = expand_via_graph

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Semantic search over the CJEU judgment corpus (environmental + agricultural preliminary rulings). Use this to find judgments relevant to a legal question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of what to search for."},
                    "k": {"type": "integer", "description": "Number of results to return (default 5)."},
                    "legal_domain": {"type": "string", "enum": ["agricultural", "environmental"], "description": "Restrict search to one domain, if known."},
                    "thematic_area": {"type": "string", "description": "Restrict to a specific thematic area, if known."},
                    "eu_instrument": {"type": "string", "description": "Restrict to judgments citing a specific Regulation/Directive (partial match), if known."},
                    "member_state": {"type": "string", "description": "Restrict to judgments referred from a specific Member State, if known."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_case_by_number",
            "description": "Look up the full text, operative part (rulings) and metadata of one specific judgment by its case number (e.g. 'C-116/20'). Use this when the question names a specific case, or after search_corpus identifies the most relevant case and you need its full content/ruling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_number": {"type": "string", "description": "e.g. 'C-116/20'"},
                },
                "required": ["case_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cases_by_filter",
            "description": "Browse/count judgments by metadata alone (domain, thematic area, EU instrument, Member State) without semantic search. Use for questions like 'how many cases from Germany' or 'list cases about organic production'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "legal_domain": {"type": "string", "enum": ["agricultural", "environmental"]},
                    "thematic_area": {"type": "string"},
                    "eu_instrument": {"type": "string"},
                    "member_state": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max results (default 20)."},
                },
                "required": [],
            },
        },
    },
]
if _GRAPH_AVAILABLE:
    TOOL_SCHEMAS.append(EXPAND_VIA_GRAPH_SCHEMA)
