"""
Step 5 - Conversation logging.

Every turn from either architecture gets appended as one JSON line to
logs/conversation_log.jsonl, in a common schema regardless of which system
answered it -- this is exactly what Step 6's evaluation dashboard reads.

Schema per logged turn:
    {
        "timestamp": ISO-8601 string,
        "system": "single_agent" | "multi_agent",
        "question": str,
        "answer": str,
        "sources": [case_number, ...],       # unchanged: plain case-number
                                              # strings, exactly as before --
                                              # metrics.py's citation_consistency
                                              # does a set() over this field
                                              # and expects strings, not dicts.
        "source_details": [                  # NEW -- guide Sec. 5 asks the
            {                                # log to store "key metadata
                "case_number": str,          # such as case number, legal
                "legal_domain": str | None,  # domain, thematic area,
                "thematic_area": str | None, # instrument type, and source"
                "eu_instruments": list | None,   # for every retrieved
                "member_state": str | None,      # document -- not just the
                "source": str | None,            # bare case number.
            }, ...
        ],
        "consulted_domains": [...] | null,   # multi_agent only
        "trace": [...],                      # full step-by-step trace
        "latency_seconds": float,
    }
"""
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "logs" / "conversation_log.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "json" / "manifest.json"

_manifest_lookup = None


def _load_manifest_lookup() -> dict:
    global _manifest_lookup
    if _manifest_lookup is None:
        _manifest_lookup = {}
        if MANIFEST_PATH.exists():
            entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            for entry in entries:
                # joined-case judgments are keyed under every individual
                # case number they cover, same alias pattern as
                # single_agent/tools.py's _by_case_number and the
                # Knowledge Graph's alias index (build_graph.py).
                for num in entry.get("case_numbers") or [entry.get("case_number")]:
                    if num:
                        _manifest_lookup[num] = entry
    return _manifest_lookup


def get_source_details(case_numbers: list) -> list:
    """Look up each retrieved case number's metadata for logging/display.
    Kept as a separate field from `sources` (rather than replacing it)
    because metrics.py's citation_consistency() does `set(sources or [])`
    and intersects it against case numbers regexed out of the answer text
    -- changing `sources` itself to a list of dicts would silently break
    every RAGAS-proxy computation and the chat UI's ", ".join(sources)."""
    lookup = _load_manifest_lookup()
    details = []
    for num in case_numbers or []:
        entry = lookup.get(num)
        details.append({
            "case_number": num,
            "legal_domain": entry.get("legal_domain") if entry else None,
            "thematic_area": entry.get("thematic_area") if entry else None,
            "eu_instruments": entry.get("eu_instruments") if entry else None,
            "member_state": entry.get("member_state") if entry else None,
            "source": entry.get("source_file") if entry else None,
        })
    return details


def log_turn(system: str, question: str, result: dict, latency_seconds: float, log_path: Path = LOG_PATH):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sources = result.get("sources", [])
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": system,
        "question": question,
        "answer": result.get("answer"),
        "sources": sources,
        "source_details": get_source_details(sources),
        "consulted_domains": result.get("consulted_domains"),
        "trace": result.get("trace"),
        "latency_seconds": round(latency_seconds, 2),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_logs(log_path: Path = LOG_PATH) -> list:
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
