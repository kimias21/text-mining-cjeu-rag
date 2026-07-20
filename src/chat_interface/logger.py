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
        "sources": [case_number, ...],
        "consulted_domains": [...] | null,   # multi_agent only
        "trace": [...],                      # full step-by-step trace
        "latency_seconds": float,
    }
"""
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "conversation_log.jsonl"


def log_turn(system: str, question: str, result: dict, latency_seconds: float, log_path: Path = LOG_PATH):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": system,
        "question": question,
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
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
