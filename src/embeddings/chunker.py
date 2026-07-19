"""
Step 2a - Chunking.

Turns each data/json/{domain}/*.json judgment into a handful of retrieval
chunks:
  - one "header" chunk (keywords + case identity) so subject-matter/instrument
    terms are directly searchable even when a question uses different wording
    than the body text,
  - several "body" chunks: paragraphs greedily merged up to ~TARGET_CHARS,
    never splitting a paragraph across chunks (unless a single paragraph
    itself exceeds the target, in which case it is split on sentence
    boundaries),
  - one "operative_part" chunk with the numbered rulings, since many
    questions ("what did the Court rule") are best answered from there.

Each chunk carries the judgment's full metadata plus its own chunk_id and
chunk_type, so retrieval can filter/boost by legal_domain, thematic_area,
eu_instruments, member_state, etc. without a second lookup.

Output: data/embeddings/chunks.jsonl (one JSON object per line).
"""
import json
import re
from pathlib import Path

TARGET_CHARS = 1000


def split_long_paragraph(text, limit=TARGET_CHARS):
    sentences = re.split(r"(?<=[.;])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > limit and current:
            chunks.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def make_body_chunks(paragraphs, target_chars=TARGET_CHARS):
    """Greedily merge consecutive paragraphs up to target_chars."""
    chunks = []  # list of (text, first_para_idx, last_para_idx)
    current_text, start_idx = "", None
    for i, p in enumerate(paragraphs):
        if len(p) > target_chars * 1.5:
            if current_text:
                chunks.append((current_text, start_idx, i - 1))
                current_text, start_idx = "", None
            for piece in split_long_paragraph(p, target_chars):
                chunks.append((piece, i, i))
            continue
        if not current_text:
            current_text, start_idx = p, i
        elif len(current_text) + len(p) + 1 <= target_chars:
            current_text += " " + p
        else:
            chunks.append((current_text, start_idx, i - 1))
            current_text, start_idx = p, i
    if current_text:
        chunks.append((current_text, start_idx, len(paragraphs) - 1))
    return chunks


def chunk_document(doc, json_file_name):
    meta = doc["metadata"]
    base_meta = {
        "case_number": meta.get("case_number"),
        "legal_domain": meta.get("legal_domain"),
        "thematic_area": meta.get("thematic_area"),
        "eu_instruments": meta.get("eu_instruments"),
        "member_state": meta.get("member_state"),
        "chamber": meta.get("chamber"),
        "date_of_judgment": meta.get("date_of_judgment"),
        "language_of_the_case": meta.get("language_of_the_case"),
        "source_json": json_file_name,
    }

    chunks = []
    case_id = meta.get("case_number") or json_file_name

    # 1. Header / keywords chunk
    header_text_parts = [p for p in [meta.get("keywords"), meta.get("referring_court")] if p]
    if header_text_parts:
        chunks.append({
            "chunk_id": f"{case_id}::header",
            "chunk_type": "header",
            "text": " | ".join(header_text_parts),
            **base_meta,
        })

    # 2. Body chunks from paragraphs
    for j, (text, start_p, end_p) in enumerate(make_body_chunks(doc.get("paragraphs", []))):
        chunks.append({
            "chunk_id": f"{case_id}::body::{j}",
            "chunk_type": "body",
            "paragraph_range": [start_p, end_p],
            "text": text,
            **base_meta,
        })

    # 3. Operative part chunk
    if doc.get("operative_part"):
        op_text = " ".join(doc["operative_part"])
        chunks.append({
            "chunk_id": f"{case_id}::operative",
            "chunk_type": "operative_part",
            "text": op_text,
            **base_meta,
        })

    return chunks


def chunk_corpus(json_root: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs, n_chunks = 0, 0
    with out_path.open("w", encoding="utf-8") as out:
        for sub in ("agricultural", "environmental"):
            for f in sorted((json_root / sub).glob("*.json")):
                doc = json.loads(f.read_text(encoding="utf-8"))
                for chunk in chunk_document(doc, f.name):
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_chunks += 1
                n_docs += 1
    return n_docs, n_chunks


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]  # repo root
    n_docs, n_chunks = chunk_corpus(base / "data" / "json", base / "data" / "embeddings" / "chunks.jsonl")
    print(f"Chunked {n_docs} judgments into {n_chunks} chunks "
          f"({n_chunks/n_docs:.1f} chunks/judgment on average).")
