"""
Step 2b - Embeddings & vector database.

Reads data/embeddings/chunks.jsonl (produced by chunker.py), computes sentence
embeddings, and builds FAISS indices:
  - one GLOBAL index over the entire corpus (used by the single-agent system,
    which must be able to reach all 279 judgments from one place), and
  - one PER-DOMAIN index for each of {agricultural, environmental} (used by
    the multi-agent system's specialized agents; cross-domain judgments,
    legal_domain == "both", are included in both).

Model choice: 'sentence-transformers/all-MiniLM-L6-v2' by default -- small,
fast, strong general-purpose semantic search performance, good default for a
corpus whose *indexed text* is in English (the EN CELEX version of each
judgment), even though the underlying cases originate from many Member
States/languages. If you expect users to type queries in other languages and
want cross-lingual matching without translation, swap MODEL_NAME for a
multilingual model such as 'paraphrase-multilingual-mpnet-base-v2' (slower,
larger, ~50 languages) -- see docs/architecture.md "Models used" section.

Usage:
    python src/embeddings/build_index.py
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_chunks(chunks_path: Path):
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_chunks(chunks, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(embeddings)
    return index


def save_index(index, chunks, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / f"{name}.faiss"))
    # chunk metadata, in the same order as vectors were added -> position i
    # in the FAISS index corresponds to chunk_meta[i]
    meta = [
        {k: v for k, v in c.items() if k != "text"} | {"text_preview": c["text"][:300]}
        for c in chunks
    ]
    (out_dir / f"{name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    base = Path(__file__).resolve().parents[2]  # repo root
    chunks_path = base / "data" / "embeddings" / "chunks.jsonl"
    index_dir = base / "src" / "embeddings" / "index"

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks.")

    embeddings = embed_chunks(chunks)
    print(f"Computed embeddings: {embeddings.shape}")

    # --- Global index (single-agent system) ---
    global_index = build_faiss_index(embeddings)
    save_index(global_index, chunks, index_dir, "global")
    print(f"Saved global index ({global_index.ntotal} vectors) -> {index_dir/'global.faiss'}")

    # --- Per-domain indices (multi-agent system's specialized agents) ---
    for domain in ("agricultural", "environmental"):
        idxs = [i for i, c in enumerate(chunks) if c["legal_domain"] in (domain, "both")]
        if not idxs:
            continue
        sub_embeddings = embeddings[idxs]
        sub_chunks = [chunks[i] for i in idxs]
        sub_index = build_faiss_index(sub_embeddings)
        save_index(sub_index, sub_chunks, index_dir, domain)
        print(f"Saved {domain} index ({sub_index.ntotal} vectors) -> {index_dir/(domain + '.faiss')}")


if __name__ == "__main__":
    main()
