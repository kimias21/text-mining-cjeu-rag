"""
Step 2c - Retriever.

Thin query interface over the FAISS indices built by build_index.py.
Both the single-agent (Task A) and each specialized agent in the multi-agent
system (Task B) should call this rather than touching FAISS directly.

Usage:
    from embeddings.retriever import Retriever

    r = Retriever(index_name="global")          # or "agricultural" / "environmental"
    hits = r.search("Can mitigation measures be considered in a Habitats Directive assessment?", k=5)

    # optional metadata post-filtering (e.g. only Regulations, or one thematic area)
    hits = r.search(query, k=5, filter_fn=lambda meta: "Nature & biodiversity" in (meta.get("thematic_area") or []))
"""
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from build_index import MODEL_NAME

INDEX_DIR = Path(__file__).resolve().parent / "index"


class Retriever:
    def __init__(self, index_name="global", index_dir=INDEX_DIR, model_name=MODEL_NAME):
        self.index = faiss.read_index(str(index_dir / f"{index_name}.faiss"))
        self.meta = json.loads((index_dir / f"{index_name}.meta.json").read_text(encoding="utf-8"))
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, k: int = 5, fetch_k: int = None, filter_fn=None):
        """Embed `query`, retrieve top candidates, optionally post-filter by
        metadata (e.g. thematic_area, eu_instruments, member_state), and
        return the top `k` remaining hits ranked by similarity.

        `fetch_k` (defaults to 5x k when filtering) controls how many raw
        ANN hits are pulled before filtering -- oversample so filtering
        doesn't starve the results, per the exam guide's suggestion to
        "retrieve more documents than will ultimately be used as context."
        """
        fetch_k = fetch_k or (k * 5 if filter_fn else k)
        q_vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idxs = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            if filter_fn and not filter_fn(m):
                continue
            results.append({**m, "score": float(score)})
            if len(results) >= k:
                break
        return results


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "young farmer start-up aid combined with small farms aid"
    r = Retriever(index_name="global")
    for hit in r.search(query, k=5):
        print(f"{hit['score']:.3f}  {hit['case_number']}  [{hit['chunk_type']}]  {hit['text_preview'][:120]}")
