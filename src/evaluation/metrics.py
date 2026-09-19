"""
Step 6 - RAGAS-inspired metrics computable directly from conversation logs,
without needing the hidden ground-truth reference answers (those only
arrive with the official evaluation -- see exam guide Sec. 6). These give a
real, if partial, read on system quality for the internal comparison table
the guide suggests (Sec. 7.2):

  - citation_consistency  -- proxy for FAITHFULNESS: what fraction of case
    numbers the answer actually cites also appear in `sources` (the
    documents genuinely retrieved)? A citation not in `sources` is either a
    hallucinated case number or one recalled from the model's own training
    data rather than from retrieval -- both are faithfulness violations.
  - answer_relevancy      -- cosine similarity between the question and the
    answer's embeddings (same embedding model as retrieval), following
    RAGAS's own approach of using semantic similarity as a cheap proxy for
    "does the answer address the question."
  - context_utilization   -- proxy for CONTEXT PRECISION: what fraction of
    distinct case numbers that appeared anywhere in the trace's retrieval
    steps ended up actually cited in the final answer? Low utilization
    suggests the retrieval step pulled in irrelevant documents relative to
    what the model actually used.
  - abstained             -- whether the answer contains hedge/refusal
    language ("does not contain", "cannot find", "no clear answer", ...),
    relevant for scoring Q20-style abstention tests.

CONTEXT RECALL and ANSWER CORRECTNESS need a reference answer and are NOT
computed here -- see the guide Sec. 6: only the official hidden evaluation
(or a self-authored small test set with reference answers, which you can
pass into `answer_correctness_llm_judge`) can produce those.
"""
import os
import re
from pathlib import Path

CASE_NUMBER_RE = re.compile(r"C-\d+/\d+")
ABSTENTION_PHRASES = [
    "does not contain", "cannot find", "no clear answer", "not contain a",
    "no ruling", "not addressed", "unable to find", "could not find",
    "does not provide", "no judgment", "not available in the corpus",
]

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embeddings"))
        from build_index import MODEL_NAME
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def citation_consistency(answer: str, sources: list) -> float | None:
    """Fraction of case numbers cited in the answer text that are also in
    `sources`. Returns None if the answer cites no case numbers at all
    (undefined rather than 0, since 0 would wrongly read as "fully
    inconsistent")."""
    cited = set(CASE_NUMBER_RE.findall(answer or ""))
    if not cited:
        return None
    sources_set = set(sources or [])
    return len(cited & sources_set) / len(cited)


def context_utilization(answer: str, trace: list) -> float | None:
    """Fraction of case numbers seen anywhere in the trace's retrieval
    steps that are actually cited in the final answer text."""
    from batch_run import format_context  # reuse the same trace-scanning logic
    retrieved = set(c for c in format_context(trace).split("; ") if c)
    if not retrieved:
        return None
    cited = set(CASE_NUMBER_RE.findall(answer or ""))
    return len(retrieved & cited) / len(retrieved)


def answer_relevancy(question: str, answer: str) -> float | None:
    """Cosine similarity between question and answer embeddings. Returns
    None (undefined, not 0) when there's no answer to embed -- e.g. a turn
    whose agent call itself failed and was logged with answer=None (see the
    per-question error handling in batch_run.py / run_kg_ablation.py). Bug
    fix 2026-09-19: this used to pass None straight to the embedder, which
    crashed with 'Unsupported input type: NoneType' and took down the
    whole aggregation -- including every already-succeeded question in the
    same run -- the moment a single question failed."""
    if not answer:
        return None
    model = _get_embedder()
    import numpy as np
    vecs = model.encode([question, answer], normalize_embeddings=True, convert_to_numpy=True)
    return float(np.dot(vecs[0], vecs[1]))


def looks_like_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return any(phrase in a for phrase in ABSTENTION_PHRASES)


def compute_all(entry: dict) -> dict:
    """Compute every ground-truth-free metric for one logged turn.

    A turn whose agent call itself failed (answer is None -- the
    per-question error handling in batch_run.py / run_kg_ablation.py logs
    exactly this shape) is flagged via "failed": True, and its quality
    metrics come back as None/False rather than being computed from a
    missing answer: a technical failure (bad API key, rate limit, ...)
    says nothing about answer quality, and averaging it in as if it were a
    real "no" would understate whatever the system actually did on the
    questions it could answer."""
    answer, sources, trace = entry.get("answer"), entry.get("sources", []), entry.get("trace", [])
    failed = answer is None
    return {
        "citation_consistency": citation_consistency(answer, sources),
        "context_utilization": context_utilization(answer, trace),
        "answer_relevancy": answer_relevancy(entry.get("question", ""), answer),
        "abstained": False if failed else looks_like_abstention(answer),
        "num_sources": len(sources or []),
        "latency_seconds": entry.get("latency_seconds"),
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Optional: LLM-as-judge faithfulness (costs one API call per turn -- not run
# automatically; call explicitly, e.g. from the dashboard's "Run LLM-judge"
# button, or answer_correctness_llm_judge if you've authored reference
# answers for a small internal test set).
# ---------------------------------------------------------------------------

JUDGE_FAITHFULNESS_PROMPT = """You are grading a RAG system's answer for FAITHFULNESS: \
does every factual claim in the answer follow from the provided context (retrieved case \
excerpts), without adding unsupported information?

Question: {question}

Context (retrieved case excerpts, may be partial):
{context}

Answer to grade:
{answer}

Respond with ONLY a single integer from 1 (answer makes claims unsupported by the context, \
or contradicts it) to 5 (every claim is directly supported by the context). No explanation."""

JUDGE_CORRECTNESS_PROMPT = """You are grading a RAG system's answer for CORRECTNESS against a \
reference (ground-truth) answer.

Question: {question}

Reference answer: {reference}

Answer to grade: {answer}

Respond with ONLY a single integer from 1 (contradicts or misses the reference answer's key \
points) to 5 (matches the reference answer's key points). No explanation."""


def _judge_score(prompt: str, client=None, model_name="gemini-3.1-flash-lite") -> int:
    from google import genai
    client = client or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(model=model_name, contents=prompt)
    match = re.search(r"[1-5]", response.text or "")
    return int(match.group()) if match else None


def faithfulness_llm_judge(question: str, answer: str, context: str, client=None) -> int:
    return _judge_score(
        JUDGE_FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer),
        client=client,
    )


def answer_correctness_llm_judge(question: str, answer: str, reference: str, client=None) -> int:
    """Only usable if you've written your own reference answer for a
    self-authored internal test question -- the official 20 questions'
    reference answers are never disclosed to students."""
    return _judge_score(
        JUDGE_CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer),
        client=client,
    )
