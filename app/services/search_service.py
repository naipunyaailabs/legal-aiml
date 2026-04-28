"""Search service — hybrid semantic + keyword search with reranking (SQLite)."""
import json
import logging
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.models.case import Case, CaseChunk, CaseAct
from app.services.embedding_service import embed_query
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))


def hybrid_search(
    db: Session,
    query: str,
    state: str | None = None,
    year: int | None = None,
    company_tag: str | None = None,
    act: str | None = None,
    stage: str | None = None,
    top_k: int = 20,
    rerank_top_k: int = 5,
) -> list[dict]:
    """Hybrid search: vector similarity + keyword, then rerank."""

    # 1. Vector search over all chunks with embeddings
    query_embedding = embed_query(query)

    chunks = db.query(CaseChunk).filter(CaseChunk.embedding_json.isnot(None)).all()

    chunk_scores = []
    for chunk in chunks:
        try:
            emb = json.loads(chunk.embedding_json)
            sim = _cosine_similarity(query_embedding, emb)
            chunk_scores.append((chunk, sim))
        except Exception:
            continue

    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = chunk_scores[:top_k]

    # 2. Keyword search
    keyword_cases = db.query(Case).filter(
        or_(
            Case.summary_text.ilike(f"%{query}%"),
            Case.raw_petitioner.ilike(f"%{query}%"),
            Case.raw_respondent.ilike(f"%{query}%"),
        )
    ).limit(top_k).all()

    # 3. Merge candidates
    case_scores: dict[int, dict] = {}

    for chunk, sim in top_chunks:
        cid = chunk.case_id
        if cid not in case_scores or sim > case_scores[cid].get("vector_score", 0):
            case_scores[cid] = {
                "vector_score": sim,
                "matched_chunk": chunk.chunk_text,
                "chunk_type": chunk.chunk_type,
            }

    for case in keyword_cases:
        if case.id not in case_scores:
            case_scores[case.id] = {"vector_score": 0.3, "matched_chunk": case.summary_text or ""}
        case_scores[case.id]["keyword_hit"] = True

    if not case_scores:
        return []

    # 4. Fetch cases and apply filters
    case_ids = list(case_scores.keys())
    q = db.query(Case).filter(Case.id.in_(case_ids))

    if state:
        q = q.filter(Case.state.ilike(f"%{state}%"))
    if year:
        q = q.filter(Case.search_year == year)
    if stage:
        q = q.filter(Case.stage_of_case.ilike(f"%{stage}%"))

    cases = q.all()

    # Filter by company_tag
    if company_tag:
        cases = [c for c in cases if company_tag in (c.company_tags or [])]

    # Filter by act
    if act:
        act_case_ids = {r.case_id for r in db.query(CaseAct.case_id).filter(CaseAct.act_name.ilike(f"%{act}%")).all()}
        cases = [c for c in cases if c.id in act_case_ids]

    # 5. Rerank
    if cases:
        try:
            reranker = get_reranker()
            pairs = [(query, case_scores.get(c.id, {}).get("matched_chunk", c.summary_text or "")) for c in cases]
            rerank_scores = reranker.predict(pairs)

            results = []
            for case, rscore in zip(cases, rerank_scores):
                cs = case_scores.get(case.id, {})
                results.append({
                    "case": case,
                    "score": float(rscore),
                    "matched_chunk": cs.get("matched_chunk", ""),
                    "match_reason": cs.get("chunk_type", "keyword"),
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            # Filter by relevance threshold (e.g., 0.2) to show only "good" matches
            relevant_results = [r for r in results if r["score"] > 0.2]
            # Return all relevant results, but at least 3 to show something
            return relevant_results if relevant_results else results[:3]
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")

    # Fallback
    results = []
    for case in cases:
        cs = case_scores.get(case.id, {})
        results.append({
            "case": case,
            "score": cs.get("vector_score", 0.5),
            "matched_chunk": cs.get("matched_chunk", ""),
            "match_reason": cs.get("chunk_type", "keyword"),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    relevant_results = [r for r in results if r["score"] > 0.3]
    return relevant_results if relevant_results else results[:3]
