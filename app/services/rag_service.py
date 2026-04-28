"""RAG service — citation-grounded legal QA (sync SQLite)."""
import json
import logging
from sqlalchemy.orm import Session

from app.services.search_service import hybrid_search
from app.services.llm_service import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "research": (
        "You are an expert legal researcher specializing in Indian law, including the Indian Penal Code (IPC), "
        "Code of Criminal Procedure (CrPC), Code of Civil Procedure (CPC), Companies Act, Income Tax Act, "
        "and judgments of the Supreme Court of India and various High Courts. "
        "Answer ONLY using the provided case evidence below. Cite each claim with the CINO number and court name. "
        "If evidence is insufficient, explicitly state so. Never invent cases or legal provisions. "
        "Output valid JSON with keys: answer (string), confidence (0.0-1.0), "
        "key_points (list of strings), insufficient_evidence (bool)."
    ),
    "advisory": (
        "You are a senior Indian legal advisor with expertise in corporate litigation, tax disputes, "
        "arbitration, and High Court/Supreme Court writ proceedings. "
        "Provide actionable recommendations grounded strictly in the provided case evidence. "
        "Reference relevant Indian statutes and precedents. Label all reasoning as AI-assisted analysis. "
        "Output valid JSON with keys: answer (string), confidence (0.0-1.0), "
        "recommendations (list of strings), caveats (list of strings), insufficient_evidence (bool)."
    ),
    "executive": (
        "You are summarizing Indian court litigation for senior business leadership (non-lawyers). "
        "Use plain, concise English. Avoid legal jargon where possible. "
        "Focus on business risk, current case status, and next steps. "
        "Output valid JSON with keys: answer (string), confidence (0.0-1.0), "
        "key_takeaways (list of strings), insufficient_evidence (bool)."
    ),
}


def _build_evidence_context(search_results: list[dict], max_sources: int) -> tuple[str, list[dict]]:
    sources = []
    evidence_parts = []
    for i, result in enumerate(search_results[:max_sources]):
        case = result["case"]
        chunk = result.get("matched_chunk", "")
        case_title = f"{case.raw_petitioner} vs {case.raw_respondent}"
        sources.append({
            "case_id": case.id,
            "cino": case.cino,
            "case_title": case_title,
            "court": case.state,
            "year": case.search_year,
            "relevant_passage": chunk[:500],
            "relevance_score": result["score"],
        })
        evidence_parts.append(
            f"[Source {i+1}] Case: {case_title} | Court: {case.state} | "
            f"Year: {case.search_year} | CINO: {case.cino}\n"
            f"Status: {case.case_status or 'N/A'} | Stage: {case.stage_of_case or 'N/A'}\n"
            f"Evidence: {chunk[:400]}"
        )
    return "\n\n".join(evidence_parts), sources


async def answer_question(db: Session, question: str, mode: str = "research", max_sources: int = 5) -> dict:
    search_results = hybrid_search(db=db, query=question, top_k=20, rerank_top_k=max_sources)

    if not search_results:
        return {
            "answer": "No relevant cases found in the database for your query.",
            "confidence": 0.0, "sources": [], "mode": mode, "insufficient_evidence": True,
        }

    evidence_text, sources = _build_evidence_context(search_results, max_sources)
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["research"])
    user_prompt = f"Question: {question}\n\nEvidence:\n{evidence_text}\n\nProvide your answer as JSON."

    try:
        raw_response = await call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = {"answer": raw_response, "confidence": 0.5, "insufficient_evidence": False}

        return {
            "answer": parsed.get("answer", raw_response),
            "confidence": float(parsed.get("confidence", 0.5)),
            "sources": sources, "mode": mode,
            "insufficient_evidence": parsed.get("insufficient_evidence", False),
        }
    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}", "confidence": 0.0,
            "sources": sources, "mode": mode, "insufficient_evidence": True,
        }


async def summarize_case(db: Session, case_id: int) -> dict:
    from app.models.case import Case
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        return {"case_id": case_id, "summary": "Case not found.", "key_facts": []}

    case_text = (
        f"Case: {case.raw_petitioner} vs {case.raw_respondent}\n"
        f"Court: {case.state}\nYear: {case.search_year}\n"
        f"Status: {case.case_status or 'N/A'}\nStage: {case.stage_of_case or 'N/A'}\n"
        f"Judge: {case.coram or 'N/A'}\nFiling: {case.filing_number or 'N/A'}"
    )
    try:
        response = await call_llm(
            messages=[
                {"role": "system", "content": "Summarize this Indian legal case concisely. Output JSON with keys: summary, key_facts (list)."},
                {"role": "user", "content": case_text},
            ],
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response)
        return {"case_id": case_id, "summary": parsed.get("summary", ""), "key_facts": parsed.get("key_facts", [])}
    except Exception as e:
        return {"case_id": case_id, "summary": f"Error: {str(e)}", "key_facts": []}
