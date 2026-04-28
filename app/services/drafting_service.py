"""Drafting service — generates legal documents (sync SQLite)."""
import json
import logging
from sqlalchemy.orm import Session

from app.models.case import Case
from app.services.llm_service import call_llm

logger = logging.getLogger(__name__)

DRAFT_PROMPTS = {
    "notice": (
        "You are a senior advocate specializing in Indian corporate and civil litigation. "
        "Draft a formal legal notice as per Indian legal conventions. "
        "Include: proper address block, subject line, factual background (drawn from the provided cases), "
        "legal grounds (citing relevant IPC/CrPC/CPC sections or Indian statutes), relief demanded, "
        "and a clear response deadline. Use formal language used in Indian High Courts. "
        "Mark [PLACEHOLDER] wherever specific facts need to be filled in. "
        "Output valid JSON: {content: string, cited_cases: list of CINOs, placeholders: list of strings}."
    ),
    "reply": (
        "You are a senior advocate drafting a formal reply on behalf of a party in Indian civil/corporate litigation. "
        "Structure as: (1) Preliminary Objections, (2) Para-wise Reply, (3) Prayer. "
        "Cite relevant Supreme Court and High Court precedents from the provided cases. "
        "Use formal language as per Indian court practice. Mark [PLACEHOLDER] where needed. "
        "Output valid JSON: {content: string, cited_cases: list of CINOs, placeholders: list of strings}."
    ),
    "brief": (
        "You are a legal researcher preparing a case brief for Indian litigation. "
        "Format: (1) Facts, (2) Issues Framed, (3) Procedural History, "
        "(4) Applicable Laws & Sections (Indian statutes), (5) Arguments, (6) Current Status & Outcome. "
        "Reference hearing dates from the evidence. "
        "Output valid JSON: {content: string, cited_cases: list of CINOs, placeholders: list of strings}."
    ),
    "memo": (
        "You are a legal associate preparing an internal legal memorandum for Indian in-house counsel. "
        "Format: TO / FROM / DATE / RE / ISSUE / BRIEF ANSWER / FACTS / ANALYSIS / CONCLUSION. "
        "Analyze risk under Indian law (Companies Act, Income Tax Act, Arbitration Act, etc.) "
        "based on the provided case details. Be concise and actionable. "
        "Output valid JSON: {content: string, cited_cases: list of CINOs, placeholders: list of strings}."
    ),
}


async def generate_draft(db: Session, draft_type: str, case_identifiers: list[str], instructions: str | None = None) -> dict:
    from sqlalchemy import or_
    
    # Separate numeric IDs from CINOs
    numeric_ids = [int(i) for i in case_identifiers if str(i).isdigit()]
    cinos = [str(i) for i in case_identifiers if not str(i).isdigit()]
    
    conditions = []
    if numeric_ids:
        conditions.append(Case.id.in_(numeric_ids))
    if cinos:
        conditions.append(Case.cino.in_(cinos))
        
    if not conditions:
        return {"draft_type": draft_type, "content": "No valid case identifiers provided.", "cited_cases": [], "placeholders": []}

    cases = db.query(Case).filter(or_(*conditions)).all()
    if not cases:
        return {"draft_type": draft_type, "content": "No cases found.", "cited_cases": [], "placeholders": []}

    case_contexts = []
    for case in cases:
        ctx = (
            f"CINO: {case.cino}\nParties: {case.raw_petitioner} vs {case.raw_respondent}\n"
            f"Court: {case.state}\nYear: {case.search_year}\n"
            f"Status: {case.case_status or 'N/A'}\nStage: {case.stage_of_case or 'N/A'}\n"
            f"Judge: {case.coram or 'N/A'}\nFiling: {case.filing_number or 'N/A'}"
        )
        case_contexts.append(ctx)

    cases_text = "\n\n---\n\n".join(case_contexts)
    system_prompt = DRAFT_PROMPTS.get(draft_type, DRAFT_PROMPTS["brief"])
    user_prompt = f"Cases:\n{cases_text}"
    if instructions:
        user_prompt += f"\n\nAdditional instructions: {instructions}"

    try:
        response = await call_llm(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=2000, response_format={"type": "json_object"},
        )
        parsed = json.loads(response)
        return {
            "draft_type": draft_type,
            "content": parsed.get("content", response),
            "cited_cases": parsed.get("cited_cases", [c.cino for c in cases]),
            "placeholders": parsed.get("placeholders", []),
        }
    except Exception as e:
        logger.error(f"Draft generation failed: {e}")
        return {"draft_type": draft_type, "content": f"Error: {str(e)}", "cited_cases": [], "placeholders": []}
