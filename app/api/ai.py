"""AI API — RAG QA and summarization (async LLM, sync DB)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.schemas import AiQueryRequest, AiAnswerResponse, CitationSource, SummarizeRequest, SummarizeResponse
from app.services.rag_service import answer_question, summarize_case

router = APIRouter()


@router.post("/answer", response_model=AiAnswerResponse)
async def ai_answer(req: AiQueryRequest, db: Session = Depends(get_db)):
    result = await answer_question(db=db, question=req.question, mode=req.mode, max_sources=req.max_sources)
    sources = [CitationSource(**s) for s in result.get("sources", [])]
    return AiAnswerResponse(answer=result["answer"], confidence=result["confidence"], sources=sources,
                             mode=result["mode"], insufficient_evidence=result.get("insufficient_evidence", False))


@router.post("/summarize", response_model=SummarizeResponse)
async def ai_summarize(req: SummarizeRequest, db: Session = Depends(get_db)):
    result = await summarize_case(db, req.case_id)
    return SummarizeResponse(**result)
