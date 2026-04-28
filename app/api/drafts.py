"""Drafts API (async LLM, sync DB)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.schemas import DraftRequest, DraftResponse
from app.services.drafting_service import generate_draft

router = APIRouter()


@router.post("/generate", response_model=DraftResponse)
async def create_draft(req: DraftRequest, db: Session = Depends(get_db)):
    result = await generate_draft(db=db, draft_type=req.draft_type, case_identifiers=req.case_identifiers, instructions=req.instructions)
    return DraftResponse(**result)
