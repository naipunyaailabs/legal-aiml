"""Ingestion API (sync)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.schemas import IngestionRequest, IngestionResponse
from app.services.ingestion_service import ingest_all

router = APIRouter()


@router.post("/run", response_model=IngestionResponse)
def run_ingestion(req: IngestionRequest, db: Session = Depends(get_db)):
    result = ingest_all(db, file_pattern=req.file_pattern)
    return IngestionResponse(**result)
