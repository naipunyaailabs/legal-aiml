"""Search API (sync)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.schemas import SearchRequest, SearchResponse, SearchResultItem, CaseListItem
from app.services.search_service import hybrid_search

router = APIRouter()


@router.post("", response_model=SearchResponse)
def search_cases(req: SearchRequest, db: Session = Depends(get_db)):
    results = hybrid_search(db=db, query=req.query, state=req.state, year=req.year,
                            company_tag=req.company_tag, act=req.act, stage=req.stage,
                            top_k=200)
    items = [SearchResultItem(case=CaseListItem.model_validate(r["case"]), score=round(r["score"], 4),
                              matched_chunk=r.get("matched_chunk", ""), match_reason=r.get("match_reason", ""))
             for r in results]
    return SearchResponse(results=items, total=len(items), query=req.query)
