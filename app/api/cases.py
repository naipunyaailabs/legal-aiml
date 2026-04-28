"""Cases API — list, filter, and detail endpoints (sync SQLite)."""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_

from app.db.database import get_db
from app.models.case import Case, CaseAct
from app.schemas.schemas import CaseListItem, CaseListResponse, CaseDetail

router = APIRouter()


@router.get("", response_model=CaseListResponse)
def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    state: str | None = None,
    year: int | None = None,
    company_tag: str | None = None,
    stage: str | None = None,
    status: str | None = None,
    search: str | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(Case)
    cq = db.query(func.count(Case.id))

    if state:
        q = q.filter(Case.state.ilike(f"%{state}%"))
        cq = cq.filter(Case.state.ilike(f"%{state}%"))
    if year:
        q = q.filter(Case.search_year == year)
        cq = cq.filter(Case.search_year == year)
    if stage:
        q = q.filter(Case.stage_of_case.ilike(f"%{stage}%"))
        cq = cq.filter(Case.stage_of_case.ilike(f"%{stage}%"))
    if status:
        q = q.filter(Case.case_status.ilike(f"%{status}%"))
        cq = cq.filter(Case.case_status.ilike(f"%{status}%"))
    if search:
        f = or_(Case.raw_petitioner.ilike(f"%{search}%"), Case.raw_respondent.ilike(f"%{search}%"), Case.cino.ilike(f"%{search}%"))
        q = q.filter(f)
        cq = cq.filter(f)

    total = cq.scalar() or 0
    offset = (page - 1) * page_size
    cases = q.order_by(Case.search_year.desc(), Case.id.desc()).offset(offset).limit(page_size).all()

    return CaseListResponse(
        cases=[CaseListItem.model_validate(c) for c in cases],
        total=total, page=page, page_size=page_size,
    )


@router.get("/filters/options")
def get_filter_options(db: Session = Depends(get_db)):
    states = [r[0] for r in db.query(Case.state).distinct().filter(Case.state.isnot(None)).all()]
    years = [r[0] for r in db.query(Case.search_year).distinct().filter(Case.search_year.isnot(None)).order_by(Case.search_year.desc()).all()]
    stages = [r[0] for r in db.query(Case.stage_of_case).distinct().filter(Case.stage_of_case.isnot(None)).all()]
    statuses = [r[0] for r in db.query(Case.case_status).distinct().filter(Case.case_status.isnot(None)).all()]
    return {"states": states, "years": years, "stages": stages, "statuses": statuses}


@router.get("/{case_id}", response_model=CaseDetail)
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).options(
        joinedload(Case.acts), joinedload(Case.hearings),
        joinedload(Case.orders), joinedload(Case.objections),
    ).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return CaseDetail.model_validate(case)
