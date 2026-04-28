"""Dashboard API (sync SQLite)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.database import get_db
from app.models.case import Case
from app.schemas.schemas import DashboardStats, CaseListItem

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    total_cases = db.query(func.count(Case.id)).scalar() or 0
    total_courts = db.query(func.count(func.distinct(Case.state))).scalar() or 0

    yr = db.query(func.min(Case.search_year), func.max(Case.search_year)).one_or_none()
    year_range = [yr[0] or 0, yr[1] or 0] if yr else [0, 0]

    by_year = db.query(Case.search_year, func.count(Case.id)).group_by(Case.search_year).order_by(Case.search_year).all()
    cases_by_year = {str(r[0]): r[1] for r in by_year if r[0]}

    by_court = db.query(Case.state, func.count(Case.id)).group_by(Case.state).order_by(func.count(Case.id).desc()).limit(15).all()
    cases_by_court = {r[0]: r[1] for r in by_court if r[0]}

    by_status = db.query(Case.case_status, func.count(Case.id)).group_by(Case.case_status).order_by(func.count(Case.id).desc()).all()
    cases_by_status = {(r[0] or "Unknown"): r[1] for r in by_status}

    by_company = db.query(Case.company_role, func.count(Case.id)).group_by(Case.company_role).all()
    cases_by_company = {(r[0] or "Unknown"): r[1] for r in by_company}

    recent = db.query(Case).order_by(Case.id.desc()).limit(5).all()

    return DashboardStats(
        total_cases=total_cases, total_courts=total_courts, year_range=year_range,
        cases_by_year=cases_by_year, cases_by_court=cases_by_court,
        cases_by_status=cases_by_status, cases_by_company=cases_by_company,
        recent_cases=[CaseListItem.model_validate(c) for c in recent],
    )
