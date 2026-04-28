"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Case Schemas ---

class CaseActOut(BaseModel):
    id: int
    act_name: Optional[str] = None
    section: Optional[str] = None
    model_config = {"from_attributes": True}


class HearingOut(BaseModel):
    id: int
    cause_list_type: Optional[str] = None
    judge: Optional[str] = None
    business_on_date: Optional[str] = None
    hearing_date: Optional[str] = None
    purpose: Optional[str] = None
    model_config = {"from_attributes": True}


class OrderOut(BaseModel):
    id: int
    order_number: Optional[str] = None
    order_on: Optional[str] = None
    judge: Optional[str] = None
    order_date: Optional[str] = None
    model_config = {"from_attributes": True}


class ObjectionOut(BaseModel):
    id: int
    scrutiny_date: Optional[str] = None
    objection_text: Optional[str] = None
    compliance_date: Optional[str] = None
    receipt_date: Optional[str] = None
    model_config = {"from_attributes": True}


class CaseListItem(BaseModel):
    id: int
    cino: str
    state: Optional[str] = None
    court_name: Optional[str] = None
    search_year: Optional[int] = None
    raw_petitioner: Optional[str] = None
    raw_respondent: Optional[str] = None
    company_tags: Optional[list] = None
    company_role: Optional[str] = None
    filing_number: Optional[str] = None
    stage_of_case: Optional[str] = None
    case_status: Optional[str] = None
    coram: Optional[str] = None
    next_hearing_date: Optional[str] = None
    model_config = {"from_attributes": True}


class CaseDetail(CaseListItem):
    cnr_number: Optional[str] = None
    bench_id: Optional[int] = None
    bench_type: Optional[str] = None
    judicial_branch: Optional[str] = None
    district: Optional[str] = None
    search_query: Optional[str] = None
    normalized_petitioner: Optional[str] = None
    normalized_respondent: Optional[str] = None
    filing_date: Optional[str] = None
    registration_number: Optional[str] = None
    registration_date: Optional[str] = None
    first_hearing_date: Optional[str] = None
    decision_date: Optional[str] = None
    nature_of_disposal: Optional[str] = None
    petitioner_advocate: Optional[str] = None
    respondent_advocate: Optional[str] = None
    summary_text: Optional[str] = None
    created_at: Optional[datetime] = None
    acts: list[CaseActOut] = []
    hearings: list[HearingOut] = []
    orders: list[OrderOut] = []
    objections: list[ObjectionOut] = []
    model_config = {"from_attributes": True}


class CaseListResponse(BaseModel):
    cases: list[CaseListItem]
    total: int
    page: int
    page_size: int


# --- Search Schemas ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    state: Optional[str] = None
    year: Optional[int] = None
    company_tag: Optional[str] = None
    act: Optional[str] = None
    stage: Optional[str] = None
    page: int = 1
    page_size: int = 20


class SearchResultItem(BaseModel):
    case: CaseListItem
    score: float
    matched_chunk: Optional[str] = None
    match_reason: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int
    query: str


# --- AI Schemas ---

class AiQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="research", pattern="^(research|advisory|executive)$")
    max_sources: int = Field(default=5, ge=1, le=10)


class CitationSource(BaseModel):
    case_id: int
    cino: str
    case_title: str
    court: Optional[str] = None
    year: Optional[int] = None
    relevant_passage: str
    relevance_score: float


class AiAnswerResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[CitationSource]
    mode: str
    insufficient_evidence: bool = False


class SummarizeRequest(BaseModel):
    case_id: int


class SummarizeResponse(BaseModel):
    case_id: int
    summary: str
    key_facts: list[str]


# --- Draft Schemas ---

class DraftRequest(BaseModel):
    draft_type: str = Field(..., pattern="^(notice|reply|brief|memo)$")
    case_identifiers: list[str] = Field(..., min_length=1, max_length=10)
    instructions: Optional[str] = None


class DraftResponse(BaseModel):
    draft_type: str
    content: str
    cited_cases: list[str]
    placeholders: list[str]


# --- Ingestion Schemas ---

class IngestionRequest(BaseModel):
    file_pattern: Optional[str] = None  # e.g., "2024" to ingest only that year


class IngestionResponse(BaseModel):
    status: str
    files_processed: int
    cases_ingested: int
    errors: list[str]


# --- Dashboard Schemas ---

class DashboardStats(BaseModel):
    total_cases: int
    total_courts: int
    year_range: list[int]
    cases_by_year: dict[str, int]
    cases_by_court: dict[str, int]
    cases_by_status: dict[str, int]
    cases_by_company: dict[str, int]
    recent_cases: list[CaseListItem]
