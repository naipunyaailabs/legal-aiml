"""Ingestion service — parses yearly JSON files into normalized DB records."""
import json
import os
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.case import Case, CaseAct, Hearing, Order, Objection, CaseChunk
from app.services.embedding_service import embed_texts
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _extract_basic_detail(details: list[dict], key_fragment: str) -> str | None:
    for row in details:
        for k, v in row.items():
            if isinstance(v, str) and key_fragment.lower() in v.lower():
                keys = list(row.keys())
                idx = keys.index(k)
                if idx + 1 < len(keys):
                    return row[keys[idx + 1]]
    return None


def _normalize_case(raw: dict, source_file: str) -> dict:
    basic = raw.get("basic_details", [])
    status = raw.get("case_status", {})

    company_tags = raw.get("matched_company_tags", [])
    petitioner = (raw.get("raw_petitioner") or "").upper()
    respondent = (raw.get("raw_respondent") or "").upper()
    company_role = "unknown"
    for tag in company_tags:
        if tag.upper() in petitioner:
            company_role = "petitioner"
            break
        elif tag.upper() in respondent:
            company_role = "respondent"
            break

    return {
        "cino": raw.get("cino", ""),
        "cnr_number": _extract_basic_detail(basic, "CNR"),
        "state": raw.get("state", ""),
        "court_name": raw.get("state", ""),
        "bench_id": raw.get("bench_id"),
        "bench_type": status.get("Bench Type"),
        "judicial_branch": status.get("Judicial Branch"),
        "district": status.get("District"),
        "search_query": raw.get("search_query"),
        "search_year": raw.get("search_year"),
        "raw_petitioner": raw.get("raw_petitioner"),
        "raw_respondent": raw.get("raw_respondent"),
        "normalized_petitioner": (raw.get("raw_petitioner") or "").strip().title(),
        "normalized_respondent": (raw.get("raw_respondent") or "").strip().title(),
        "company_tags": company_tags,
        "company_role": company_role,
        "filing_number": _extract_basic_detail(basic, "Filing Number"),
        "filing_date": _extract_basic_detail(basic, "Filing Date"),
        "registration_number": _extract_basic_detail(basic, "Registration Number"),
        "registration_date": _extract_basic_detail(basic, "Registration Date"),
        "first_hearing_date": status.get("First Hearing Date"),
        "next_hearing_date": status.get("Next Hearing Date"),
        "decision_date": status.get("Decision Date"),
        "stage_of_case": status.get("Stage of Case"),
        "case_status": status.get("Case Status"),
        "nature_of_disposal": status.get("Nature of Disposal"),
        "coram": status.get("Coram"),
        "petitioner_advocate": raw.get("petitioner_advocate"),
        "respondent_advocate": raw.get("respondent_advocate"),
        "source_year_file": source_file,
        "raw_json": raw,
    }


def _build_summary_text(case_data: dict, raw: dict) -> str:
    parts = []
    parts.append(f"Case: {case_data.get('raw_petitioner', '')} vs {case_data.get('raw_respondent', '')}")
    parts.append(f"Court: {case_data.get('state', '')}")
    if case_data.get("district"):
        parts.append(f"District: {case_data['district']}")
    parts.append(f"Year: {case_data.get('search_year', '')}")
    if case_data.get("stage_of_case"):
        parts.append(f"Stage: {case_data['stage_of_case']}")
    if case_data.get("case_status"):
        parts.append(f"Status: {case_data['case_status']}")
    if case_data.get("coram"):
        parts.append(f"Judge: {case_data['coram']}")
    for a in raw.get("acts", []):
        act = a.get("Under Act(s)", "")
        sec = a.get("Under Section(s)", "")
        if act:
            parts.append(f"Act: {act} Section: {sec}")
    tags = case_data.get("company_tags", [])
    if tags:
        parts.append(f"Company: {', '.join(tags)} (as {case_data.get('company_role', 'unknown')})")
    if case_data.get("filing_number"):
        parts.append(f"Filing: {case_data['filing_number']}")
    return ". ".join(parts)


def _build_chunks(case_data: dict, raw: dict) -> list[dict]:
    chunks = []
    summary = _build_summary_text(case_data, raw)
    chunks.append({"chunk_type": "summary", "chunk_text": summary})

    hearings = raw.get("history_of_hearing", [])
    if hearings:
        hearing_lines = []
        for h in hearings:
            line = f"Hearing on {h.get('Hearing Date', 'N/A')}: {h.get('Purpose of hearing', '')} before {h.get('Judge', '')}"
            hearing_lines.append(line)
        chunks.append({"chunk_type": "hearing", "chunk_text": ". ".join(hearing_lines)})

    status_parts = [f"Case status: {case_data.get('case_status', 'N/A')}"]
    if case_data.get("stage_of_case"):
        status_parts.append(f"Stage: {case_data['stage_of_case']}")
    if case_data.get("nature_of_disposal"):
        status_parts.append(f"Disposal: {case_data['nature_of_disposal']}")
    if case_data.get("next_hearing_date"):
        status_parts.append(f"Next hearing: {case_data['next_hearing_date']}")
    chunks.append({"chunk_type": "status", "chunk_text": ". ".join(status_parts)})

    return chunks


def ingest_file(filepath: str, db: Session) -> dict:
    filename = os.path.basename(filepath)
    logger.info(f"Ingesting {filename}...")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    ingested = 0
    skipped = 0
    errors = []

    for raw in raw_cases:
        try:
            cino = raw.get("cino", "")
            if not cino:
                skipped += 1
                continue

            existing = db.query(Case).filter(Case.cino == cino).first()
            if existing:
                skipped += 1
                continue

            case_data = _normalize_case(raw, filename)
            summary = _build_summary_text(case_data, raw)
            case_data["summary_text"] = summary

            case = Case(**case_data)
            db.add(case)
            db.flush()

            for act_raw in raw.get("acts", []):
                db.add(CaseAct(
                    case_id=case.id,
                    act_name=str(act_raw.get("Under Act(s)", "")),
                    section=str(act_raw.get("Under Section(s)", "")),
                ))

            for h in raw.get("history_of_hearing", []):
                db.add(Hearing(
                    case_id=case.id,
                    cause_list_type=h.get("Cause List Type"),
                    judge=h.get("Judge"),
                    business_on_date=h.get("Business On Date"),
                    hearing_date=h.get("Hearing Date"),
                    purpose=h.get("Purpose of hearing"),
                ))

            for o in raw.get("orders", []):
                if o.get("0") == "Order Number":
                    continue
                db.add(Order(
                    case_id=case.id,
                    order_number=o.get("0"),
                    order_on=o.get("1"),
                    judge=o.get("2"),
                    order_date=o.get("3"),
                ))

            for obj in raw.get("objections", []):
                if obj.get("0") == "Sr.No.":
                    continue
                db.add(Objection(
                    case_id=case.id,
                    scrutiny_date=obj.get("1"),
                    objection_text=obj.get("2"),
                    compliance_date=obj.get("3"),
                    receipt_date=obj.get("4"),
                ))

            for cd in _build_chunks(case_data, raw):
                db.add(CaseChunk(
                    case_id=case.id,
                    chunk_type=cd["chunk_type"],
                    chunk_text=cd["chunk_text"],
                ))

            ingested += 1

        except Exception as e:
            errors.append(f"Error on cino={raw.get('cino', '?')}: {str(e)}")
            logger.error(f"Ingestion error: {e}")

    db.commit()
    logger.info(f"Ingested {ingested} cases from {filename} ({skipped} skipped)")
    return {"file": filename, "ingested": ingested, "skipped": skipped, "errors": errors}


def generate_embeddings(db: Session, batch_size: int = 100):
    chunks = db.query(CaseChunk).filter(CaseChunk.embedding_json.is_(None)).all()
    if not chunks:
        logger.info("All chunks already have embeddings.")
        return 0

    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.chunk_text for c in batch]
        embeddings = embed_texts(texts)
        for chunk, emb in zip(batch, embeddings):
            chunk.embedding_json = json.dumps(emb)
        db.commit()

    logger.info(f"Generated embeddings for {len(chunks)} chunks.")
    return len(chunks)


def ingest_all(db: Session, file_pattern: str | None = None) -> dict:
    data_dir = Path(settings.DATA_DIR)
    if not data_dir.exists():
        return {"status": "error", "files_processed": 0, "cases_ingested": 0,
                "errors": [f"Data directory not found: {data_dir}"]}

    files = sorted(data_dir.glob("*.json"))
    if file_pattern:
        files = [f for f in files if file_pattern in f.name]

    total_ingested = 0
    total_errors = []

    for filepath in files:
        result = ingest_file(str(filepath), db)
        total_ingested += result["ingested"]
        total_errors.extend(result["errors"])

    generate_embeddings(db)

    return {
        "status": "success",
        "files_processed": len(files),
        "cases_ingested": total_ingested,
        "errors": total_errors[:50],
    }
