"""SQLAlchemy models for legal case data — SQLite compatible."""
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, JSON, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cino = Column(String(100), unique=True, nullable=False, index=True)
    cnr_number = Column(String(100), index=True)
    state = Column(String(200), index=True)
    court_name = Column(String(200))
    bench_id = Column(Integer)
    bench_type = Column(String(50))
    judicial_branch = Column(String(100))
    district = Column(String(200))
    search_query = Column(String(200))
    search_year = Column(Integer, index=True)

    # Parties
    raw_petitioner = Column(Text)
    raw_respondent = Column(Text)
    normalized_petitioner = Column(Text)
    normalized_respondent = Column(Text)
    company_tags = Column(JSON)
    company_role = Column(String(50))

    # Filing / Registration
    filing_number = Column(String(200))
    filing_date = Column(String(100))
    registration_number = Column(String(200))
    registration_date = Column(String(100))

    # Case status
    first_hearing_date = Column(String(100))
    next_hearing_date = Column(String(100))
    decision_date = Column(String(100))
    stage_of_case = Column(String(300))
    case_status = Column(String(200))
    nature_of_disposal = Column(String(300))
    coram = Column(Text)

    # Advocates
    petitioner_advocate = Column(Text)
    respondent_advocate = Column(Text)

    # Meta
    source_year_file = Column(String(200))
    raw_json = Column(JSON)
    summary_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    acts = relationship("CaseAct", back_populates="case", cascade="all, delete-orphan")
    hearings = relationship("Hearing", back_populates="case", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="case", cascade="all, delete-orphan")
    objections = relationship("Objection", back_populates="case", cascade="all, delete-orphan")
    chunks = relationship("CaseChunk", back_populates="case", cascade="all, delete-orphan")


class CaseAct(Base):
    __tablename__ = "case_acts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    act_name = Column(String(500))
    section = Column(String(200))

    case = relationship("Case", back_populates="acts")


class Hearing(Base):
    __tablename__ = "hearings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    cause_list_type = Column(String(100))
    judge = Column(Text)
    business_on_date = Column(String(100))
    hearing_date = Column(String(100))
    purpose = Column(Text)

    case = relationship("Case", back_populates="hearings")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    order_number = Column(String(50))
    order_on = Column(String(300))
    judge = Column(Text)
    order_date = Column(String(100))

    case = relationship("Case", back_populates="orders")


class Objection(Base):
    __tablename__ = "objections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    scrutiny_date = Column(String(100))
    objection_text = Column(Text)
    compliance_date = Column(String(100))
    receipt_date = Column(String(100))

    case = relationship("Case", back_populates="objections")


class CaseChunk(Base):
    __tablename__ = "case_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    chunk_type = Column(String(50))
    chunk_text = Column(Text, nullable=False)
    # Store embedding as JSON array (numpy-based search at runtime)
    embedding_json = Column(Text)

    case = relationship("Case", back_populates="chunks")
