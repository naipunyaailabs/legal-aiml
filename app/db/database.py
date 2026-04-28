"""SQLite database engine and session management."""
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from app.config import get_settings

settings = get_settings()

db_path = settings.DATABASE_PATH
engine = create_engine(f"sqlite:///{db_path}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    # Import models so they register with Base
    from app.models.case import Case, CaseAct, Hearing, Order, Objection, CaseChunk  # noqa
    Base.metadata.create_all(bind=engine)
