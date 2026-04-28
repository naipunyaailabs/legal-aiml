"""FastAPI application entry point."""
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.db.database import init_db, get_db
from app.api import cases, search, ai, drafts, ingestion, dashboard
from app.services.ingestion_service import ingest_all
from app.services.data_hash import compute_data_hash, read_stored_hash, write_stored_hash

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    settings = get_settings()
    data_dir = Path(settings.DATA_DIR)
    hash_file = data_dir.parent / ".data_hash.json"

    if data_dir.exists():
        current_hash = compute_data_hash(data_dir)
        stored_hash = read_stored_hash(hash_file)

        if current_hash != stored_hash:
            logger.info("Data directory changed — running ingestion.")
            db = next(get_db())
            try:
                result = ingest_all(db)
                logger.info(f"Ingestion complete: {result['cases_ingested']} cases ingested from {result['files_processed']} files.")
                write_stored_hash(hash_file, current_hash)
            finally:
                db.close()
        else:
            logger.info("Data directory unchanged — skipping ingestion.")
    else:
        logger.warning(f"Data directory not found: {data_dir}")

    yield


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(cases.router, prefix="/api/cases", tags=["Cases"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI"])
app.include_router(drafts.router, prefix="/api/drafts", tags=["Drafts"])
app.include_router(ingestion.router, prefix="/api/ingestion", tags=["Ingestion"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME}
