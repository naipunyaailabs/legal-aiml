"""Embedding service using local sentence-transformers model."""
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
from app.config import get_settings

settings = get_settings()

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load embedding model singleton."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Generate embedding for a single search query."""
    model = get_embedding_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()
