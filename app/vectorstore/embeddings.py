from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode([query], normalize_embeddings=True)[0]
    return vector.tolist()
