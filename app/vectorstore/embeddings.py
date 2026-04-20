from __future__ import annotations

import logging
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when embedding service cannot provide valid vectors."""


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        openai_api_key=settings.embedding_api_key,
        openai_api_base=settings.embedding_base_url,
        # Local OpenAI-compatible services do not need tiktoken length checks.
        check_embedding_ctx_length=False,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        vectors = get_embedding_client().embed_documents(texts)
        if not vectors:
            raise EmbeddingServiceError("embedding service returned empty vectors for documents")
        if len(vectors) != len(texts):
            raise EmbeddingServiceError(
                f"embedding vector count mismatch: expected={len(texts)}, got={len(vectors)}"
            )
        return vectors
    except Exception as e:
        logger.warning(f"[embedding] embedding failed: {e}")
        raise EmbeddingServiceError(f"embedding service unavailable for documents: {e}") from e


def embed_query(query: str) -> list[float]:
    try:
        vector = get_embedding_client().embed_query(query)
        if not vector:
            raise EmbeddingServiceError("embedding service returned empty vector for query")
        return vector
    except Exception as e:
        logger.warning(f"[embedding] embedding failed: {e}")
        raise EmbeddingServiceError(f"embedding service unavailable for query: {e}") from e
