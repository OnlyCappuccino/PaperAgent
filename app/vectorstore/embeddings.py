from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        openai_api_key=settings.embedding_api_key,
        openai_api_base=settings.embedding_base_url,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_embedding_client().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    return get_embedding_client().embed_query(query)
