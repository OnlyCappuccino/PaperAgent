from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    llm_base_url: str = 'http://127.0.0.1:11434/v1'
    llm_api_key: str = 'EMPTY'
    llm_model_name: str = 'qwen2.5:7b-instruct'

    embedding_base_url: str = 'http://127.0.0.1:11434/v1'
    embedding_api_key: str = 'EMPTY'
    embedding_model_name: str = 'BAAI/bge-m3'

    reranker_model: str = 'BAAI/bge-reranker-base'

    chroma_collection: str = 'research_chunks'
    chroma_dir: str = './data/chroma'
    docs_dir: str = './data/papers'

    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_rewrite_rounds: int = 2
    request_timeout: int = 120
    bm25_score_threshold: float = 0.1

    min_retrieval_hits_threshold: int = 3
    rerank_margin_threshold: float = 0.05
    dense_bm25_overlap_threshold: int = 3
    rerank_top1_score_threshold: float = 0.3
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
