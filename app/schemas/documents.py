from typing import Any, Dict

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    page: int | None = None
    chunk_index: int = 0
    doc_name: str = ""
    section: str = ""


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float = Field(default=0.0, description='相似度分数或距离')
    metadata: Dict[str, Any]  # 放置text、source、page、doc_name、section等信息

