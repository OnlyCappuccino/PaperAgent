from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    page: int | None = None
    chunk_index: int = 0


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    page: int | None = None
    score: float = Field(default=0.0, description='相似度分数或距离')
