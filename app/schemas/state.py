from pydantic import BaseModel, Field
from app.schemas.documents import RetrievedChunk


class CritiqueResult(BaseModel):
    passed: bool = False
    reason: str = ''
    missing_evidence: list[str] = Field(default_factory=list)
    rewrite_suggestion: str = ''


class CitationRecord(BaseModel):
    chunk_id: str
    source: str = ''
    page: int | None = None
    doc_name: str = ''
    section: str = ''
    text_snippet: str = ''


class ResearchState(BaseModel):
    user_query: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    draft_answer: str = ''
    critique: CritiqueResult | None = None
    rewrite_round: int = 0
    citation_ids: list[str] = Field(default_factory=list)
    citations: list[CitationRecord] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    citation_valid: bool = False
    failure_reason: str = ''

class RetrievalResult(BaseModel):
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    final_hits_count: int = 0
    rerank_top1_score: float = 0.0
    rerank_margin: float = 0.0  #top1和top2的分数差距
    dense_bm25_overlap: int = 0
