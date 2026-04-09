from pydantic import BaseModel, Field
from app.schemas.documents import RetrievedChunk


class CritiqueResult(BaseModel):
    passed: bool = False
    reason: str = ''
    missing_evidence: list[str] = Field(default_factory=list)
    rewrite_suggestion: str = ''


class ResearchState(BaseModel):
    user_query: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    draft_answer: str = ''
    critique: CritiqueResult | None = None
    rewrite_round: int = 0
