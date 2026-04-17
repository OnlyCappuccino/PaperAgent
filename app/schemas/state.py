from pydantic import BaseModel, Field
from app.schemas.documents import RetrievedChunk

# 评估结果模型
class CritiqueResult(BaseModel):
    passed: bool = False
    reason: str = ''
    missing_evidence: list[str] = Field(default_factory=list)
    rewrite_suggestion: str = ''

# 引用记录模型
class CitationRecord(BaseModel):
    chunk_id: str
    source: str = ''
    page: int | None = None
    doc_name: str = ''
    section: str = ''
    text_snippet: str = ''

# 搜索结果模型
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

# 检索结果模型
class RetrievalResult(BaseModel):
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    final_hits_count: int = 0
    rerank_top1_score: float = 0.0
    rerank_margin: float = 0.0  #top1和top2的分数差距
    dense_bm25_overlap: int = 0

# API请求和响应模型
class AskRequest(BaseModel):
    query: str
    session_id: str | None = None

# 索引请求和响应模型
class IndexRequest(BaseModel):
    collection: str | None = None
    clear: bool = False
    rebuild: bool = False

# 会话轮次模型
class ConversationTurn(BaseModel):
    role: str
    text: str
    citations: list[str] = Field(default_factory=list)
    status: str = 'ok'
    timestamp: float

class SessionSummary(BaseModel):
    summary_text: str = ''
    updated_at: float = 0.0
    count: int = 0  # session轮次
