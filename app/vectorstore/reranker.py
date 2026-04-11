from collections import defaultdict
import math

from app.config import get_settings
from app.schemas.documents import RetrievedChunk
from sentence_transformers import CrossEncoder





def rrf_fuse(dense_hits: list[RetrievedChunk], bm25_hits: list[RetrievedChunk], k: int = 60) -> list[RetrievedChunk]:
    """
    RRF score = sum(1 / (k + rank))
    rank 从 1 开始；k 常用 60
    """
    rrg_scores = defaultdict(float)
    d_id = {}
    for rank, hit in enumerate(dense_hits[:k], start=1):
        rrg_scores[hit.chunk_id] += 1 / (k + rank)
        d_id.setdefault(hit.chunk_id, hit)
    
    for rank, hit in enumerate(bm25_hits[:k], start=1):
        rrg_scores[hit.chunk_id] += 1 / (k + rank)
        d_id.setdefault(hit.chunk_id, hit)

    fused_hits = sorted(rrg_scores.items(), key=lambda x: x[1], reverse=True)

    result: list[RetrievedChunk] = []
    for id, score in fused_hits:
        chunk = d_id[id]
        hit = RetrievedChunk(chunk_id=id, score=score, metadata=chunk.metadata)
        result.append(hit)
    return result

class Reranker:
    def __init__(self, reranker_model: str = None):
        self.setting = get_settings()
        self.reranker_model = CrossEncoder(reranker_model or self.setting.reranker_model)

    @staticmethod
    def sigmoid(score: float) -> float:
        return 1 / (1 + math.exp(-score))
    
    def rerank(self, query: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not hits:
            return []
        pairs = [(query, hit.metadata['text']) for hit in hits]
        scores = self.reranker_model.predict(pairs)
        result: list[RetrievedChunk] = []
        for hit, score in zip(hits, scores):
            hit_chunk = RetrievedChunk(
                chunk_id=hit.chunk_id,
                score=self.sigmoid(score),
                metadata=hit.metadata
            )
            result.append(hit_chunk)
        result.sort(key=lambda x: x.score, reverse=True)
        return result



