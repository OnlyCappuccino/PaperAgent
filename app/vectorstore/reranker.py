from collections import defaultdict

from pydantic import BaseModel

from app.schemas.documents import RetrievedChunk





def rrf_fuse(dense_hits: list[RetrievedChunk], bm25_hits: list[RetrievedChunk], k: int = 60, top_k: int = 20) -> list[RetrievedChunk]:
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

    fused_hits = sorted(rrg_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    result: list[RetrievedChunk] = []
    for id, score in fused_hits:
        chunk = d_id[id]
        hit = RetrievedChunk(chunk_id=id, score=score, metadata=chunk.metadata)
        result.append(hit)
    return result
