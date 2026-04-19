import logging

from app.config import get_settings
from app.schemas.documents import DocumentChunk, RetrievedChunk
from app.schemas.state import RetrievalResult
from app.vectorstore.BM25_retriever import BM25Retriever
from app.vectorstore.chroma_store import ChromaResearchStore
from app.vectorstore.reranker import Reranker, rrf_fuse

logger = logging.getLogger(__name__)


def gate(retriever_result: RetrievalResult, use_rerank_metrics: bool = True) -> tuple[bool, str]:
    settings = get_settings()
    if retriever_result.final_hits_count == 0:
        return False, "未检索到任何相关证据。"
    if retriever_result.final_hits_count < settings.min_retrieval_hits_threshold:
        return (
            False,
            f"检索到的相关证据过少（{retriever_result.final_hits_count}条），可能无法支持后续分析。",
        )
    if retriever_result.dense_bm25_overlap < settings.dense_bm25_overlap_threshold:
        return (
            False,
            f"检索结果过于分散（dense 与 BM25 的重叠为 {retriever_result.dense_bm25_overlap}），可能噪声较多。",
        )
    if use_rerank_metrics:
        if retriever_result.rerank_top1_score < settings.rerank_top1_score_threshold:
            return (
                False,
                f"检索结果相关性偏低（top1={retriever_result.rerank_top1_score:.4f}），可能无法支持后续分析。",
            )
    return True, "检索结果通过门控检查。"


class RetrieverAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaResearchStore()
        self.bm25_retriever: BM25Retriever | None = None
        # 懒加载，避免 API 启动时阻塞在模型下载
        self.reranker: Reranker | None = None

    def _get_reranker(self) -> Reranker | None:
        if self.reranker is not None:
            return self.reranker
        try:
            self.reranker = Reranker()
            return self.reranker
        except Exception as e:
            logger.warning(f"Reranker 初始化失败，回退到 RRF: {e}")
            return None

    def run(self, query: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], str]:
        effective_top_k = top_k or self.settings.top_k
        chunks: list[DocumentChunk] = self.store.get_chunks()

        if not chunks:
            _, reason = gate(RetrievalResult(), use_rerank_metrics=False)
            return [], reason

        self.bm25_retriever = BM25Retriever(chunks=chunks)
        bm25_hits = self.bm25_retriever.search(query=query, k=60)
        dense_hits = self.store.search(query=query, k=60)

        dense_ids = {hit.chunk_id for hit in dense_hits}
        bm25_ids = {hit.chunk_id for hit in bm25_hits}
        dense_bm25_overlap = len(dense_ids & bm25_ids)

        rrf_fused_hits = rrf_fuse(dense_hits, bm25_hits, k=60)
        rrf_fused_hit_scores = {hit.chunk_id: hit.score for hit in rrf_fused_hits}

        def build_result(hits: list[RetrievedChunk]) -> RetrievalResult:
            limited_hits = hits[:effective_top_k]
            top1_score = limited_hits[0].score if limited_hits else 0.0
            margin = limited_hits[0].score - limited_hits[1].score if len(limited_hits) > 1 else 0.0
            return RetrievalResult(
                retrieved_chunks=limited_hits,
                final_hits_count=len(limited_hits),
                rerank_top1_score=top1_score,
                rerank_margin=margin,
                dense_bm25_overlap=dense_bm25_overlap,
            )

        reranker_hits: list[RetrievedChunk] = []
        try:
            reranker = self._get_reranker()
            if reranker is not None:
                reranker_hits = reranker.rerank(query=query, hits=rrf_fused_hits)
        except Exception as e:
            logger.warning(f"Reranker 重排序失败，回退到 RRF: {e}")

        if reranker_hits:
            result_chunks: list[RetrievedChunk] = []
            for rerank in reranker_hits:
                result_chunks.append(
                    RetrievedChunk(
                        chunk_id=rerank.chunk_id,
                        score=0.85 * rerank.score + 0.15 * rrf_fused_hit_scores.get(rerank.chunk_id, 0.0),
                        metadata=rerank.metadata,
                    )
                )
            result_chunks.sort(key=lambda x: x.score, reverse=True)

            result = build_result(result_chunks)
            valid, reason = gate(result)
            if valid:
                return result.retrieved_chunks, reason
            logger.debug(f"Gate check failed: {reason}")
            return [], reason

        # reranker 不可用时，回退到 RRF 结果
        rrf_fused_hits.sort(key=lambda x: x.score, reverse=True)
        fallback_result = build_result(rrf_fused_hits)
        valid, reason = gate(fallback_result, use_rerank_metrics=False)
        if valid:
            return fallback_result.retrieved_chunks, reason
        logger.debug(f"Gate check failed: {reason}")
        return [], reason
