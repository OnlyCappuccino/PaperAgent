from app.config import get_settings
from app.schemas.state import RetrievalResult
from app.schemas.documents import DocumentChunk, RetrievedChunk
from app.vectorstore.chroma_store import ChromaResearchStore
from app.vectorstore.reranker import Reranker, rrf_fuse
from app.vectorstore.BM25_retriever import BM25Retriever


# 检索结果门控
def gate(retriever_result: RetrievalResult, use_rerank_metrics: bool = True) -> tuple[bool, str]:
    settings = get_settings()
    if retriever_result.final_hits_count == 0:
        return False, '未检索到任何相关证据。'
    if retriever_result.final_hits_count < settings.min_retrieval_hits_threshold:
        return False, f'检索到的相关证据过少（{retriever_result.final_hits_count}条），可能无法支持后续分析。'
    if retriever_result.dense_bm25_overlap < settings.dense_bm25_overlap_threshold:
        return False, f'检索结果过于分散（密集检索和BM25检索的重叠度为{retriever_result.dense_bm25_overlap}），可能包含较多噪声。'
    if use_rerank_metrics:
        # if retriever_result.rerank_margin < settings.rerank_margin_threshold:
        #     return False, f'检索结果的相关性较低（top1和top2的分数差距仅{retriever_result.rerank_margin:.4f}），可能无法支持后续分析。'
        if retriever_result.rerank_top1_score < settings.rerank_top1_score_threshold:
            return False, f'检索结果的相关性较低（top1得分为{retriever_result.rerank_top1_score:.4f}），可能无法支持后续分析。'
    return True, '检索结果通过门控检查。'

class RetrieverAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaResearchStore()
        self.bm25_retriever = None  # 延迟初始化，等到有数据时再创建
        self.reranker = Reranker()

    def run(self, query: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], str]:
        effective_top_k = top_k or self.settings.top_k
        chunks: list[DocumentChunk] = []
        # 构建chunks列表用于BM25检索器的初始化
        collection_data = self.store.collection.get(include=['documents', 'metadatas'])
        documents = collection_data.get('documents') or []
        metadatas = collection_data.get('metadatas') or []
        ids = collection_data.get('ids') or []

        for idx, (chunk_id, doc) in enumerate(zip(ids, documents)):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            meta = metadata or {}
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=meta.get('source', ''),
                    text=doc,
                    page=meta.get('page'),
                    chunk_index=meta.get('chunk_index', 0),
                    doc_name=meta.get('doc_name', ''),
                    section=meta.get('section', ''),
                )
            )

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
            # 使用reranker对RRF融合的结果进行重排序
            reranker_hits = self.reranker.rerank(query=query, hits=rrf_fused_hits)
        except Exception as e:
            print(f'Reranker failed, fallback to RRF: {e}')

        if reranker_hits:
            result_chunks: list[RetrievedChunk] = []
            for rerank in reranker_hits:
                result_hit = RetrievedChunk(
                    chunk_id=rerank.chunk_id,
                    score=0.85 * rerank.score + 0.15 * rrf_fused_hit_scores.get(rerank.chunk_id, 0.0),  # 综合考虑reranker得分和RRF得分
                    metadata= rerank.metadata
                )
                result_chunks.append(result_hit)
            result_chunks.sort(key=lambda x: x.score, reverse=True)

            result = build_result(result_chunks)
            valid, reason = gate(result)
            if valid:
                return result.retrieved_chunks, reason
            print(f'Gate check failed: {reason}')
            return [], reason

        # reranker 不可用时，回退到 RRF 结果，但仍执行门控（不使用 rerank 指标）
        rrf_fused_hits.sort(key=lambda x: x.score, reverse=True)
        fallback_result = build_result(rrf_fused_hits)
        valid, reason = gate(fallback_result, use_rerank_metrics=False)
        if valid:
            return fallback_result.retrieved_chunks, reason
        print(f'Gate check failed: {reason}')
        return [], reason



        
