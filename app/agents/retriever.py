from app.config import get_settings
from app.schemas.state import RetrievalResult
from app.schemas.documents import DocumentChunk, RetrievedChunk
from app.vectorstore.chroma_store import ChromaResearchStore
from app.vectorstore.reranker import Reranker, rrf_fuse
from app.vectorstore.BM25_retriever import BM25Retriever


# 检索结果门控
def gate(retriever_result: RetrievalResult) -> tuple[bool, str]:
    settings = get_settings()
    if retriever_result.final_hits_count == 0:
        return False, '未检索到任何相关证据。'
    if retriever_result.final_hits_count < settings.min_retrieval_hits_threshold:
        return False, f'检索到的相关证据过少（{retriever_result.final_hits_count}条），可能无法支持后续分析。'
    if retriever_result.rerank_margin < settings.rerank_margin_threshold:
        return False, f'检索结果的相关性较低（top1和top2的分数差距仅{retriever_result.rerank_margin:.4f}），可能无法支持后续分析。'
    if retriever_result.dense_bm25_overlap < settings.dense_bm25_overlap_threshold:
        return False, f'检索结果过于分散（密集检索和BM25检索的重叠度为{retriever_result.dense_bm25_overlap}），可能包含较多噪声。'
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
        chunks: list[DocumentChunk] = []
        result_chunks: list[RetrievedChunk] = []
        result: RetrievalResult = RetrievalResult()
        # 构建chunks列表用于BM25检索器的初始化
        collection_data = self.store.collection.get(include=['documents', 'metadatas'])
        documents = collection_data.get('documents') or []
        metadatas = collection_data.get('metadatas') or []
        ids = collection_data.get('ids') or []

        for chunk_id, doc, metadata in zip(ids, documents, metadatas):
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
            return self.store.search(query=query, k=top_k or self.settings.top_k), ""

        self.bm25_retriever = BM25Retriever(chunks=chunks)
        bm25_hits = self.bm25_retriever.search(query=query, k=60)
        dense_hits = self.store.search(query=query, k=60)
        

        rrf_fuseed_hits = rrf_fuse(dense_hits, bm25_hits, k=60)
        rrf_fuseed_hits_score = {hit.chunk_id: hit.score for hit in rrf_fuseed_hits}
        try:
            # 使用reranker对RRF融合的结果进行重排序
            reranker_hits = self.reranker.rerank(query=query, hits=rrf_fuseed_hits)
            for rerank in reranker_hits:
                result_hit = RetrievedChunk(
                    chunk_id=rerank.chunk_id,
                    score=0.85 * rerank.score + 0.15 * rrf_fuseed_hits_score.get(rerank.chunk_id, 0.0),  # 综合考虑reranker得分和RRF得分
                    metadata= rerank.metadata
                )
                result_chunks.append(result_hit)
            result_chunks.sort(key=lambda x: x.score, reverse=True)
            result = RetrievalResult(
                retrieved_chunks=result_chunks[:top_k],
                rerank_top1_score=result_chunks[0].score,
                rerank_margin=result_chunks[0].score - result_chunks[1].score if len(result_chunks) > 1 else 0.0,
                dense_bm25_overlap=len(set(hit.chunk_id for hit in dense_hits) & set(hit.chunk_id for hit in bm25_hits))
            )
            result.final_hits_count = len(result.retrieved_chunks)
            valid, reason = gate(result)
            if valid:
                return result.retrieved_chunks, reason
            else:
                print(f"Gate check failed: {reason}")
                return [], reason
        except:
            # 如果reranker出问题了，就退回到RRF融合的结果
            rrf_fuseed_hits.sort(key=lambda x: x.score, reverse=True)
            result = RetrievalResult(
                retrieved_chunks=rrf_fuseed_hits[:top_k],
                rerank_top1_score=rrf_fuseed_hits[0].score,
                final_hits_count=len(rrf_fuseed_hits),
                rerank_margin=rrf_fuseed_hits[0].score - rrf_fuseed_hits[1].score if len(rrf_fuseed_hits) > 1 else 0.0,
                dense_bm25_overlap=len(set(hit.chunk_id for hit in dense_hits) & set(hit.chunk_id for hit in bm25_hits))
            )
            return result.retrieved_chunks, ""



        
