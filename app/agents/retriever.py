from app.config import get_settings
from app.schemas.documents import DocumentChunk, RetrievedChunk
from app.vectorstore.chroma_store import ChromaResearchStore
from app.vectorstore.reranker import Reranker, rrf_fuse
from app.vectorstore.BM25_retriever import BM25Retriever

class RetrieverAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaResearchStore()
        self.bm25_retriever = None  # 延迟初始化，等到有数据时再创建
        self.reranker = Reranker()

    def run(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        chunks: list[DocumentChunk] = []

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
            return self.store.search(query=query, k=top_k or self.settings.top_k)

        self.bm25_retriever = BM25Retriever(chunks=chunks)
        bm25_hits = self.bm25_retriever.search(query=query, k=60)
        dense_hits = self.store.search(query=query, k=60)
        

        rrf_fuseed_hits = rrf_fuse(dense_hits, bm25_hits, k=60)
        try:
            # 使用reranker对RRF融合的结果进行重排序
            reranker_hits = self.reranker.rerank(query=query, hits=rrf_fuseed_hits)
            result: list[RetrievedChunk] = []
            for rerank, rrf_fuseed in zip(reranker_hits, rrf_fuseed_hits):
                result_hit = RetrievedChunk(
                    chunk_id=rerank.chunk_id,
                    score=0.85 * rerank.score + 0.15 * rrf_fuseed.score,  # 综合考虑reranker得分和RRF得分
                    metadata=rrf_fuseed.metadata
                )
                result.append(result_hit)
            result.sort(key=lambda x: x.score, reverse=True)
            return result[:top_k]
        except:
            # 如果reranker出问题了，就退回到RRF融合的结果
            rrf_fuseed_hits.sort(key=lambda x: x.score, reverse=True)
            return rrf_fuseed_hits[:top_k]



        
