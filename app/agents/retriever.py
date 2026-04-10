from app.config import get_settings
from app.schemas.documents import DocumentChunk, RetrievedChunk
from app.vectorstore.chroma_store import ChromaResearchStore
from app.vectorstore.reranker import rrf_fuse
from app.vectorstore.BM25_retriever import BM25Retriever

class RetrieverAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaResearchStore()
        self.bm25_retriever = None  # 延迟初始化，等到有数据时再创建

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

        rrf_fuseed = rrf_fuse(dense_hits, bm25_hits, top_k=top_k or self.settings.top_k)
        return rrf_fuseed
