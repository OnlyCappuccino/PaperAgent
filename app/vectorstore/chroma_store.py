from __future__ import annotations

from typing import Any
import chromadb

from app.config import get_settings
from app.schemas.documents import DocumentChunk, RetrievedChunk
from app.vectorstore.embeddings import embed_query, embed_texts


class ChromaResearchStore:
    def __init__(self, collection_name: str = None) -> None:
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name or settings.chroma_collection)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                'source': chunk.source,
                'page': chunk.page,
                'chunk_index': chunk.chunk_index,
                'doc_name': chunk.doc_name,
                'section': chunk.section,
            }
            for chunk in chunks
        ]
        embeddings = embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, k: int = 60) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[embed_query(query)],
            n_results=k,
            include=['documents', 'metadatas', 'distances'],
        )
        
        documents = result.get('documents', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]
        distances = result.get('distances', [[]])[0]
        ids = result.get('ids', [[]])[0]

        outputs: list[RetrievedChunk] = []
        for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            outputs.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=float(distance),
                    metadata={
                        'page': meta.get('page'),
                        'text': doc,
                        'source': meta.get('source', ''),
                        'doc_name': meta.get("doc_name", ""),
                        'section': meta.get("section", ""),
                    },
                )
            )
        return outputs
    
    def del_collection(self, collection_name: str | None = None, is_all: bool | None = False) -> None:
        if is_all:
            for collection in self.client.list_collections():
                if collection.name == self.collection.name:
                    continue
                self.client.delete_collection(collection.name)
        else:
            self.client.delete_collection(collection_name)
    
    def rebuild_collection(self, collection_name: str) -> None:
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'}
        )
    
    def switch_collection(self, collection_name: str) -> None:
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'}
        )
    
    def all_collections(self) -> list[str]:
        return [collection.name for collection in self.client.list_collections()]

    # 获取所有chunks
    def get_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        collection_data = self.collection.get(include=['documents', 'metadatas'])
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
        return chunks
