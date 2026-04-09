from __future__ import annotations

from typing import Any
import chromadb

from app.config import get_settings
from app.schemas.documents import DocumentChunk, RetrievedChunk
from app.vectorstore.embeddings import embed_query, embed_texts


class ChromaResearchStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection)

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

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[embed_query(query)],
            n_results=top_k,
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
                    source=meta.get('source', ''),
                    page=meta.get('page'),
                    text=doc,
                    score=float(distance),
                )
            )
        return outputs
