import jieba
from rank_bm25 import BM25Okapi
from app.config import get_settings
from app.schemas.documents import DocumentChunk, RetrievedChunk
def tokenize(text: str) -> list[str]:
    return [w.strip().lower() for w in jieba.cut(text) if w.strip()] 


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk]):
        self.settings = get_settings()
        self.chunks = chunks
        self.tokenized_docs = [tokenize(doc.text) for doc in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def search(self, query: str, k: int = 60) -> list[RetrievedChunk]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        
        results: list[RetrievedChunk] = []
        for i in range(min(k, len(scored_docs))):
            if scored_docs[i][1] < self.settings.bm25_score_threshold:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=scored_docs[i][0].chunk_id,
                    score=scored_docs[i][1],
                    metadata={
                        'doc_name': scored_docs[i][0].doc_name,
                        'page': scored_docs[i][0].page,
                        'section': scored_docs[i][0].section,
                        'source': scored_docs[i][0].source,
                        'text': scored_docs[i][0].text,
                    }
                )
            
            )
        return results



    


