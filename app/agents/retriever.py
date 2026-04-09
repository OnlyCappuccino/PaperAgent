from app.config import get_settings
from app.schemas.documents import RetrievedChunk
from app.vectorstore.chroma_store import ChromaResearchStore


class RetrieverAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaResearchStore()

    def run(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.store.search(query=query, top_k=top_k or self.settings.top_k)
