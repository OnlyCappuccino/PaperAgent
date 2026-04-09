from app.config import get_settings
from app.ingestion.chunker import build_chunks
from app.ingestion.loaders import load_documents
from app.vectorstore.chroma_store import ChromaResearchStore


def build_index() -> int:
    settings = get_settings()
    # 加载文档内容
    records = load_documents(settings.docs_dir)
    # 将文档切分成chunk
    chunks = build_chunks(
        records=records,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    store = ChromaResearchStore()
    # 将chunk写入向量数据库
    store.upsert_chunks(chunks)
    return len(chunks)
