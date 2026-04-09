from __future__ import annotations

from app.schemas.documents import DocumentChunk


def sliding_window_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap 必须小于 chunk_size')

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step

    return chunks


def build_chunks(records: list[dict], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    results: list[DocumentChunk] = []

    for record in records:
        parts = sliding_window_chunk(
            text=record['text'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for idx, part in enumerate(parts):
            chunk_id = f"{record['source']}::p{record['page']}::c{idx}"
            results.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=record['source'],
                    text=part,
                    page=record.get('page'),
                    chunk_index=idx,
                )
            )
    return results
