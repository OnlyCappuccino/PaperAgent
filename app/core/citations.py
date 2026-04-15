from __future__ import annotations

import re

from app.schemas.documents import RetrievedChunk
from app.schemas.state import CitationRecord


CITATION_BLOCK_PATTERN = re.compile(r"引用证据[:：]\s*(.*)$", re.DOTALL)
LIST_CITATION_PATTERN = re.compile(r"-\s*\[([^\[\]\r\n]+)\]")
GENERIC_BRACKET_PATTERN = re.compile(r"\[([^\[\]\r\n]+)\]")


def extract_chunk_ids(answer: str) -> list[str]:
    text = (answer or "").strip()
    if not text:
        return []

    block_match = CITATION_BLOCK_PATTERN.search(text)
    target = block_match.group(1) if block_match else text
    ids = LIST_CITATION_PATTERN.findall(target)
    if not ids:
        ids = GENERIC_BRACKET_PATTERN.findall(target)
    cleaned = [cid.strip() for cid in ids if cid and cid.strip()]
    return list(dict.fromkeys(cleaned))


def build_citation_records(
    citation_ids: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> tuple[list[CitationRecord], list[str]]:
    evidence_map = {
        chunk.chunk_id: CitationRecord(
            chunk_id=chunk.chunk_id,
            source=chunk.metadata.get("source", ""),
            page=chunk.metadata.get("page"),
            doc_name=chunk.metadata.get("doc_name", ""),
            section=chunk.metadata.get("section", ""),
            text_snippet=chunk.metadata.get("text", ""),
        )
        for chunk in retrieved_chunks
    }

    valid = [evidence_map[cid] for cid in citation_ids if cid in evidence_map]
    invalid = [cid for cid in citation_ids if cid not in evidence_map]
    return valid, invalid


def build_evidence_map(retrieved_chunks: list[RetrievedChunk]) -> dict[str, CitationRecord]:
    return {
        chunk.chunk_id: CitationRecord(
            chunk_id=chunk.chunk_id,
            source=chunk.metadata.get("source", ""),
            page=chunk.metadata.get("page"),
            doc_name=chunk.metadata.get("doc_name", ""),
            section=chunk.metadata.get("section", ""),
            text_snippet=chunk.metadata.get("text", ""),
        )
        for chunk in retrieved_chunks
    }
