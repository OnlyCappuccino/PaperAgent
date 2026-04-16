from __future__ import annotations

import re

from app.schemas.documents import RetrievedChunk
from app.schemas.state import CitationRecord


CITATION_BLOCK_PATTERN = re.compile(r"引用证据[:：]\s*(.*)$", re.DOTALL)
LIST_CITATION_PATTERN = re.compile(r"-\s*\[([^\[\]\r\n]+)\]")
GENERIC_BRACKET_PATTERN = re.compile(r"\[([^\[\]\r\n]+)\]")
CITATION_TAIL_PATTERN = re.compile(r"\n*引用证据[:：]\s*.*$", re.DOTALL)



# 抽取引用的 chunk_id 列表，支持两种格式：
# 1. 列表格式：引用证据:\n- [chunk_id1]\n- [chunk_id2]
# 2. 行内格式：引用证据: chunk_id1, chunk_id2
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


# 构建可信引用记录列表和无效引用ID列表
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

# 去除引用块，返回纯文本答案
def strip_citation_block(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    return CITATION_TAIL_PATTERN.sub("", text).strip()

