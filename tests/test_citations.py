from app.core.citations import build_citation_records, extract_chunk_ids
from app.schemas.documents import RetrievedChunk


def test_extract_chunk_ids_from_citation_block():
    answer = """
    这是正文。

    引用证据：
    - [paperA::p3::c2]
    - [paperB::p7::c0]
    """
    assert extract_chunk_ids(answer) == ["paperA::p3::c2", "paperB::p7::c0"]


def test_extract_chunk_ids_with_path_style_ids():
    answer = """
    引用证据：
    - [C:/Users/Cappuccino/Desktop/实习项目/local_multi_agent_research_assistant/data/papers/a.pdf::p1::c0]
    - [D:\\docs\\b.pdf::p2::c3]
    """
    assert extract_chunk_ids(answer) == [
        "C:/Users/Cappuccino/Desktop/实习项目/local_multi_agent_research_assistant/data/papers/a.pdf::p1::c0",
        "D:\\docs\\b.pdf::p2::c3",
    ]


def test_build_citation_records_separates_invalid_ids():
    chunks = [
        RetrievedChunk(
            chunk_id="paperA::p3::c2",
            score=0.8,
            metadata={"source": "paperA.pdf", "page": 3, "text": "snippet A"},
        )
    ]

    valid, invalid = build_citation_records(
        citation_ids=["paperA::p3::c2", "paperX::p1::c0"],
        retrieved_chunks=chunks,
    )

    assert len(valid) == 1
    assert valid[0].chunk_id == "paperA::p3::c2"
    assert invalid == ["paperX::p1::c0"]
