from fastapi import HTTPException
import pytest

pytest.importorskip('chromadb')
pytest.importorskip('sentence_transformers')

from app.agents.retriever import gate
from app.api import server
from app.schemas.documents import RetrievedChunk
from app.schemas.state import ResearchState, RetrievalResult


class DummyWorkflow:
    def __init__(self, state: ResearchState) -> None:
        self._state = state

    def run(self, _: str) -> ResearchState:
        return self._state


class ErrorWorkflow:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def run(self, _: str) -> ResearchState:
        raise self._error


def test_gate_skips_rerank_metrics_in_fallback_mode():
    # Meets min-hit/overlap thresholds, but fails rerank-specific thresholds.
    result = RetrievalResult(
        final_hits_count=3,
        dense_bm25_overlap=3,
        rerank_margin=0.0,
        rerank_top1_score=0.0,
    )

    valid_with_rerank, _ = gate(result, use_rerank_metrics=True)
    valid_without_rerank, _ = gate(result, use_rerank_metrics=False)

    assert valid_with_rerank is False
    assert valid_without_rerank is True


def test_ask_response_schema_is_consistent_for_success_and_fail(monkeypatch):
    fail_state = ResearchState(user_query='q1', failure_reason='gate failed')
    monkeypatch.setattr(server, 'workflow', DummyWorkflow(fail_state))
    fail_resp = server.ask_question(server.AskRequest(query='q1'))

    success_state = ResearchState(
        user_query='q2',
        draft_answer='ok answer',
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id='docA::p1::c0',
                score=0.9,
                metadata={'source': 'docA.pdf', 'page': 1, 'text': 'snippet'},
            )
        ],
    )
    monkeypatch.setattr(server, 'workflow', DummyWorkflow(success_state))
    success_resp = server.ask_question(server.AskRequest(query='q2'))

    assert set(fail_resp.keys()) == set(success_resp.keys())
    assert fail_resp['status'] == 'fail'
    assert success_resp['status'] == 'ok'
    assert fail_resp['answer'] == 'gate failed'
    assert success_resp['answer'] == 'ok answer'
    assert isinstance(fail_resp['citations'], list)
    assert isinstance(success_resp['citations'], list)


def test_ask_rethrows_http_exception(monkeypatch):
    monkeypatch.setattr(
        server,
        'workflow',
        ErrorWorkflow(HTTPException(status_code=418, detail='teapot')),
    )

    try:
        server.ask_question(server.AskRequest(query='q'))
        raise AssertionError('Expected HTTPException')
    except HTTPException as exc:
        assert exc.status_code == 418
        assert exc.detail == 'teapot'
