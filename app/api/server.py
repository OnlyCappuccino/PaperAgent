from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Path
from pydantic import BaseModel

from app.core.citations import build_evidence_map
from app.core.logging import setup_logging
from app.workflow.engine import ResearchWorkflow
from app.workflow.indexing import build_index
from app.schemas.state import AskRequest, IndexRequest, ResearchState
from app.workflow.value import Evaluator

workflow : ResearchWorkflow | None = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow
    # 启动时执行
    setup_logging()
    print("startup")
    workflow = ResearchWorkflow()
    if not workflow.summarizer.client:
        raise RuntimeError("未检测到可用的语言模型客户端，请检查配置")
    yield
    print("shutdown")
    


app = FastAPI(title='Local Multi-Agent Research Assistant', lifespan=lifespan)







@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@app.post('/index')
def index_documents(request: IndexRequest) -> dict:
    count = build_index(request)
    return {'indexed_chunks': count}


@app.post('/switch_collection')
def switch_collection(collection_name: str) -> dict:
    if not workflow:
        raise HTTPException(status_code=503, detail='workflow 未初始化')
    workflow.retriever.store.switch_collection(collection_name)
    return {'status': 'ok'}

@app.post('/all_collections')
def get_all_collections() -> dict:
    if not workflow:
        raise HTTPException(status_code=503, detail='workflow 未初始化')
    collections = workflow.retriever.store.all_collections()
    return {'collections': collections}


@app.post('/ask')
def ask_question(request: AskRequest) -> dict:
    if not workflow:
        raise HTTPException(status_code=503, detail='workflow 未初始化')

    def build_ask_response(state: ResearchState, status: str) -> dict:
        evidence_map = {
            chunk_id: citation.model_dump()
            for chunk_id, citation in build_evidence_map(state.retrieved_chunks).items()
        }
        return {
            "session_id": sid,
            'status': status,
            'query': state.user_query,
            'answer': state.draft_answer if status == 'ok' else state.failure_reason,
            'reason': state.failure_reason,
            'rewrite_round': state.rewrite_round,
            'critique': state.critique.model_dump() if state.critique else None,
            'retrieved_chunks': [chunk.model_dump() for chunk in state.retrieved_chunks],
            'citation_ids': state.citation_ids,
            'citation_valid': state.citation_valid,
            'citations': [citation.model_dump() for citation in state.citations],
            'invalid_citation_ids': state.invalid_citation_ids,
            'evidence_map': evidence_map,
        }
    sid = request.session_id or str(uuid4())
    try:
        state = workflow.run(request.query, session_id=sid)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f'模型服务不可用或 embedding 服务异常: {e}'
        )

    status = 'fail' if (not state.retrieved_chunks and state.failure_reason) else 'ok'
    return build_ask_response(state, status=status)


@app.get('/get_chunks')
def get_chunks() -> dict:
    if not workflow:
        raise HTTPException(status_code=503, detail='workflow 未初始化')
    chunks = workflow.retriever.store.get_chunks()
    return {'chunks': chunks}

@app.get('/system_evaluation')
def system_evaluation():
    if not workflow:
        raise HTTPException(status_code=503, detail='workflow 未初始化')
    evaluator = Evaluator(workflow=workflow)
    state = evaluator.evaluate_system()
    return (
        f"Hit Rate: {state.hit_rate * 100:.2f}%"
        f"\nRecall@K: {state.recall_at_k * 100:.2f}%"
        f"\nMRR@K: {state.mrr_at_k:.2f}"
        f"\nPrecision@K: {state.precision_at_k * 100:.2f}%"
    )