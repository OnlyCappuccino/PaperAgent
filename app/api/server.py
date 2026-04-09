from fastapi import FastAPI
from pydantic import BaseModel

from app.workflow.engine import ResearchWorkflow
from app.workflow.indexing import build_index

app = FastAPI(title='Local Multi-Agent Research Assistant')
workflow = ResearchWorkflow()


class AskRequest(BaseModel):
    query: str


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@app.post('/index')
def index_documents() -> dict:
    count = build_index()
    return {'indexed_chunks': count}


@app.post('/ask')
def ask_question(request: AskRequest) -> dict:
    state = workflow.run(request.query)
    return {
        'query': state.user_query,
        'answer': state.draft_answer,
        'rewrite_round': state.rewrite_round,
        'critique': state.critique.model_dump() if state.critique else None,
        'retrieved_chunks': [chunk.model_dump() for chunk in state.retrieved_chunks],
    }
