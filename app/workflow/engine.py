from app.agents.critic import CriticAgent
from app.agents.retriever import RetrieverAgent
from app.agents.rewriter import Rewriter
from app.agents.summarizer import SummarizerAgent
from app.config import get_settings
from app.core.citations import build_citation_records, extract_chunk_ids, strip_citation_block
from app.memory.history_context import build_history_context
from app.memory.redis_memory_store import RedisMemoryStore
from app.schemas.state import ResearchState


class ResearchWorkflow:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        self.rewriter = Rewriter()
        self.template_memory = RedisMemoryStore()

    def run(self, user_query: str, session_id: str) -> ResearchState:
        # 是否是第一次会话轮次，第一次轮次没有历史上下文，不需要重写用户query
        not_first_turn = self.template_memory.session_exists(session_id=session_id)
        state = ResearchState(user_query=user_query)

        if  not_first_turn:
            # 获取最近的会话历史记录，并构建历史上下文文本，供重写器使用
            history_context_turns = self.template_memory.get_recent_turns(session_id=session_id)
            history_context = build_history_context(history_context_turns)
            # 重写用户query
            user_query = self.rewriter.rewrite(user_query, history_context)
        
        # 保存用户query到会话历史记录中
        self.template_memory.append_turn(
            session_id=session_id,
            role='user',
            text=state.user_query,
        )
        
        
        
        state.retrieved_chunks, state.failure_reason = self.retriever.run(user_query, top_k=self.settings.top_k)

        if not state.retrieved_chunks:
            self.template_memory.append_turn(
                session_id=session_id,
                role='assistant',
                text=state.failure_reason,
            )
            return state
        
        rewrite_hint = ''
        for round_id in range(self.settings.max_rewrite_rounds + 1):
            state.rewrite_round = round_id
            state.draft_answer = self.summarizer.run(
                user_query=user_query,
                chunks=state.retrieved_chunks,
                rewrite_hint=rewrite_hint,
            )
            state.critique = self.critic.run(
                user_query=user_query,
                answer=state.draft_answer,
                chunks=state.retrieved_chunks,
            )
            if state.critique.passed:
                break
            rewrite_hint = state.critique.rewrite_suggestion
        # 引用检查（引用片段是否为模型幻觉生成）
        state.citation_ids = extract_chunk_ids(state.draft_answer)
        state.citations, state.invalid_citation_ids = build_citation_records(
            citation_ids=state.citation_ids,
            retrieved_chunks=state.retrieved_chunks,
        )

        # 删除引用块，得到纯文本答案，方便后续保存和展示
        state.draft_answer = strip_citation_block(state.draft_answer)

        # 将最终的回答保存到会话历史记录中，并标注引用的chunk_id列表
        self.template_memory.append_turn(
            session_id=session_id,
            role='assistant',
            text=state.draft_answer,
            citations=[chunk.chunk_id for chunk in state.citations],
        )
        state.citation_valid = bool(state.citations) and not state.invalid_citation_ids

        return state
