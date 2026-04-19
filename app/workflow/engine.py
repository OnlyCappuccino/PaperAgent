from app.agents.critic import CriticAgent
from app.agents.retriever import RetrieverAgent
from app.agents.rewriter import RewriterAgent
from app.agents.response_summarizer import SummarizerAgent
from app.agents.session_summarizer import Session_SummaryAgent
from app.config import get_settings
import logging
from app.core.citations import build_citation_records, extract_chunk_ids, strip_citation_block
from app.memory.history_context import build_history_context
from app.memory.redis_memory_store import RedisMemoryStore
from app.schemas.state import ResearchState, SessionSummary

logger = logging.getLogger(__name__)

class ResearchWorkflow:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        self.rewriter = RewriterAgent()
        self.summary = Session_SummaryAgent()
        self.template_memory = RedisMemoryStore()



    def run(self, user_query: str, session_id: str) -> ResearchState:
        # 是否是第一次会话轮次，第一次轮次没有历史上下文，不需要重写用户query
        try:
            has_history  = self.template_memory.session_exists(session_id=session_id)
        except:
            logger.warning(f'[redis]redis连接失败，无法获取会话历史，默认没有历史记录')
            has_history = False
        state = ResearchState(user_query=user_query)
        existing_summary = self.template_memory.get_session_summary(session_id=session_id) 

        if  has_history :
            # 获取最近的会话历史记录，并构建历史上下文文本，供重写器使用
            history_context_turns = self.template_memory.get_recent_turns(session_id=session_id)
            history_context = build_history_context(history_context_turns,existing_summary)
            # 重写用户query
            user_query = self.rewriter.rewrite(user_query, history_context)
        
        # 保存用户query和会话总结到会话历史记录中
        self.template_memory.append_turn(
            session_id=session_id,
            role='user',
            text=state.user_query,
        )
        
        
        state.retrieved_chunks, state.failure_reason = self.retriever.run(user_query, top_k=self.settings.top_k)

        # 没有搜索结果，直接保存失败原因到会话历史记录中，并更新会话总结计数
        if not state.retrieved_chunks:
            self.template_memory.append_turn(
                session_id=session_id,
                role='assistant',
                text=state.failure_reason,
            )
            self.summary_session(session_id=session_id, current_summary= existing_summary)
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

        # 更新会话总结，只有当轮次计数达到阈值时才进行总结，以减少不必要的总结调用
        self.summary_session(session_id=session_id, current_summary= existing_summary)
        
        state.citation_valid = bool(state.citations) and not state.invalid_citation_ids

        return state
    
    def summary_session(self, session_id: str, current_summary: SessionSummary) -> SessionSummary:
        try:
            updated_summary = self.template_memory.increase_summary_count(session_id)
            if updated_summary.count >= self.settings.summary_max_turns:
                recent_turns = self.template_memory.get_recent_turns(session_id=session_id)
                new_summary = self.summary.summarize(
                    existing_summary=current_summary.summary_text,
                    recent_turns_text=recent_turns,
                )
                self.template_memory.set_summary(session_id, summary_text=new_summary)
        except Exception as e:
            logger.warning(f"Failed to summarize session {session_id}: {e}")
