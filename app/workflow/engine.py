from app.agents.critic import CriticAgent
from app.agents.retriever import RetrieverAgent
from app.agents.summarizer import SummarizerAgent
from app.config import get_settings
from app.schemas.state import ResearchState


class ResearchWorkflow:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()

    def run(self, user_query: str) -> ResearchState:
        state = ResearchState(user_query=user_query)
        state.retrieved_chunks = self.retriever.run(user_query)

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

        return state
