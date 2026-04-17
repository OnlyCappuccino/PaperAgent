"""
会话总结agent
"""



from app.config import get_settings
from app.llm.client import LocalOpenAIClient
from app.memory.history_context import build_history_context
from app.prompts.templates import session_summary_messages
from app.schemas.state import ConversationTurn, SessionSummary

class Session_SummaryAgent:
    def __init__(self) -> None:
        self.client = LocalOpenAIClient()
        self.settings = get_settings()

    def summarize(self, existing_summary: str, recent_turns_text: list[ConversationTurn]) -> str:
        max_turns = self.settings.summary_max_turns
        history_context = build_history_context(recent_turns_text, SessionSummary(), max_turns=max_turns)
        messages = session_summary_messages(existing_summary, history_context)
        response = self.client.chat(messages, temperature=0.2)
        return response.strip()