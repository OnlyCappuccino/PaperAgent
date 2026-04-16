
from app.config import get_settings
from app.llm.client import LocalOpenAIClient
from app.prompts.templates import rewriter_messages


class Rewriter:
    def __init__(self) -> None:
        self.client = LocalOpenAIClient()

    def rewrite(self, user_query: str, history_context: str) -> str:
        messages = rewriter_messages(user_query=user_query, history_context=history_context)
        return self.client.chat(messages=messages, temperature=0.2)
        