import logging

from app.llm.client import LocalOpenAIClient
from app.prompts.templates import rewriter_messages

logger = logging.getLogger(__name__)


class RewriterAgent:
    def __init__(self) -> None:
        self.client = LocalOpenAIClient()

    def rewrite(self, user_query: str, history_context: str) -> str:
        messages = rewriter_messages(user_query=user_query, history_context=history_context)
        try:
            rewritten = self.client.chat(messages=messages, temperature=0.2)
            if not rewritten:
                logger.warning("[LLM] rewriter returned empty content, fallback to original query")
                return user_query
            return rewritten
        except Exception as e:
            logger.warning(f"[LLM] rewriter failed, fallback to original query: {e}")
            return user_query
