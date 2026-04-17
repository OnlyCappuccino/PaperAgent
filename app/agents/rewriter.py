
import logging
from app.config import get_settings
from app.llm.client import LocalOpenAIClient
from app.prompts.templates import rewriter_messages

logger = logging.getLogger(__name__)   

class RewriterAgent:
    def __init__(self) -> None:
        self.client = LocalOpenAIClient()

    def rewrite(self, user_query: str, history_context: str) -> str:
        messages = rewriter_messages(user_query=user_query, history_context=history_context)
        try:
            return self.client.chat(messages=messages, temperature=0.2)
        except:
            logger.warning(f'[LLM]LLM模型启动失败')