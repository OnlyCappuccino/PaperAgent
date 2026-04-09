from app.llm.client import LocalOpenAIClient
from app.prompts.templates import summarizer_messages
from app.schemas.documents import RetrievedChunk


class SummarizerAgent:
    def __init__(self) -> None:
        self.client = LocalOpenAIClient()

    def run(self, user_query: str, chunks: list[RetrievedChunk], rewrite_hint: str = '') -> str:
        messages = summarizer_messages(user_query=user_query, chunks=chunks, rewrite_hint=rewrite_hint)
        return self.client.chat(messages=messages, temperature=0.2)
