from app.core.json_utils import extract_json_object
from app.llm.client import LocalOpenAIClient
from app.prompts.templates import critic_messages
from app.schemas.documents import RetrievedChunk
from app.schemas.state import CritiqueResult


class CriticAgent:
    def __init__(self) -> None:
        self.client = LocalOpenAIClient()

    def run(self, user_query: str, answer: str, chunks: list[RetrievedChunk]) -> CritiqueResult:
        messages = critic_messages(user_query=user_query, answer=answer, chunks=chunks)
        raw = self.client.chat(messages=messages, temperature=0.0)

        try:
            payload = extract_json_object(raw)
            return CritiqueResult.model_validate(payload)
        except Exception:
            # TODO: 后续你可以在这里增加更稳健的兜底逻辑，例如：
            # 1. 再请求一次“只返回 JSON”
            # 2. 用规则解析布尔字段
            # 3. 按文本启发式生成 critique 结果
            return CritiqueResult(
                passed=False,
                reason='Critic 输出解析失败，默认进入保守重写流程。',
                missing_evidence=[],
                rewrite_suggestion='请严格基于证据片段重写，并删除无法直接支持的结论。',
            )
