"""
redis短期记忆存储
"""

from __future__ import annotations
import json
import logging
import time
import redis
from app.config import get_settings

from app.schemas.state import ConversationTurn, SessionSummary

logger = logging.getLogger(__name__)

class RedisMemoryStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    @staticmethod
    def _turns_key(session_id: str) -> str:
        return f'chat:{session_id}:turns'

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f'chat:{session_id}:meta'

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f'chat:{session_id}:summary'

    def append_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        citations: list[str] | None = None,
        status: str = 'ok',
    ) -> None:
        turn = ConversationTurn(
            role=role,
            text=text,
            citations=citations or [],
            status=status,
            timestamp=time.time(),
        )

        turns_key = self._turns_key(session_id)
        meta_key = self._meta_key(session_id)

        with self.client.pipeline() as pipe:
            # 向列表键turns_key右端添加一条记录
            pipe.rpush(turns_key, turn.model_dump_json())
            # 保持列表长度不超过 memory_max_turns，用负号表示从列表末尾开始计算索引（即保留最新的 memory_max_turns 条记录）
            pipe.ltrim(turns_key, -self.settings.memory_max_turns, -1)
            # 更新会话元数据，包括最近更新时间和轮次计数（轮次计数等于当前列表长度加1，因为新记录尚未添加到列表中）
            pipe.hset(meta_key, 'updated_at', str(turn.timestamp))
            pipe.hincrby(meta_key, 'turn_count', 1)
            # 设置键的过期时间，单位为秒，过期时间从当前时间开始计算，即每次添加新记录都会延长会话的存活时间
            pipe.expire(turns_key, self.settings.session_ttl_seconds)
            pipe.expire(meta_key, self.settings.session_ttl_seconds)
            pipe.execute()


    def get_recent_turns(self, session_id: str, n: int | None = None) -> list[ConversationTurn]:
        limit = n or self.settings.memory_max_turns
        # 获取最近n条记录，lrange的索引是闭区间，所以结束索引是-1表示最后一条记录
        raw_items = self.client.lrange(self._turns_key(session_id), -limit, -1)

        turns: list[ConversationTurn] = []
        for item in raw_items:
            try:
                payload = json.loads(item)
                turns.append(ConversationTurn.model_validate(payload))
            except Exception as e:
                logger.warning(f'获取对话记录失败: {e}')
                continue
        return turns

    def session_exists(self, session_id: str) -> bool:
        return self.client.exists(self._turns_key(session_id)) == 1

    def clear_session(self, session_id: str) -> None:
        self.client.delete(
            self._turns_key(session_id),
            self._meta_key(session_id),
            self._summary_key(session_id),
        )

    def get_session_summary(self, session_id: str) -> SessionSummary:
        raw = self.client.get(self._summary_key(session_id))
        if not raw:
            return SessionSummary()
        try:
            payload = json.loads(raw)
            return SessionSummary.model_validate(payload)
        except Exception:
            logger.warning(f'获取会话摘要失败: {raw}')
            return SessionSummary()
    
    def increase_summary_count(self, session_id: str) -> SessionSummary:
        current = self.get_session_summary(session_id)

        payload = SessionSummary(
            summary_text=current.summary_text,
            updated_at=current.updated_at,
            count=current.count + 1,
        )

        key = self._summary_key(session_id)
        self.client.set(key, payload.model_dump_json())
        self.client.expire(key, self.settings.session_ttl_seconds)
        return payload

    def set_summary(self, session_id: str, summary_text: str) -> None:
        payload = SessionSummary(
            summary_text=summary_text,
            updated_at=time.time(),
            count=0,
        )
        key = self._summary_key(session_id)
        self.client.set(key, payload.model_dump_json())
        self.client.expire(key, self.settings.session_ttl_seconds)



    