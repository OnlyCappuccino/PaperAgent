"""
会话上下文构建
"""

from app.config import get_settings
from app.schemas.state import ConversationTurn


def build_history_context(turns: list[ConversationTurn], max_turns: int = None) -> str:
    max_turns = max_turns or get_settings().history_turns_display
    selected = turns[-max_turns:]
    lines: list[str] = []

    for turn in selected:
        role = '用户' if turn.role == 'user' else '助手'
        lines.append(f'{role}: {turn.text}')

    return '\n'.join(lines)
