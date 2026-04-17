"""
会话上下文构建
"""

from app.config import get_settings
from app.schemas.state import ConversationTurn, SessionSummary


def build_history_context(
    turns: list[ConversationTurn],
    session_summary: SessionSummary,
    max_turns: int | None = None,
) -> str:
    max_turns = max_turns or get_settings().history_turns_display
    selected = turns[-max_turns:]
    lines: list[str] = []

    if session_summary.summary_text:
        lines.append(f'会话摘要: {session_summary.summary_text}')
        
    for turn in selected:
        role = '用户' if turn.role == 'user' else '助手'
        lines.append(f'{role}: {turn.text}')

    return '\n'.join(lines)


