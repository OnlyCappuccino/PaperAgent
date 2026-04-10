from __future__ import annotations

from app.schemas.documents import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return '无可用证据片段。'

    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        page_info = f"第{chunk.metadata.get('page')}页" if chunk.metadata.get('page') else '页码未知'
        lines.append(f"[证据{idx}] 来源: {chunk.metadata.get('source', '')} | {page_info}\n{chunk.metadata.get('text', '')}")
    return '\n\n'.join(lines)


def summarizer_messages(user_query: str, chunks: list[RetrievedChunk], rewrite_hint: str = '') -> list[dict[str, str]]:
    hint_text = f'\n额外修订要求：{rewrite_hint}' if rewrite_hint else ''
    return [
        {
            'role': 'system',
            'content': (
                '你是一个严谨的科研文献总结助手。'
                '只能根据提供的证据回答，不允许编造。'
                '请尽量输出结构化内容：背景、方法、结果、局限、结论。'
                '若证据不足，请明确说明证据不足。'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'用户问题：{user_query}\n\n'
                f'证据片段：\n{format_context(chunks)}\n'
                f'{hint_text}\n\n'
                '请基于证据作答，并在末尾附一个“引用证据”小节，列出你使用了哪些证据编号。'
            ),
        },
    ]


def critic_messages(user_query: str, answer: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': (
                '你是一个严格的答案审查员。'
                '请判断回答是否完全基于证据片段。'
                '输出必须是 JSON，字段包括：'
                'passed(boolean), reason(string), missing_evidence(list[string]), rewrite_suggestion(string)。'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'用户问题：{user_query}\n\n'
                f'候选答案：\n{answer}\n\n'
                f'证据片段：\n{format_context(chunks)}\n\n'
                '请检查：\n'
                '1. 是否有无依据的结论\n'
                '2. 是否遗漏关键信息\n'
                '3. 是否存在夸大或错误归纳\n'
                '只输出 JSON。'
            ),
        },
    ]
