import json
import re
from typing import Any


JSON_BLOCK_PATTERN = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """尽量从模型输出中提取 JSON。

    支持三种情况：
    1. 纯 JSON
    2. ```json ... ``` 包裹
    3. 文本中夹带一个大括号对象
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    block_match = JSON_BLOCK_PATTERN.search(text)
    if block_match:
        block = block_match.group(1)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    first = text.find('{')
    last = text.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        return json.loads(candidate)

    raise ValueError('未能从模型输出中解析 JSON')
