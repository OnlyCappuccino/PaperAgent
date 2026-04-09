from __future__ import annotations

from typing import Any
import requests

from app.config import get_settings


class LocalOpenAIClient:
    """兼容 OpenAI Chat Completions 风格的本地客户端。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.llm_base_url.rstrip('/')
        self.api_key = self.settings.llm_api_key
        self.model_name = self.settings.llm_model_name
        self.timeout = self.settings.request_timeout

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2, **kwargs: Any) -> str:
        url = f'{self.base_url}/chat/completions'
        payload = {
            'model': self.model_name,
            'messages': messages,
            'temperature': temperature,
            **kwargs,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
