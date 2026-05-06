from __future__ import annotations

import threading
from typing import Any

import requests


class VllmOpenAIClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        timeout_seconds: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.timeout_seconds = timeout_seconds
        self._tls = threading.local()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _session(self) -> requests.Session:
        session = getattr(self._tls, "session", None)
        if session is None:
            session = requests.Session()
            self._tls.session = session
        return session

    def generate(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
            "stream": False,
        }
        response = self._session().post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise ValueError(f"Invalid vLLM response: {body}")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError(f"Invalid vLLM response content: {body}")
        return content

