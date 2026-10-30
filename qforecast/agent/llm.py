"""Minimal async client for an OpenAI-compatible chat API (DeepSeek by default).

Same environment variables as the local DeepSeek MCP server, so one key serves both:
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL (default https://api.deepseek.com), DEEPSEEK_MODEL
The key is read from the environment only; it is never logged or stored.
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str = "", base_url: str = "", api_key: str = "", timeout_s: float = 30.0):
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self._key = api_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._json_mode = True
        self.calls = 0
        self.tokens = 0

    @property
    def configured(self) -> bool:
        return bool(self._key)

    async def ask_json(self, system: str, user: str, max_tokens: int = 400) -> tuple[dict, float]:
        """Return (parsed JSON object, latency seconds). Raises LLMError."""
        if not self._key:
            raise LLMError("DEEPSEEK_API_KEY is not set")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            r = await self._client.post(f"{self.base_url}/chat/completions", json=body,
                                        headers={"Authorization": f"Bearer {self._key}"})
            if r.status_code == 400 and self._json_mode:  # endpoint without JSON mode
                self._json_mode = False
                return await self.ask_json(system, user, max_tokens)
        except httpx.HTTPError as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        self.calls += 1
        self.tokens += int(data.get("usage", {}).get("total_tokens", 0))
        return parse_json(data["choices"][0]["message"].get("content") or ""), time.perf_counter() - t0

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating ```json fences or prose around it."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise LLMError(f"no JSON in reply: {text[:120]!r}") from None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(f"bad JSON in reply: {e}") from None
    if not isinstance(obj, dict):
        raise LLMError("reply is not a JSON object")
    return obj
