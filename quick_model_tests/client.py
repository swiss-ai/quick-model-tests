"""Thin requests-based client for the OpenAI-compatible chat completions API.

Intentionally minimal: no automatic retries or response massaging, because the
raw wire behavior (SSE framing, tool_calls JSON, error bodies) is itself under
test. See SPEC.md section 5.
"""

import json
from collections.abc import Iterator
from typing import Optional

import requests

from .config import Config


class ApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class ChatClient:
    def __init__(self, config: Config):
        self.config = config

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.config.api_key:
            h["Authorization"] = f"Bearer {self.config.api_key}"
        return h

    def _payload(
        self,
        messages,
        *,
        stream,
        tools=None,
        tool_choice=None,
        max_tokens=None,
        stop=None,
        temperature=0.0,
        response_format=None,
        extra=None,
    ) -> dict:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)
        return payload

    def raw(self, payload: dict, *, stream: bool = False) -> requests.Response:
        """Escape hatch for error-path / malformed-request tests."""
        return requests.post(
            f"{self.config.api_base}/chat/completions",
            headers=self._headers(),
            json=payload,
            stream=stream,
            timeout=self.config.timeout,
        )

    def chat(self, messages, **kw) -> dict:
        resp = self.raw(self._payload(messages, stream=False, **kw))
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()

    def stream(self, messages, **kw) -> Iterator[dict]:
        """Yield parsed SSE delta chunks until the [DONE] sentinel."""
        resp = self.raw(self._payload(messages, stream=True, **kw), stream=True)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)

    # -- convenience helpers used by suites ---------------------------------

    @staticmethod
    def content(response: dict) -> Optional[str]:
        return response["choices"][0]["message"].get("content")

    @staticmethod
    def stream_text(chunks) -> str:
        out = []
        for ch in chunks:
            # The final chunk often carries usage with an empty `choices` list.
            for choice in ch.get("choices") or []:
                delta = choice.get("delta", {})
                if delta.get("content"):
                    out.append(delta["content"])
        return "".join(out)
