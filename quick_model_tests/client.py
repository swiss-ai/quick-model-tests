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

    def prompt_token_ids(self, prompt: str, n: int = 6) -> list:
        """Return the first ``n`` prompt token ids the server actually tokenized
        a raw ``/v1/completions`` prompt into, via ``prompt_logprobs`` (works
        through gateways that don't expose ``/tokenize``).

        The first ``prompt_logprobs`` entry is ``null`` (no logprob for the very
        first token), so position 0 is returned as ``None``; positions 1+ carry
        real ids. Raises ApiError if the endpoint doesn't return
        ``prompt_logprobs``."""
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 0,
        }
        resp = requests.post(
            f"{self.config.api_base}/completions",
            headers=self._headers(),
            json=body,
            timeout=self.config.timeout,
        )
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        pl = resp.json()["choices"][0].get("prompt_logprobs")
        if not pl:
            raise ApiError(resp.status_code, "endpoint returned no prompt_logprobs")
        return [int(next(iter(e))) if e else None for e in pl[:n]]

    # -- convenience helpers used by suites ---------------------------------

    @staticmethod
    def content(response: dict) -> Optional[str]:
        return response["choices"][0]["message"].get("content")

    @staticmethod
    def reasoning_content(response: dict) -> Optional[str]:
        """The separate reasoning channel a reasoning-parser populates, or None.

        vLLM/SGLang surface chain-of-thought in `message.reasoning_content`,
        distinct from the user-facing `content`. None means the endpoint did not
        split out a reasoning channel (a plain model, or the field was dropped)."""
        return response["choices"][0]["message"].get("reasoning_content")

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
