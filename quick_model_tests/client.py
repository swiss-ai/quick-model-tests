"""Thin requests-based client for the OpenAI-compatible chat completions API.

Intentionally minimal: no automatic retries or response massaging, because the
raw wire behavior (SSE framing, tool_calls JSON, error bodies) is itself under
test. See SPEC.md section 5.
"""

import json
from collections.abc import Iterator
from typing import Optional

import requests

from . import recording
from .config import Config


def _record_iter_lines(resp: "requests.Response") -> None:
    """Wrap ``resp.iter_lines`` so the streamed SSE body is recorded as the
    consumer drains it. Records on normal exhaustion or early close (finally),
    while the test context is still active. No-op when recording is off."""
    original = resp.iter_lines

    def teed(*args, **kwargs):
        buf = []
        try:
            for line in original(*args, **kwargs):
                if line:
                    buf.append(
                        line
                        if isinstance(line, str)
                        else line.decode("utf-8", "replace")
                    )
                yield line
        finally:
            recording.record("output", "\n".join(buf))

    resp.iter_lines = teed


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
        recording.record("input", json.dumps(payload, indent=2, ensure_ascii=False))
        resp = requests.post(
            f"{self.config.api_base}/chat/completions",
            headers=self._headers(),
            json=payload,
            stream=stream,
            timeout=self.config.timeout,
        )
        if not stream:
            # Non-stream body is safe to read here; requests caches it so the
            # caller's .json()/.text still works.
            recording.record("output", resp.text)
        else:
            # Tee iter_lines so the SSE body is recorded as whoever consumes it
            # (client.stream() OR a test iterating raw() directly) drains it.
            _record_iter_lines(resp)
        return resp

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
        for line in resp.iter_lines(decode_unicode=True):  # teed by raw() to record
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

    def tokenize(self, prompt: str, add_special_tokens: bool = True) -> list:
        """Token ids for ``prompt`` via the ``/tokenize`` endpoint."""
        resp = requests.post(
            f"{self.config.api_base}/tokenize",
            headers=self._headers(),
            json={
                "model": self.config.model,
                "prompt": prompt,
                "add_special_tokens": add_special_tokens,
            },
            timeout=self.config.timeout,
        )
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()["tokens"]

    def detokenize(self, tokens: list) -> str:
        """Text for ``tokens`` via the ``/detokenize`` endpoint."""
        resp = requests.post(
            f"{self.config.api_base}/detokenize",
            headers=self._headers(),
            json={"model": self.config.model, "tokens": tokens},
            timeout=self.config.timeout,
        )
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()["prompt"]

    # -- convenience helpers used by suites ---------------------------------

    @staticmethod
    def content(response: dict) -> Optional[str]:
        return response["choices"][0]["message"].get("content")

    # Field names different stacks use for the separate reasoning channel:
    # vLLM/SGLang use `reasoning_content`; the swissai gateway uses `reasoning`
    # (the DeepSeek-style name). Accept either, in priority order.
    _REASONING_KEYS = ("reasoning_content", "reasoning")

    @staticmethod
    def reasoning_content(response: dict) -> Optional[str]:
        """The separate reasoning channel a reasoning-parser populates, or None.

        Surfaced under different field names by different stacks (see
        `_REASONING_KEYS`). None means no reasoning channel was surfaced at all
        (a plain model, or a gateway that drops it)."""
        msg = response["choices"][0]["message"]
        return next((msg[k] for k in ChatClient._REASONING_KEYS if msg.get(k)), None)

    @staticmethod
    def reasoning_delta(delta: dict) -> Optional[str]:
        """The reasoning piece from a streaming `delta`, under either field name."""
        return next(
            (delta[k] for k in ChatClient._REASONING_KEYS if delta.get(k)), None
        )

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
