"""streaming suite -- SSE streaming behavior. See SPEC.md 7.2.

Deterministic structural checks only (SPEC.md 6): chunk count, stream
termination, finish_reason presence, stop-sequence honoring. The SSE framing is
itself under test, so stream-basic parses the raw wire bytes.

Note (2026-06): the swissai endpoint does NOT emit a `data: [DONE]` sentinel --
it terminates with a usage chunk (`choices: []`, `usage: {...}`). stream-basic
accepts either terminal convention. See SPEC.md 7.2.

stream-equiv (concatenated stream == non-stream content) is intentionally NOT
implemented yet -- it depends on temperature=0 determinism (open question 5),
which is unconfirmed for this endpoint.
"""

import json

import pytest

from quick_model_tests.client import ChatClient

pytestmark = pytest.mark.streaming


def test_stream_basic(client):
    """stream-basic: >=2 SSE chunks, non-empty content, clean termination.

    Termination is either a `[DONE]` sentinel OR a final usage chunk (empty
    `choices` + `usage`) -- this endpoint uses the latter.
    """
    resp = client.raw(
        client._payload(
            [{"role": "user", "content": "Count from one to five."}],
            stream=True,
            max_tokens=64,
        ),
        stream=True,
    )
    assert resp.ok, f"stream request failed: {resp.status_code}"
    n_data, saw_terminal, content = 0, False, []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            saw_terminal = True
            continue
        n_data += 1
        chunk = json.loads(payload)
        choices = chunk.get("choices") or []
        if not choices and chunk.get("usage"):
            saw_terminal = True  # this endpoint's terminal marker
        for choice in choices:
            piece = choice.get("delta", {}).get("content")
            if piece:
                content.append(piece)
    assert n_data >= 2, f"expected >=2 SSE chunks, got {n_data}"
    assert saw_terminal, "stream did not terminate with [DONE] or a usage chunk"
    assert "".join(content).strip(), "concatenated stream content was empty"


def test_stream_finish(client):
    """stream-finish: some chunk carries a finish_reason."""
    chunks = list(
        client.stream([{"role": "user", "content": "Say hello."}], max_tokens=32)
    )
    reasons = [
        choice.get("finish_reason")
        for ch in chunks
        for choice in (ch.get("choices") or [])
    ]
    assert any(r for r in reasons), f"no finish_reason in any chunk: {reasons}"


def test_stream_stop(client):
    """stream-stop: a streamed response honors the stop sequence."""
    text = ChatClient.stream_text(
        client.stream(
            [{"role": "user", "content": "Count: one two three four five"}],
            stop=["three"],
            max_tokens=64,
        )
    )
    assert text.strip(), "empty streamed content"
    assert "three" not in text, f"stop string leaked into stream: {text!r}"
