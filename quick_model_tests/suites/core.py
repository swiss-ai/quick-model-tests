"""core suite -- API contract basics. Reference implementation; see SPEC.md 7.1.

This is the pattern other suites should follow:
  - every assertion is deterministic (status, schema, token counts, substring /
    regex / closed-set membership) -- there is no LLM judge
  - prompts are short and constrained so the structural check is reliable
  - temperature=0 for reproducibility
"""

import pytest

from quick_model_tests.client import ChatClient

pytestmark = pytest.mark.core


def test_core_health(client):
    """core-health: a basic completion returns non-empty content + usage."""
    resp = client.chat([{"role": "user", "content": "Who is Pablo Picasso?"}])
    assert resp["choices"], "no choices in response"
    content = ChatClient.content(resp)
    assert content and content.strip(), "empty assistant content"
    assert "usage" in resp and resp["usage"]["total_tokens"] > 0


def test_core_system(client):
    """core-system: the system prompt is honored -- deterministic closed-set.

    A tightly constrained instruction (one word, from a known set) makes the
    check fully structural: membership in {red, blue, yellow}.
    """
    resp = client.chat(
        [
            {
                "role": "system",
                "content": "Reply with exactly one word: the name of a primary color "
                "(red, blue, or yellow). No punctuation.",
            },
            {"role": "user", "content": "Give me a primary color."},
        ],
        max_tokens=20,
    )
    content = (ChatClient.content(resp) or "").strip().lower().rstrip(".")
    assert len(content.split()) <= 2, f"expected ~one word, got: {content!r}"
    assert content in {"red", "blue", "yellow"}, f"not in closed set: {content!r}"


def test_core_maxtokens(client):
    """core-maxtokens: max_tokens is honored and finish_reason reflects it."""
    resp = client.chat(
        [{"role": "user", "content": "Write a long essay about the ocean."}],
        max_tokens=16,
    )
    usage = resp["usage"]
    assert usage["completion_tokens"] <= 16 + 1, usage  # allow off-by-one
    assert resp["choices"][0]["finish_reason"] in ("length", "stop")


def test_core_stop(client):
    """core-stop: a stop sequence is not present in the output."""
    resp = client.chat(
        [{"role": "user", "content": "Count: one two three four five"}],
        stop=["three"],
        max_tokens=50,
    )
    content = ChatClient.content(resp)
    assert content and content.strip(), "empty assistant content"
    assert "three" not in content, f"stop string leaked: {content!r}"


def test_core_usage(client):
    """core-usage: token accounting is internally consistent."""
    resp = client.chat([{"role": "user", "content": "Say hi."}], max_tokens=10)
    u = resp["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
