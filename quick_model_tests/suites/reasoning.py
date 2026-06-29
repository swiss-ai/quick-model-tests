"""reasoning suite -- `<think>` / inner-monologue path. See SPEC.md 7.6.

No quality judging -- only that the reasoning path is structurally well-formed:
the answer is correct on a constrained closed-set prompt, and no raw thinking
tokens leak into user-visible `content` (SPEC.md line 148-149).

reason-produced (assert a thinking segment exists) is intentionally NOT
implemented yet: how thinking is surfaced is deployment-dependent (open question
2) -- e.g. the swissai gateway was dropping `reasoning_content` entirely (fixed
in serving-api), and some builds expose nothing. The separation + answer checks
below hold regardless of how (or whether) thinking is surfaced.

Budgets are generous: a reasoning model may spend 100+ tokens thinking before it
emits the answer, so a tight max_tokens would truncate it.
"""

import re

import pytest

from quick_model_tests.client import ChatClient

pytestmark = pytest.mark.reasoning

# Raw reasoning / special tokens that must never appear in user-visible content.
THINK_TOKEN_RE = re.compile(
    r"<\|inner_prefix\|>|<\|inner_suffix\|>|</?think\b|<\|[^>]*\|>", re.IGNORECASE
)


def test_reason_answer(client):
    """reason-answer: a constrained arithmetic prompt yields the exact answer."""
    content = (
        ChatClient.content(
            client.chat(
                [
                    {
                        "role": "user",
                        "content": "What is 6 times 7? Reply with only the number.",
                    }
                ],
                max_tokens=512,
            )
        )
        or ""
    )
    assert "42" in content, f"expected 42 in the answer, got: {content!r}"


def test_reason_separation(client):
    """reason-separation: final content carries NO raw <think> / <|inner_*|> tokens."""
    content = (
        ChatClient.content(
            client.chat(
                [
                    {
                        "role": "user",
                        "content": "Think step by step, then give the answer: "
                        "if a train travels 60 km in 1.5 hours, what is "
                        "its average speed in km/h?",
                    }
                ],
                max_tokens=512,
            )
        )
        or ""
    )
    assert content.strip(), "empty content"
    leak = THINK_TOKEN_RE.search(content)
    assert not leak, (
        f"raw reasoning token {leak.group(0)!r} leaked into content: {content!r}"
    )
