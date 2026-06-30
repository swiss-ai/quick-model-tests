"""core suite -- API contract basics. Reference implementation; see SPEC.md 7.1.

This is the pattern other suites should follow:
  - every assertion is deterministic (status, schema, token counts, substring /
    regex / closed-set membership) -- there is no LLM judge
  - prompts are short and constrained so the structural check is reliable
  - temperature=0 for reproducibility
"""

import pytest

from quick_model_tests.client import ApiError, ChatClient

pytestmark = pytest.mark.core

# Apertus BOS is `<s>` = token id 1. The reasoning/answer chat template hardcodes
# `{{ bos_token }}`, so a client that applies the template and posts the result to
# /completions hits a server that prepends BOS again -> `<s><s>...` -> text
# degeneration (apertus-program #420, raised by the SML eval team on vLLM 0.19).
_APERTUS_BOS_ID = 1

# Thinking-safe budget. A reasoning model (e.g. Qwen3.5) spends tokens on a
# `<think>` block that the server's reasoning parser strips out of `content`
# before the answer is emitted; on an endpoint that does not surface
# `reasoning_content`, a tight max_tokens truncates mid-thought and leaves
# `content` empty (looks like a model failure, but is just budget). Checks that
# assert post-thinking content use this; checks that deliberately probe a tight
# budget (core-maxtokens, core-usage) keep their own small value. See SPEC.md 7.6.
_THINKING_MAX_TOKENS = 1024


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
        max_tokens=_THINKING_MAX_TOKENS,
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
        max_tokens=_THINKING_MAX_TOKENS,
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


def test_core_no_double_bos(client, config):
    """core-no-double-bos: the /completions path must not prepend a 2nd BOS.

    The risk path is /completions (used by e.g. OpenWebUI): when a client applies
    the chat template (which hardcodes `<s>`) and posts the rendered prompt, a
    server that also auto-adds BOS produces `<s><s>...` -> degeneration (#420).

    Detect it directly: tokenize a `<s>`-prefixed prompt via /completions
    prompt_logprobs (gateway-friendly, unlike /tokenize). With a single BOS the
    leading ids are [<s>, <first text token>, ...]; with a double BOS they are
    [<s>, <s>, ...]. So the first non-null logprob position must NOT be the BOS.

    Apertus-specific (BOS id known); skips for other models and when the endpoint
    doesn't expose prompt_logprobs.
    """
    if "apertus" not in config.model.lower():
        pytest.skip("double-BOS check is Apertus-specific (BOS='<s>', id 1)")
    try:
        ids = client.prompt_token_ids("<s>The capital of France is Paris.")
    except ApiError as exc:
        pytest.skip(f"/completions prompt_logprobs not available: {exc}")
    first_known = next((i for i in ids if i is not None), None)
    assert first_known != _APERTUS_BOS_ID, (
        f"double-BOS on /completions: a '<s>'-prefixed prompt tokenized to two "
        f"leading BOS tokens (first ids {ids}). A client posting a chat-templated "
        f"prompt here gets `<s><s>...` -> degeneration (apertus-program #420)."
    )
