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

# Double-BOS background: when a chat template hardcodes the BOS token (Apertus'
# reasoning/answer template emits `{{ bos_token }}` = `<s>`), a client that
# applies the template and posts the rendered prompt to /completions hits a
# server that prepends BOS again -> `<s><s>...` -> text degeneration
# (apertus-program #420, raised by the SML eval team on vLLM 0.19). The check
# below is model-agnostic: it discovers the model's BOS from the tokenizer.

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
    """core-stop: a stop sequence is honored, across BOTH output channels.

    The stop string is matched against the raw generation, so a reasoning model
    can hit it while still inside its `<think>` block -- the partial output then
    lands in `reasoning_content` (or, on an endpoint that drops that channel,
    nowhere visible) rather than `content`. Checking only `content` would mistake
    that for an empty response. So: assert the stop string leaked into NEITHER
    channel, and that the stop actually took effect -- either some output came
    back, or `finish_reason` reports the stop.
    """
    resp = client.chat(
        [{"role": "user", "content": "Count: one two three four five"}],
        stop=["three"],
        max_tokens=_THINKING_MAX_TOKENS,
    )
    content = ChatClient.content(resp) or ""
    reasoning = ChatClient.reasoning_content(resp) or ""
    finish = resp["choices"][0]["finish_reason"]
    assert "three" not in content, f"stop string leaked into content: {content!r}"
    assert (
        "three" not in reasoning
    ), f"stop string leaked into reasoning_content: {reasoning!r}"
    assert (
        content.strip() or reasoning.strip() or finish == "stop"
    ), f"no output in either channel and finish_reason={finish!r} (expected 'stop')"


def test_core_usage(client):
    """core-usage: token accounting is internally consistent."""
    resp = client.chat([{"role": "user", "content": "Say hi."}], max_tokens=10)
    u = resp["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_core_no_double_bos(client, config):
    """core-no-double-bos: the /completions path must not prepend a 2nd BOS.

    Model-agnostic. The risk path is /completions (used by e.g. OpenWebUI): when a
    client applies the chat template (which hardcodes the BOS token) and posts the
    rendered prompt, a server that ALSO auto-adds BOS produces `<bos><bos>...` ->
    degeneration (apertus-program #420).

    1. Discover the model's BOS by tokenizing with vs without special tokens; the
       id that `add_special_tokens=True` prepends is the BOS. If nothing is
       prepended (e.g. Qwen has no BOS), double-BOS is impossible -> skip.
    2. Detokenize that id to the BOS string, prefix it onto a prompt, and read
       back what /completions tokenizes it to (via prompt_logprobs). If the first
       real token is the BOS again, the server double-added it -> fail.
    """
    try:
        with_special = client.tokenize("Paris", add_special_tokens=True)
        without_special = client.tokenize("Paris", add_special_tokens=False)
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    if not with_special:
        pytest.skip("could not tokenize")
    # add_special_tokens=True prepends the BOS that plain tokenization omits.
    if with_special == without_special:
        pytest.skip("model does not auto-prepend a BOS (double-BOS not possible)")
    bos_id = with_special[0]
    try:
        bos_str = client.detokenize([bos_id])
        ids = client.prompt_token_ids(f"{bos_str}The capital of France is Paris.")
    except ApiError as exc:
        pytest.skip(f"/detokenize or prompt_logprobs not available: {exc}")
    first_known = next((i for i in ids if i is not None), None)
    assert first_known != bos_id, (
        f"double-BOS on /completions: a {bos_str!r}-prefixed prompt tokenized to "
        f"two leading BOS tokens (id {bos_id}, first ids {ids}). A client posting a "
        f"chat-templated prompt here gets `{bos_str}{bos_str}...` -> degeneration "
        f"(apertus-program #420)."
    )
