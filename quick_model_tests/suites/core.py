"""core suite -- API contract basics. Reference implementation; see SPEC.md 7.1.

This is the pattern other suites should follow:
  - every assertion is deterministic (status, schema, token counts, substring /
    regex / closed-set membership) -- there is no LLM judge
  - prompts are short and constrained so the structural check is reliable
  - temperature=0 for reproducibility
"""

import re
from collections import Counter

import pytest

from quick_model_tests.client import ApiError, ChatClient

pytestmark = pytest.mark.core

# Control/special-token families that must never appear in user-visible content,
# regardless of model: `<|...|>` (im_start/end, eot_id, endoftext, inner_*,
# tools_*, assistant_end, ...), `<s>`/`</s>`, Llama-style `[INST]`/`[/INST]`, and
# raw `<think>` tags. Leakage means the served chat template / parser is not
# cleanly separating structure from content.
_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]*\|>|</?s>|\[/?INST\]|</?think\b", re.IGNORECASE)

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


def test_core_eos(client):
    """core-eos: the model stops on its own EOS for a short, complete answer.

    A bounded question with a one-word answer should finish with
    `finish_reason="stop"`, not run to the token budget (`"length"`). Always
    finishing with "length" points at a misconfigured `eos_token_id` / generation
    config -- the served model never emits its stop token. The final content must
    also carry no raw EOS/control token (the template/parser should consume it).
    """
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "What is the capital of France? Answer with just the "
                "city name.",
            }
        ],
        max_tokens=_THINKING_MAX_TOKENS,
    )
    finish = resp["choices"][0]["finish_reason"]
    assert finish == "stop", (
        f"expected natural stop, got finish_reason={finish!r} -- the model did not "
        f"reach EOS within the budget: a misconfigured eos_token_id / generation "
        f"config, or a reasoning model that never finished thinking"
    )
    content = ChatClient.content(resp) or ""
    leak = _CONTROL_TOKEN_RE.search(content)
    assert not leak, (
        f"raw control/EOS token {leak.group(0)!r} leaked into content: {content!r}"
    )


def test_core_template_no_leak(client):
    """core-template-no-leak: chat-template control tokens never reach content.

    Across a short multi-turn exchange, assistant `content` must be free of any
    chat-template/control marker (BOS/EOS, role markers, think/tool scaffolding)
    from any common family -- see `_CONTROL_TOKEN_RE`. Leakage means the served
    template/parser isn't cleanly separating structure from content.
    """
    messages = [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Reply with a one-sentence greeting."},
    ]
    resp = client.chat(messages, max_tokens=_THINKING_MAX_TOKENS)
    content = ChatClient.content(resp) or ""
    reasoning = ChatClient.reasoning_content(resp) or ""
    # A thinking model may spend the whole budget in `reasoning` and leave
    # `content` empty; either channel counts as real output. Control tokens must
    # leak into NEITHER.
    assert content.strip() or reasoning.strip(), "empty assistant output (both channels)"
    for name, chan in (("content", content), ("reasoning", reasoning)):
        leak = _CONTROL_TOKEN_RE.search(chan)
        assert not leak, (
            f"chat-template control token {leak.group(0)!r} leaked into {name}: "
            f"{chan!r}"
        )


def test_core_tokenizer_roundtrip(client):
    """core-tokenizer-roundtrip: detokenize(tokenize(text)) == text.

    A basic sanity check on the served tokenizer/detokenizer config -- a broken
    or mismatched tokenizer corrupts every prompt. Skips if the endpoint does not
    expose /tokenize or /detokenize.
    """
    text = "The quick brown fox jumps over the lazy dog. 42!"
    try:
        ids = client.tokenize(text, add_special_tokens=False)
        back = client.detokenize(ids)
    except ApiError as exc:
        pytest.skip(f"/tokenize or /detokenize not available: {exc}")
    assert back == text, f"tokenizer round-trip mismatch: {back!r} != {text!r}"


def test_core_tokenizer_unicode(client):
    """core-tokenizer-unicode: multilingual/emoji/RTL text round-trips intact.

    Byte-level BPE and normalization bugs surface as a lossy round-trip on
    non-ASCII input (`core-tokenizer-roundtrip` only covers ASCII). Skips if
    /tokenize or /detokenize is absent.
    """
    text = "ZÜRICH-🦊-مرحبا-4827"
    try:
        ids = client.tokenize(text, add_special_tokens=False)
        back = client.detokenize(ids)
    except ApiError as exc:
        pytest.skip(f"/tokenize or /detokenize not available: {exc}")
    assert back == text, f"unicode round-trip mismatch: {back!r} != {text!r}"


def test_core_bos_single_token(client):
    """core-bos-single-token: the model's BOS string re-encodes to exactly one
    BOS token.

    Discovers the BOS (the id `add_special_tokens=True` prepends), detokenizes it
    to its string form, and re-tokenizes that string WITHOUT specials: it must be
    exactly ``[bos_id]``. If the BOS string splits into several tokens, the chat
    template (which emits that string) and the tokenizer disagree -- the template
    text won't map back to the BOS the model was trained on. Skips when the model
    has no BOS or the tokenizer endpoints are absent.
    """
    try:
        with_special = client.tokenize("Paris", add_special_tokens=True)
        without_special = client.tokenize("Paris", add_special_tokens=False)
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    if not with_special or with_special == without_special:
        pytest.skip("model does not auto-prepend a BOS")
    bos_id = with_special[0]
    try:
        bos_str = client.detokenize([bos_id])
        reencoded = client.tokenize(bos_str, add_special_tokens=False)
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")
    assert reencoded == [bos_id], (
        f"BOS string {bos_str!r} did not re-encode to a single BOS token "
        f"(id {bos_id}); got {reencoded} -- chat-template/tokenizer mismatch"
    )


def test_core_no_degeneration(client):
    """core-no-degeneration: a normal prompt produces coherent, non-degenerate text.

    Config breakage (e.g. double-BOS) shows up as degeneration -- the output
    collapses into one token/phrase repeated far past any natural limit. This is
    the end-to-end effect that `core-no-double-bos` catches at the token level.
    Structural heuristic (lenient, to avoid flagging legitimate repetition): no
    word repeats >=6x consecutively, and no single word is >50% of the output.
    """
    resp = client.chat(
        [{"role": "user", "content": "Write two sentences about the ocean."}],
        max_tokens=_THINKING_MAX_TOKENS,
    )
    content = ChatClient.content(resp) or ChatClient.reasoning_content(resp) or ""
    words = content.split()
    if len(words) < 8:
        pytest.skip(f"answer too short to assess ({len(words)} words): {content!r}")
    max_run = run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    assert max_run < 6, (
        f"degenerate: a word repeats {max_run}x consecutively: {content[:200]!r}"
    )
    word, count = Counter(words).most_common(1)[0]
    assert count / len(words) <= 0.5, (
        f"degenerate: {word!r} is {count}/{len(words)} of the output: {content[:200]!r}"
    )


def test_core_multi_system(client):
    """core-multi-system: multiple system messages are accepted and honored.

    Multiple system turns are a standard OpenAI-shaped input, so the server should
    handle them. Two failure modes, both red: an explicit refusal (e.g. 400
    "system message must be at the beginning") means the template can't take more
    than one system turn; a silent drop -- keeping only the first and discarding
    the rest -- shows up as a reply outside {red, blue, yellow}, since the
    closed-set constraint lives in the SECOND system message.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": "Reply with exactly one word: the name of a primary color "
            "(red, blue, or yellow). No punctuation.",
        },
        {"role": "user", "content": "Give me a primary color."},
    ]
    try:
        resp = client.chat(messages, max_tokens=_THINKING_MAX_TOKENS)
    except ApiError as exc:
        pytest.fail(
            f"endpoint rejected a second system message ({exc.status}) -- a standard "
            f"OpenAI-shaped input the server should handle: {exc.body[:200]}"
        )
    content = (ChatClient.content(resp) or "").strip().lower().rstrip(".")
    assert content in {"red", "blue", "yellow"}, (
        f"second system message not honored (answer out of closed set): {content!r}"
    )


def test_core_assistant_prefill(client):
    """core-assistant-prefill: a trailing assistant turn is CONTINUED, not restarted.

    With `continue_final_message`/`add_generation_prompt=false`, a history ending
    in an assistant turn should have the model continue that exact text rather
    than open a fresh turn -- the template must emit no generation prompt / closing
    role marker after it. Red on both failure modes: an explicit rejection of the
    flags, or an empty/wrong continuation (the endpoint accepted the flags but
    ignored them -- e.g. re-opened a fresh turn and left content empty).
    """
    messages = [
        {"role": "user", "content": "Complete the sentence with a single word."},
        {"role": "assistant", "content": "The capital of France is"},
    ]
    try:
        resp = client.chat(
            messages,
            max_tokens=_THINKING_MAX_TOKENS,
            extra={"continue_final_message": True, "add_generation_prompt": False},
        )
    except ApiError as exc:
        pytest.fail(
            f"endpoint rejected assistant-prefill continuation flags ({exc.status}): "
            f"{exc.body[:200]}"
        )
    content = (ChatClient.content(resp) or "").strip()
    assert content, (
        "prefill produced no content -- the endpoint ignored continue_final_message "
        "(the trailing assistant turn was not continued)"
    )
    assert "paris" in content.lower(), (
        f"prefill not continued (expected 'Paris'): {content!r}"
    )


def test_core_determinism(client):
    """core-determinism: temperature=0 generation is reproducible.

    `chat()` sends temperature=0, so the same prompt must yield byte-identical
    output on two calls -- the greedy-decoding contract. A mismatch points at a
    non-deterministic generation config (or batched-serving nondeterminism, the
    SPEC.md open question). Compares both visible channels; skips only if nothing
    at all came back.
    """

    def _visible(resp: dict) -> tuple:
        return (ChatClient.content(resp) or "", ChatClient.reasoning_content(resp) or "")

    prompt = [{"role": "user", "content": "Name three primary colors, comma-separated."}]
    a = _visible(client.chat(prompt, max_tokens=_THINKING_MAX_TOKENS))
    b = _visible(client.chat(prompt, max_tokens=_THINKING_MAX_TOKENS))
    if not any(a):
        pytest.skip("no visible output to compare")
    assert a == b, f"temp=0 output not reproducible:\n  run1={a!r}\n  run2={b!r}"
