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

# Generous budget for the hard-prompt degeneration probe: large enough that a
# healthy model finishes a one-sentence answer well within it, so a run to
# `finish_reason="length"` is the double-BOS runaway signature, not a tight budget.
_HARD_MAX_TOKENS = 4096


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


def _discover_bos(client):
    """Return the model's BOS id, or skip if it has none / tokenizer is absent.

    The BOS is the id that ``add_special_tokens=True`` prepends and
    ``add_special_tokens=False`` does not. If they agree, the model auto-prepends
    nothing (e.g. Qwen has no BOS) and double-BOS is impossible."""
    try:
        with_special = client.tokenize("Paris", add_special_tokens=True)
        without_special = client.tokenize("Paris", add_special_tokens=False)
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    if not with_special or with_special == without_special:
        pytest.skip("model does not auto-prepend a BOS (double-BOS not possible)")
    return with_special[0]


def _leading_bos_count(ids, bos_id):
    """How many BOS ids the sequence starts with (0, 1, or more)."""
    n = 0
    for t in ids:
        if t == bos_id:
            n += 1
        else:
            break
    return n


def test_core_no_double_bos_chat(client):
    """core-no-double-bos-chat: the /chat tokenization path must not add a 2nd BOS
    on top of the template's own.

    This is the path the bug was reported on (apertus-program #420): the chat
    template emits `{{ bos_token }}`, so the rendered chat prompt already starts
    with `<s>`. If the server then tokenizes that rendered string with
    `add_special_tokens=True` -- which the Apertus multimodal path restores via
    `mm_processor.info.default_tok_params` -- the prompt begins `<s><s>...` ->
    degeneration. `core-no-double-bos` probes the same fault on /completions with a
    hand-crafted prompt; this probes the real chat path a chat client hits, by
    asking /tokenize to apply the server's own template.
    """
    bos_id = _discover_bos(client)
    try:
        ids = client.tokenize_chat(
            [{"role": "user", "content": "The capital of France is Paris."}]
        )
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept chat messages: {exc}")
    if not ids:
        pytest.skip("chat tokenization returned no tokens")
    leading = _leading_bos_count(ids, bos_id)
    assert leading <= 1, (
        f"double-BOS on the chat path: the server applied its chat template (which "
        f"emits the BOS) and then tokenized with add_special_tokens=True, so the "
        f"prompt starts with {leading} BOS tokens (id {bos_id}, first ids "
        f"{ids[:6]}) -> `<s><s>...` -> degeneration (apertus-program #420)."
    )


def test_core_no_double_bos_tokenize(client):
    """core-no-double-bos-tokenize: /tokenize must not add a 2nd BOS to a string
    that already begins with the BOS.

    Same fault as `core-no-double-bos`, but observed through `/tokenize` instead of
    `/completions` `prompt_logprobs`, so the check still fires on gateways that
    expose one endpoint but not the other. Prefixes the BOS string onto a prompt
    (mimicking an already-rendered chat template), tokenizes with
    `add_special_tokens=True`, and asserts the result does not start with two BOS.
    """
    bos_id = _discover_bos(client)
    try:
        bos_str = client.detokenize([bos_id])
        ids = client.tokenize(
            f"{bos_str}The capital of France is Paris.", add_special_tokens=True
        )
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")
    assert _leading_bos_count(ids, bos_id) <= 1, (
        f"double-BOS on /tokenize: a {bos_str!r}-prefixed prompt tokenized to two "
        f"leading BOS tokens (id {bos_id}, first ids {ids[:6]}). A client posting a "
        f"chat-templated prompt gets `{bos_str}{bos_str}...` -> degeneration "
        f"(apertus-program #420)."
    )


# Recognizable start-of-sequence tokens across model families. Used to identify
# the BOS from a rendered chat prompt WITHOUT relying on the tokenizer
# auto-prepending it -- after a template-owns fix (apertus-program #420) the
# tokenizer no longer adds a BOS, but the chat template still emits one as the
# rendered prompt's first token.
_BOS_TOKEN_RE = re.compile(
    r"^\s*(?:<s>|<\|begin_of_text\|>|<\|startoftext\|>|<bos>|\[BOS\]|"
    r"<\|begin▁of▁sentence\|>)\s*$"
)


def _discover_bos_from_chat(client):
    """Return ``(bos_id, bos_str, chat_ids)`` where the BOS is read from a rendered
    chat prompt -- the chat template emits it first -- so this works even when the
    tokenizer no longer auto-prepends a BOS (unlike ``_discover_bos``). Skips if the
    chat form / detokenize is unavailable, or if the prompt's first token is not a
    recognizable BOS (a model with no BOS, e.g. Qwen)."""
    try:
        ids = client.tokenize_chat([{"role": "user", "content": "Paris"}])
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept chat messages: {exc}")
    if len(ids) < 2:
        pytest.skip("chat tokenization returned too few tokens")
    try:
        bos_str = client.detokenize([ids[0]])
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")
    if not _BOS_TOKEN_RE.match(bos_str):
        pytest.skip(
            f"chat prompt does not begin with a recognizable BOS ({bos_str!r}); "
            f"model's chat format carries no leading BOS"
        )
    return ids[0], bos_str, ids


def test_core_bos_single_in_chat(client):
    """core-bos-single-in-chat: a chat-templated prompt begins with exactly one BOS.

    The *positive* companion to the double-BOS probes. Once the chat template is
    the sole BOS owner (the fix for apertus-program #420), the tokenizer no longer
    auto-prepends a BOS, so `_discover_bos` -- and every check that depends on it --
    skips. This check instead reads the BOS straight from the rendered chat prompt
    (the template emits it first), so it keeps *running* -- and stays green -- after
    the fix, asserting the surviving invariant: when the chat prompt begins with a
    BOS, there is exactly one, never two (the original bug). Model-agnostic: a chat
    format whose first token is not a recognizable BOS -- whether a model with no
    BOS (e.g. Qwen) or a regression that dropped it -- skips rather than failing, so
    this is a guard against re-doubling, not against a zero-BOS over-correction
    (which `core-bos-single-in-completion` and `core-no-degeneration` surface).
    """
    bos_id, bos_str, ids = _discover_bos_from_chat(client)
    count = _leading_bos_count(ids, bos_id)
    assert count == 1, (
        f"chat prompt must begin with exactly one BOS, got {count} leading "
        f"{bos_str!r} (first ids {ids[:6]}). Two means the double-BOS regression "
        f"is back -- the chat template must be the sole BOS owner "
        f"(apertus-program #420)."
    )


def test_core_bos_single_in_completion(client):
    """core-bos-single-in-completion: the raw (non-chat) tokenization path supplies
    exactly one BOS.

    Guards the OTHER side of BOS ownership. `/completions`, offline
    `generate`, and lm-eval loglikelihood tasks never invoke the chat template;
    they tokenize raw text with `add_special_tokens=True` and rely on the tokenizer
    to supply the BOS the model was pretrained with (the attention-sink first
    token). This check discovers the model's BOS from the chat template, then
    asserts a raw `add_special_tokens=True` tokenization begins with exactly one
    of it -- never zero, never two.

    Zero is the failure mode of an over-correction that makes the template the
    *sole* BOS owner (e.g. stripping the tokenizer's post-processor BOS): chat is
    fixed, but the completion/eval paths lose their BOS -> train/inference mismatch
    (apertus-program #420 discussion). Two is the original double-BOS. Skips for
    models with no BOS.
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    try:
        raw = client.tokenize(
            "The capital of France is Paris.", add_special_tokens=True
        )
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    count = _leading_bos_count(raw, bos_id)
    detail = (
        "0 means the tokenizer no longer prepends the BOS the model was pretrained "
        "with -- raw /completions and lm-eval loglikelihood paths now mismatch the "
        "training format (attention-sink token missing). "
        if count == 0
        else "2 means a double-BOS. "
        if count > 1
        else ""
    )
    assert count == 1, (
        f"raw tokenization (add_special_tokens=True) must supply exactly one BOS "
        f"(id {bos_id}, {bos_str!r}); got {count} (first ids {raw[:6]}). {detail}"
        f"The chat template and add_special_tokens are each authoritative for "
        f"different paths -- keep the completion path's BOS (apertus-program #420)."
    )


def test_core_bos_absent_without_specials(client):
    """core-bos-absent-without-specials: add_special_tokens=False prepends no BOS.

    The escape hatch that lets a caller which supplies its own BOS -- e.g. posting
    an already-rendered chat template -- avoid a double: with
    `add_special_tokens=False` the tokenizer must add nothing. Discovers the BOS
    from the chat template, then asserts a plain `add_special_tokens=False`
    tokenization does not begin with it. Skips for models with no BOS.
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    try:
        raw = client.tokenize(
            "The capital of France is Paris.", add_special_tokens=False
        )
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    assert not raw or raw[0] != bos_id, (
        f"add_special_tokens=False still prepended the BOS (id {bos_id}, "
        f"{bos_str!r}); first ids {raw[:6]}. The no-specials path must be clean so a "
        f"client supplying its own BOS is not doubled (apertus-program #420)."
    )


def test_core_bos_consistent_identity(client):
    """core-bos-consistent-identity: the chat and raw paths agree on the BOS token.

    The BOS the chat template emits must be the same id the tokenizer prepends on
    the raw `add_special_tokens=True` path -- otherwise the two paths feed the model
    different 'start' tokens. Only checked when the raw path actually prepends a BOS
    (skips on the over-corrected config with no raw-path BOS, which
    `core-bos-single-in-completion` already flags).
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    try:
        with_special = client.tokenize("Paris", add_special_tokens=True)
        without_special = client.tokenize("Paris", add_special_tokens=False)
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    if not with_special or with_special == without_special:
        pytest.skip("raw path prepends no BOS (see core-bos-single-in-completion)")
    assert with_special[0] == bos_id, (
        f"BOS identity mismatch: the chat template emits id {bos_id} ({bos_str!r}) "
        f"but the raw add_special_tokens=True path prepends id {with_special[0]}. "
        f"Both paths must feed the model the same BOS it was trained with "
        f"(apertus-program #420)."
    )


def test_core_eos_not_appended_to_prompt(client):
    """core-eos-not-appended: add_special_tokens=True must not append an EOS to a
    raw prompt.

    The EOS-side mirror of the BOS ownership checks. A raw completion prompt is a
    prefix the model continues from; a tokenizer misconfigured with
    `add_eos_token=True` appends the EOS, so the model sees a premature stop token
    mid-context -> truncated or degenerate continuations. Asserts the last token of
    a raw `add_special_tokens=True` tokenization is not a control/EOS token. (The
    thread noted #420 is BOS-only -- this pins that the EOS side stays clean too.)
    """
    try:
        raw = client.tokenize(
            "The capital of France is Paris", add_special_tokens=True
        )
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    if not raw:
        pytest.skip("could not tokenize")
    try:
        last = client.detokenize([raw[-1]])
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")
    assert not _CONTROL_TOKEN_RE.search(last), (
        f"add_special_tokens=True appended a control/EOS token {last!r} to a raw "
        f"prompt (last ids {raw[-3:]}). A completion prompt must not end in an EOS -- "
        f"the model would see a premature stop mid-context."
    )


def _degeneration_reason(content):
    """Structural degeneration heuristic (lenient, to avoid flagging legitimate
    repetition): flag if a word repeats >=6x consecutively or a single word is >50%
    of the output. Returns a failure message, or None when the output is clean --
    including when it is too short (<8 words) to assess (a short answer is not
    degenerate)."""
    words = content.split()
    if len(words) < 8:
        return None
    max_run = run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    if max_run >= 6:
        return f"a word repeats {max_run}x consecutively: {content[:200]!r}"
    word, count = Counter(words).most_common(1)[0]
    if count / len(words) > 0.5:
        return f"{word!r} is {count}/{len(words)} of the output: {content[:200]!r}"
    return None


def _assert_not_degenerate(content):
    """Assert `content` is not degenerate. Skips when it is too short to assess --
    used where the prompt is expected to produce enough text (e.g. "two
    sentences")."""
    if len(content.split()) < 8:
        pytest.skip(f"answer too short to assess: {content!r}")
    reason = _degeneration_reason(content)
    assert reason is None, f"degenerate: {reason}"


def test_core_no_degeneration(client):
    """core-no-degeneration: a normal prompt produces coherent, non-degenerate text.

    Config breakage (e.g. double-BOS) shows up as degeneration -- the output
    collapses into one token/phrase repeated far past any natural limit. This is
    the end-to-end effect that `core-no-double-bos` catches at the token level.
    """
    resp = client.chat(
        [{"role": "user", "content": "Write two sentences about the ocean."}],
        max_tokens=_THINKING_MAX_TOKENS,
    )
    content = ChatClient.content(resp) or ChatClient.reasoning_content(resp) or ""
    _assert_not_degenerate(content)


def test_core_no_degeneration_hard(client):
    """core-no-degeneration-hard: a hard prompt completes and stops, not runs away.

    The behavioral signature of the double-BOS bug (apertus-program #420) showed up
    only on *hard* prompts (medqa/math): the model ran to the token budget and never
    emitted its stop token. `core-no-degeneration` uses an easy prompt and so misses
    it; this uses a hard, bounded-answer clinical prompt with a generous budget and
    asserts the model reaches a natural stop (`finish_reason="stop"`, not `"length"`)
    and does not degenerate. This is the only end-to-end catch for the multimodal /
    completion-path double-BOS that the `/tokenize` probes can only proxy. Answer
    correctness is not asserted (the prompt is a vehicle, not a knowledge test); a
    legitimately long answer that trips `finish_reason` can relax the budget.
    """
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "A 45-year-old presents with sudden tearing chest pain "
                "radiating to the back, unequal arm blood pressures, and a widened "
                "mediastinum on chest X-ray. Give the single most likely diagnosis "
                "in one short sentence.",
            }
        ],
        max_tokens=_HARD_MAX_TOKENS,
    )
    finish = resp["choices"][0]["finish_reason"]
    content = ChatClient.content(resp) or ChatClient.reasoning_content(resp) or ""
    assert finish == "stop", (
        f"hard prompt ran to finish_reason={finish!r} (budget {_HARD_MAX_TOKENS}) "
        f"without stopping -- the runaway signature of a double-BOS on the "
        f"chat/mm generation path (apertus-program #420). Tail: {content[-200:]!r}"
    )
    # A correct short answer that stopped is a pass -- reaching a natural stop IS
    # the signal here; only flag if longer output is actually degenerate.
    reason = _degeneration_reason(content)
    assert reason is None, f"degenerate on hard prompt: {reason}"


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
