"""special_tokens suite -- BOS/EOS ownership across every tokenization path.

Every check in here is token-level: it reads back what the server's DEFAULT
tokenization produced (via `/tokenize`, `/detokenize`, or `/completions`
`prompt_logprobs`) and asserts an invariant about the special tokens at the
edges of the prompt. No request in this suite sends `add_special_tokens` -- the
suite judges the server's default behavior, not what a client can override.

Background (apertus-program #420, raised by the SML eval team on vLLM 0.19):
when a chat template hardcodes the BOS token (Apertus' reasoning/answer template
emits `{{ bos_token }}` = `<s>`), a client that applies the template and posts
the rendered prompt hits a server that prepends BOS again -> `<s><s>...` -> text
degeneration. The fix makes exactly one layer the BOS owner per path, which is
what these checks pin down:

    chat path        template owns the BOS   -> bos_single_in_chat, no_double_bos_chat
    raw/completion   tokenizer owns the BOS  -> bos_single_in_completion
    both agree       same token id           -> bos_consistent_identity
    EOS side         nobody appends one      -> eos_not_appended_to_prompt

The over-correction matters as much as the original bug: stripping the
tokenizer's post-processor BOS fixes chat but leaves `/completions` and lm-eval
loglikelihood paths with no BOS at all -> train/inference mismatch. So the checks
assert *exactly one*, never "at most one", wherever a BOS is expected.

Every check is model-agnostic: the BOS is discovered from a rendered chat prompt
(see `_discover_bos_from_chat`), and a model with no BOS (e.g. Qwen) skips.

The end-to-end behavioral counterpart -- degeneration and runaway generation, the
symptom a double-BOS actually produces -- stays in the `core` suite
(`core-no-degeneration`, `core-no-degeneration-hard`).
"""

import re

import pytest

from quick_model_tests.client import ApiError, ChatClient
from quick_model_tests.suites.core import _CONTROL_TOKEN_RE, _THINKING_MAX_TOKENS
from quick_model_tests.suites.multimodal import _audio, _image, _text

pytestmark = pytest.mark.special_tokens

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
    chat prompt -- the chat template emits it first -- so discovery never has to
    override ``add_special_tokens``: every request in the BOS checks runs with the
    server's own defaults, which is what the suite is judging. Skips if the
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


def _leading_bos_count(ids, bos_id):
    """How many BOS ids the sequence starts with (0, 1, or more)."""
    n = 0
    for t in ids:
        if t == bos_id:
            n += 1
        else:
            break
    return n


# --- BOS: the chat path (template owns the BOS) ------------------------------


def test_core_bos_single_in_chat(client):
    """core-bos-single-in-chat: a chat-templated prompt begins with exactly one BOS.

    The *positive* companion to the double-BOS probes. Reads the BOS straight from
    the rendered chat prompt (the template emits it first) and asserts the
    invariant: when the chat prompt begins with a BOS, there is exactly one, never
    two (the original bug). Model-agnostic: a chat format whose first token is not
    a recognizable BOS -- whether a model with no BOS (e.g. Qwen) or a regression
    that dropped it -- skips rather than failing, so this is a guard against
    re-doubling, not against a zero-BOS over-correction (which
    `core-bos-single-in-completion` and `core-no-degeneration` surface).
    """
    bos_id, bos_str, ids = _discover_bos_from_chat(client)
    count = _leading_bos_count(ids, bos_id)
    assert count == 1, (
        f"chat prompt must begin with exactly one BOS, got {count} leading "
        f"{bos_str!r} (first ids {ids[:6]}). Two means the double-BOS regression "
        f"is back -- the chat template must be the sole BOS owner "
        f"(apertus-program #420)."
    )


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
    asking /tokenize to apply the server's own template (all defaults, nothing
    overridden).
    """
    bos_id, _, _ = _discover_bos_from_chat(client)
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
        f"emits the BOS) and then tokenized with specials added on top, so the "
        f"prompt starts with {leading} BOS tokens (id {bos_id}, first ids "
        f"{ids[:6]}) -> `<s><s>...` -> degeneration (apertus-program #420)."
    )


# --- BOS: the raw / completion path (tokenizer owns the BOS) -----------------


def test_core_no_double_bos(client):
    """core-no-double-bos: the /completions path must not prepend a 2nd BOS.

    Model-agnostic. The risk path is /completions (used by e.g. OpenWebUI): when a
    client applies the chat template (which hardcodes the BOS token) and posts the
    rendered prompt, a server that ALSO auto-adds BOS produces `<bos><bos>...` ->
    degeneration (apertus-program #420).

    1. Discover the model's BOS from a rendered chat prompt (the template emits
       it first) -- no `add_special_tokens` override anywhere, so every request
       exercises the server's own defaults. No recognizable BOS -> skip.
    2. Prefix the BOS string onto a prompt and read back what /completions
       tokenizes it to by default (via prompt_logprobs). If the first real token
       is the BOS again, the server double-added it -> fail.
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    try:
        ids = client.prompt_token_ids(f"{bos_str}The capital of France is Paris.")
    except ApiError as exc:
        pytest.skip(f"prompt_logprobs not available: {exc}")
    first_known = next((i for i in ids if i is not None), None)
    assert first_known != bos_id, (
        f"double-BOS on /completions: a {bos_str!r}-prefixed prompt tokenized to "
        f"two leading BOS tokens (id {bos_id}, first ids {ids}). A client posting a "
        f"chat-templated prompt here gets `{bos_str}{bos_str}...` -> degeneration "
        f"(apertus-program #420)."
    )


def test_core_no_double_bos_tokenize(client):
    """core-no-double-bos-tokenize: /tokenize must not add a 2nd BOS to a string
    that already begins with the BOS.

    Same fault as `core-no-double-bos`, but observed through `/tokenize` instead of
    `/completions` `prompt_logprobs`, so the check still fires on gateways that
    expose one endpoint but not the other. Prefixes the BOS string onto a prompt
    (mimicking an already-rendered chat template), tokenizes with the server's
    DEFAULT settings (no `add_special_tokens` sent), and asserts the result does
    not start with two BOS: a sensible default must not double a BOS the caller
    already supplied.
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    try:
        ids = client.tokenize(f"{bos_str}The capital of France is Paris.")
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    assert _leading_bos_count(ids, bos_id) <= 1, (
        f"double-BOS on /tokenize: a {bos_str!r}-prefixed prompt tokenized to two "
        f"leading BOS tokens (id {bos_id}, first ids {ids[:6]}). A client posting a "
        f"chat-templated prompt gets `{bos_str}{bos_str}...` -> degeneration "
        f"(apertus-program #420)."
    )


def test_core_bos_single_in_completion(client):
    """core-bos-single-in-completion: the raw (non-chat) tokenization path supplies
    exactly one BOS.

    Guards the OTHER side of BOS ownership. `/completions`, offline
    `generate`, and lm-eval loglikelihood tasks never invoke the chat template;
    they tokenize raw text with the server's defaults and rely on the tokenizer
    to supply the BOS the model was pretrained with (the attention-sink first
    token). This check discovers the model's BOS from the chat template, then
    asserts a raw DEFAULT tokenization (no `add_special_tokens` sent) begins with
    exactly one of it -- never zero, never two.

    Zero is the failure mode of an over-correction that makes the template the
    *sole* BOS owner (e.g. stripping the tokenizer's post-processor BOS): chat is
    fixed, but the completion/eval paths lose their BOS -> train/inference mismatch
    (apertus-program #420 discussion). Two is the original double-BOS. Skips for
    models with no BOS.
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    try:
        raw = client.tokenize("The capital of France is Paris.")
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    count = _leading_bos_count(raw, bos_id)
    detail = (
        "0 means the default tokenization does not prepend the BOS the model was "
        "pretrained with -- raw /completions and lm-eval loglikelihood paths "
        "mismatch the training format (attention-sink token missing). "
        if count == 0
        else "2 means a double-BOS. "
        if count > 1
        else ""
    )
    assert count == 1, (
        f"default raw tokenization must supply exactly one BOS "
        f"(id {bos_id}, {bos_str!r}); got {count} (first ids {raw[:6]}). {detail}"
        f"The chat template and the raw default are each authoritative for "
        f"different paths -- keep the completion path's BOS (apertus-program #420)."
    )


# --- BOS: the two paths must agree -------------------------------------------


def test_core_bos_single_token(client):
    """core-bos-single-token: the model's BOS string encodes to exactly one
    BOS token.

    Discovers the BOS from a rendered chat prompt, then tokenizes a plain prompt
    and the same prompt with the BOS string prefixed -- both with the server's
    DEFAULT tokenization (no `add_special_tokens` override). The prefixed result
    must be exactly `[bos_id] + plain`: whatever the default adds, it adds to
    both, so the difference isolates how the BOS *string* encodes. If it splits
    into several tokens, the chat template (which emits that string) and the
    tokenizer disagree -- the template text won't map back to the BOS the model
    was trained on. Skips when the model has no BOS or /tokenize is absent.
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    text = "The capital of France is Paris."
    try:
        plain = client.tokenize(text)
        prefixed = client.tokenize(f"{bos_str}{text}")
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    assert prefixed == [bos_id] + plain, (
        f"BOS string {bos_str!r} did not encode to a single BOS token "
        f"(id {bos_id}) on the default /tokenize path: prefixed ids {prefixed[:8]} "
        f"vs plain ids {plain[:8]} -- chat-template/tokenizer mismatch"
    )


def test_core_bos_consistent_identity(client):
    """core-bos-consistent-identity: the chat and raw paths agree on the BOS token.

    The BOS the chat template emits must be the same id the tokenizer prepends on
    the raw default path -- otherwise the two paths feed the model different
    'start' tokens. Only checked when the raw default actually prepends a
    recognizable BOS (skips on a config with no raw-path BOS, which
    `core-bos-single-in-completion` already flags).
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    try:
        raw = client.tokenize("Paris")
        raw_first = client.detokenize([raw[0]]) if raw else ""
    except ApiError as exc:
        pytest.skip(f"/tokenize or /detokenize not available: {exc}")
    if not _BOS_TOKEN_RE.match(raw_first):
        pytest.skip(
            "raw default path prepends no BOS (see core-bos-single-in-completion)"
        )
    assert raw[0] == bos_id, (
        f"BOS identity mismatch: the chat template emits id {bos_id} ({bos_str!r}) "
        f"but the raw default path prepends id {raw[0]} ({raw_first!r}). "
        f"Both paths must feed the model the same BOS it was trained with "
        f"(apertus-program #420)."
    )


# --- BOS: the multimodal chat path -------------------------------------------


def _assert_single_bos_in_mm_chat(client, part, kind):
    """Tokenize a `<text> + <modality>` chat through the server's own template and
    assert exactly one leading BOS. The path apertus-program #420 was reported on:
    vLLM's multimodal tokenization restores ``add_special_tokens=True`` (via
    ``mm_processor.info.default_tok_params``), so if the chat template also emits
    ``{{ bos_token }}`` the rendered prompt starts ``<s><s>``. Skips if
    ``/tokenize`` does not accept multimodal messages or the model has no BOS.

    (Best-effort proxy: ``/tokenize`` may not traverse the same mm code path as
    generation, but it is the only observable surface for the rendered mm prompt.)
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    messages = [{"role": "user", "content": [_text("Describe this input."), part]}]
    try:
        ids = client.tokenize_chat(messages)
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept multimodal messages: {exc}")
    if not ids:
        pytest.skip("multimodal chat tokenization returned no tokens")
    count = _leading_bos_count(ids, bos_id)
    assert count == 1, (
        f"{kind} chat prompt must begin with exactly one BOS (id {bos_id}, "
        f"{bos_str!r}), got {count} (first ids {ids[:6]}). Two is the multimodal "
        f"double-BOS from `default_tok_params` restoring add_special_tokens=True "
        f"(apertus-program #420)."
    )


def test_mm_bos_single_in_chat(client):
    """mm-bos-single-in-chat: an image+text chat prompt begins with exactly one BOS.

    The mm counterpart to ``core-no-double-bos-chat`` -- the exact path #420 was
    reported on. See ``_assert_single_bos_in_mm_chat``.
    """
    _assert_single_bos_in_mm_chat(client, _image("image_4827.png"), "image")


def test_mm_bos_single_in_chat_audio(client):
    """mm-bos-single-in-chat-audio: an audio+text chat prompt begins with one BOS.

    Audio triggers the same ``default_tok_params`` mm path as image, so it gets its
    own guard. See ``_assert_single_bos_in_mm_chat``.
    """
    _assert_single_bos_in_mm_chat(client, _audio("audio_fox.wav"), "audio")


# --- EOS ----------------------------------------------------------------------


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


def test_core_eos_not_appended_to_prompt(client):
    """core-eos-not-appended: default tokenization must not append an EOS to a
    raw prompt.

    The EOS-side mirror of the BOS ownership checks. A raw completion prompt is a
    prefix the model continues from; a tokenizer misconfigured with
    `add_eos_token=True` appends the EOS, so the model sees a premature stop token
    mid-context -> truncated or degenerate continuations. Asserts the last token of
    a raw DEFAULT tokenization (no `add_special_tokens` sent) is not a control/EOS
    token. (The thread noted #420 is BOS-only -- this pins that the EOS side stays
    clean too.)
    """
    try:
        raw = client.tokenize("The capital of France is Paris")
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    if not raw:
        pytest.skip("could not tokenize")
    try:
        last = client.detokenize([raw[-1]])
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")
    assert not _CONTROL_TOKEN_RE.search(last), (
        f"default tokenization appended a control/EOS token {last!r} to a raw "
        f"prompt (last ids {raw[-3:]}). A completion prompt must not end in an EOS -- "
        f"the model would see a premature stop mid-context."
    )
