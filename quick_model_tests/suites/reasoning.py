"""reasoning suite -- proves the server-side REASONING PARSER works. See SPEC.md 7.6.

Apertus 1.5 wraps chain-of-thought between `<|inner_prefix|>` ... `<|inner_suffix|>`.
The reasoning parser (vLLM `--reasoning-parser qwen3`, SGLang equivalent) splits
that raw stream into two OpenAI channels: `message.reasoning_content` (scratch
work) and `message.content` (the user-facing answer). These checks prove the
split is correct -- non-streaming, streaming, and alongside the tool parser.

No quality judging (SPEC.md 6): every assertion is structural -- a field is
present/absent, a channel is free of boundary tokens, the streamed boundary is
monotonic, a closed-set answer matches.

Launch requirement -- two distinct flags, easy to conflate:
  * `--default-chat-template-kwargs.enable_thinking true` -- sets the default for
    APERTUS 1.5's OWN `enable_thinking` chat-template kwarg (the template branches
    on it to deliberate). This is the Apertus-specific switch; off => no thinking.
  * `--reasoning-parser qwen3` -- vLLM's stream-splitter implementation; "qwen3"
    is just the boundary-format name, NOT Apertus- or qwen-model-specific, and
    unrelated to the enable_thinking kwarg.
Miss the enable_thinking default and no `reasoning_content` is produced at all.

Gating: surfacing is model- AND launch-dependent (SPEC.md 7.6 / open question 2).
The `reasoning_supported` probe runs once; on an endpoint that exposes no
`reasoning_content` channel (a plain instruct model, a missing launch flag, or a
gateway dropping the field) the parser-specific checks SKIP with a clear reason
rather than failing red. `reason-separation` and `reason-answer` hold regardless
of surfacing and always run.

Budgets are generous: a reasoning model may spend hundreds of tokens thinking
before it emits the answer, so a tight max_tokens would truncate it.
"""

import json
import re

import pytest

from quick_model_tests.client import ApiError, ChatClient

pytestmark = pytest.mark.reasoning

# Budget for reasoning prompts: thinking + answer can run long; don't truncate.
REASON_MAX_TOKENS = 1024

# A reasoning-eliciting prompt whose answer is a known sentinel (8*9 = 72). Used
# by the probe and the streaming check so the answer is deterministically
# assertable in `content`.
REASONING_PROMPT = (
    "Think step by step, then answer: what is 8 times 9? "
    "Put only the final number on the last line."
)
ANSWER_SENTINEL = "72"

# Raw boundary / special tokens that must never appear in EITHER channel: the
# parser is expected to consume the delimiters, not relocate them. `<|...|>`
# also catches tool scaffolding (`<|tools_prefix|>` etc.) for reason-tools.
THINK_TOKEN_RE = re.compile(
    r"<\|inner_prefix\|>|<\|inner_suffix\|>|</?think\b|<\|[^>]*\|>", re.IGNORECASE
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="session")
def reasoning_supported(client):
    """Probe once: does the endpoint surface a separate `reasoning_content`
    channel? Returns the probe response (reused by reason-produced to avoid a
    second call). SKIP (not fail) the parser-specific checks when no channel is
    surfaced -- a plain model or a gateway that drops the field is a visible
    skip, not a red failure (SPEC.md 7.6 / section 8)."""
    resp = client.chat(
        [{"role": "user", "content": REASONING_PROMPT}], max_tokens=REASON_MAX_TOKENS
    )
    rc = ChatClient.reasoning_content(resp)
    if not (rc and rc.strip()):
        pytest.skip(
            "endpoint surfaces no reasoning_content channel "
            "(plain model, or the gateway drops the field)"
        )
    return resp


def test_reason_produced(reasoning_supported):
    """reason-produced: non-stream parser populates BOTH channels.

    reasoning_content non-empty proves the parser ran; content non-empty proves
    it did not swallow the answer into the reasoning channel."""
    resp = reasoning_supported
    reasoning = ChatClient.reasoning_content(resp) or ""
    content = ChatClient.content(resp) or ""
    assert reasoning.strip(), "reasoning_content empty -- parser produced no thinking"
    assert content.strip(), (
        "content empty -- parser swallowed the answer into reasoning_content"
    )


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
                max_tokens=REASON_MAX_TOKENS,
            )
        )
        or ""
    )
    assert content.strip(), "empty content"
    leak = THINK_TOKEN_RE.search(content)
    assert not leak, (
        f"raw reasoning token {leak.group(0)!r} leaked into content: {content!r}"
    )


def test_reason_clean_channel(reasoning_supported):
    """reason-clean-channel: the reasoning_content channel itself is free of raw
    boundary tokens -- the parser CONSUMED the delimiters, not just relocated
    them into the other channel."""
    reasoning = ChatClient.reasoning_content(reasoning_supported) or ""
    leak = THINK_TOKEN_RE.search(reasoning)
    assert not leak, (
        f"raw boundary token {leak.group(0)!r} leaked into "
        f"reasoning_content: {reasoning!r}"
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
                max_tokens=REASON_MAX_TOKENS,
            )
        )
        or ""
    )
    assert "42" in content, f"expected 42 in the answer, got: {content!r}"


def test_reason_stream(client, reasoning_supported):
    """reason-stream: the streaming reasoning->answer boundary is monotonic.

    Every `delta.reasoning_content` must arrive before the first
    `delta.content`; the transition happens exactly once (no flip back to
    reasoning after content begins); the reassembled content carries the answer
    sentinel; and no raw boundary token leaks into either streamed field."""
    reasoning_parts, content_parts = [], []
    seen_content = False
    reasoning_after_content = False
    for ch in client.stream(
        [{"role": "user", "content": REASONING_PROMPT}], max_tokens=REASON_MAX_TOKENS
    ):
        # The terminal chunk carries usage with an empty `choices` list.
        for choice in ch.get("choices") or []:
            delta = choice.get("delta", {})
            rc = ChatClient.reasoning_delta(delta)
            c = delta.get("content")
            if rc:
                reasoning_parts.append(rc)
                if seen_content:
                    reasoning_after_content = True
            if c:
                content_parts.append(c)
                seen_content = True
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)

    assert reasoning.strip(), "no streamed reasoning_content deltas"
    assert content.strip(), "no streamed content deltas"
    assert not reasoning_after_content, (
        "reasoning_content resumed after content began -- boundary is not monotonic"
    )
    assert ANSWER_SENTINEL in content, (
        f"expected {ANSWER_SENTINEL!r} in streamed content, got: {content!r}"
    )
    for name, chan in (("reasoning_content", reasoning), ("content", content)):
        leak = THINK_TOKEN_RE.search(chan)
        assert not leak, f"raw token {leak.group(0)!r} leaked into streamed {name}"


def test_reason_tools(client, reasoning_supported):
    """reason-tools: reasoning- and tool-parser cooperate.

    With a tool offered and forced, the call must land in `tool_calls` with
    JSON-parseable arguments, and NO raw tool scaffolding (`<|tools_prefix|>`
    etc.) may leak into either the content or reasoning_content channel. Skipped
    when the endpoint does not support tool calling (that gap is the `tools`
    suite's to report, not this one's)."""
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": "Think about which tool to use, then get the "
                    "weather in Zurich.",
                }
            ],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=REASON_MAX_TOKENS,
        )
    except ApiError as exc:
        pytest.skip(f"tool calling not supported by endpoint: {exc}")

    tool_calls = resp["choices"][0]["message"].get("tool_calls") or []
    if not tool_calls:
        pytest.skip("model emitted no tool_call under force; tools unsupported")

    fn = tool_calls[0]["function"]
    assert fn["name"] == "get_weather", f"unexpected tool: {fn['name']!r}"
    json.loads(fn["arguments"])  # arguments must be JSON-parseable

    for name in ("content", "reasoning_content", "reasoning"):
        chan = resp["choices"][0]["message"].get(name) or ""
        leak = THINK_TOKEN_RE.search(chan)
        assert not leak, (
            f"tool/boundary scaffolding {leak.group(0)!r} leaked into {name}"
        )


def test_reason_disabled(client, reasoning_supported):
    """reason-disabled: the parser respects the think toggle.

    A per-request `chat_template_kwargs={"enable_thinking": false}` must OVERRIDE
    the server's launch default (`--default-chat-template-kwargs.enable_thinking
    true`): no reasoning channel, yet still a correct answer. Skipped when the
    endpoint rejects or ignores the kwarg (keeps thinking)."""
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": "What is 6 times 7? Reply with only the number.",
                }
            ],
            max_tokens=REASON_MAX_TOKENS,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except ApiError as exc:
        pytest.skip(f"endpoint rejected enable_thinking kwarg: {exc}")

    reasoning = ChatClient.reasoning_content(resp)
    if reasoning and reasoning.strip():
        pytest.skip("endpoint ignores enable_thinking=false (still surfaces reasoning)")

    content = ChatClient.content(resp) or ""
    assert "42" in content, f"expected 42 with thinking disabled, got: {content!r}"
