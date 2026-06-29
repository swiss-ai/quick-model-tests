"""tools suite -- OpenAI-compatible function / tool calling. See SPEC.md 7.3.

Deterministic structural checks only (see SPEC.md section 6): tool-call shape,
JSON-parseable `arguments`, required-key presence, closed-set sentinels. No
LLM-as-judge.

Target model: a tool-capable Apertus, e.g.
    --model swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo-tools
The whole suite HARD-FAILS (via the `tools_supported` probe) when the configured
model does not emit tool_calls -- no silent skips.

Probed wire behavior (2026-06, sft-dpo-tools model):
  - forced call -> finish_reason="tool_calls", message.content=null,
    tool_calls[0].function.{name, arguments(JSON string)}
  - tool_choice "required" and {"function":{"name":...}} both honored
  - streamed tool calls arrive as deltas; the final chunk carries usage with an
    EMPTY choices list -- callers must guard `choices`
  - PARALLEL calls: FAIL -- a 2-target prompt yields a single call
  - MULTI-TURN loop: FAIL -- echoing an assistant tool_calls message back 400s
    with "can only concatenate str (not dict) to str" (server chat-template bug)
  - TOOL-MARKUP LEAK: FAIL -- with an agentic system prompt the model emits its
    tool intent as `<info>`/`<bash>` text in `content` with EMPTY tool_calls,
    instead of a structured call (breaks opencode etc.). See SPEC.md 7.3 / 7.7.
"""

import json
import re

import pytest

from quick_model_tests.client import ApiError, ChatClient

pytestmark = pytest.mark.tools

# Tool-orchestration / special-token markup that must never appear in
# user-visible content -- the model's tool intent belongs in `tool_calls`, not
# as literal tags. Catches the Apertus-1.5 `<info>...</info>` / `<bash>...</bash>`
# leak and any `<|...|>` / <think> special-token leak (SPEC.md line 148-149).
TOOL_MARKUP_RE = re.compile(
    r"<\|[^>]*\|>"
    r"|</?(?:info|bash|tool_call|tool_calls|function_call|think"
    r"|inner_prefix|inner_suffix)\b",
    re.IGNORECASE,
)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_code",
        "description": "Look up a secret numeric code by name.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}


def _tool_calls(resp: dict) -> list:
    """tool_calls for choice 0, normalized to a list (never None)."""
    return resp["choices"][0]["message"].get("tool_calls") or []


@pytest.fixture(scope="session")
def tools_supported(client):
    """Probe once: does the configured model emit tool_calls when forced?

    HARD FAIL (not skip) when the target model lacks tool calling -- pointing
    this suite at a non-tools model is treated as an error so the gate goes red,
    never silently green. Run it against a tool-capable build (see module doc).
    """
    try:
        resp = client.chat(
            [{"role": "user", "content": "What is the weather in Paris?"}],
            # generous budget: a reasoning model may think before it tool-calls
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=256,
        )
    except ApiError as e:
        pytest.fail(
            f"model {client.config.model!r} rejected tool calling "
            f"({e.status}): {e.body[:200]}"
        )
    assert _tool_calls(resp), (
        f"model {client.config.model!r} produced no tool_calls when forced"
    )
    return True


def test_tools_single(client, tools_supported):
    """tools-single: forced use -> expected name + schema-valid arguments."""
    resp = client.chat(
        [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
    )
    calls = _tool_calls(resp)
    assert calls, "expected a tool call, got none"
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather", f"unexpected tool: {fn['name']!r}"
    args = json.loads(fn["arguments"])  # must be JSON-parseable
    assert "city" in args, f"required key 'city' missing: {args!r}"


def test_tools_choice_required(client, tools_supported):
    """tools-choice (required): tool_choice='required' forces a call even when
    the prompt does not obviously call for one."""
    resp = client.chat(
        [{"role": "user", "content": "Just say hello."}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        max_tokens=128,
    )
    assert _tool_calls(resp), "tool_choice='required' did not force a call"
    assert resp["choices"][0]["finish_reason"] == "tool_calls"


def test_tools_choice_named(client, tools_supported):
    """tools-choice (named): a specific function name forces THAT function."""
    resp = client.chat(
        [{"role": "user", "content": "Hello there."}],
        tools=[WEATHER_TOOL, LOOKUP_TOOL],
        tool_choice={"type": "function", "function": {"name": "lookup_code"}},
        max_tokens=128,
    )
    calls = _tool_calls(resp)
    assert calls, "named tool_choice did not produce a call"
    assert calls[0]["function"]["name"] == "lookup_code", (
        f"forced wrong function: {calls[0]['function']['name']!r}"
    )


def test_tools_stream(client, tools_supported):
    """tools-stream: streamed tool-call arg deltas accumulate to parseable JSON.

    The final chunk carries usage with an empty `choices` list -- guarded here.
    """
    name, args = None, ""
    saw_delta = False
    for chunk in client.stream(
        [{"role": "user", "content": "Weather in Paris? Use the tool."}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
    ):
        for choice in chunk.get("choices") or []:
            for tc in choice.get("delta", {}).get("tool_calls") or []:
                saw_delta = True
                fn = tc.get("function", {})
                if fn.get("name"):
                    name = fn["name"]
                if fn.get("arguments"):
                    args += fn["arguments"]
    assert saw_delta, "no streamed tool_call deltas"
    assert name == "get_weather", f"unexpected streamed tool: {name!r}"
    parsed = json.loads(args)  # accumulated deltas must form valid JSON
    assert "city" in parsed, f"required key 'city' missing: {parsed!r}"


def test_tools_none(client, tools_supported):
    """tools-none: tools offered but prompt irrelevant -> plain content, no call."""
    resp = client.chat(
        [{"role": "user", "content": "Reply with exactly the word: hello"}],
        tools=[WEATHER_TOOL],
        max_tokens=64,
    )
    assert not _tool_calls(resp), "model called a tool for an irrelevant prompt"
    content = ChatClient.content(resp)
    assert content and content.strip(), "expected normal content, got none"


def test_tools_parallel(client, tools_supported):
    """tools-parallel: a 2-target prompt must yield >=2 calls.

    HARD FAIL when the model emits a single call. Parallel calling is currently
    unsupported on the -tools build (2-target prompt -> 1 call), so this fails
    today by design -- the gap is a red failure, not a hidden skip. See SPEC.md
    7.3 / open question 3.
    """
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "Get the weather in Paris AND in Tokyo. "
                "Call the tool separately for each city.",
            }
        ],
        tools=[WEATHER_TOOL],
        max_tokens=256,
    )
    calls = _tool_calls(resp)
    assert len(calls) >= 2, (
        f"expected >=2 parallel tool calls, got {len(calls)} (parallel unsupported)"
    )
    names = [c["function"]["name"] for c in calls]
    assert all(n == "get_weather" for n in names), names


def test_tools_loop(client, tools_supported):
    """tools-loop: call -> append tool result with sentinel 4827 -> final
    content contains '4827'.

    HARD FAIL today: the round-trip 400s the moment the assistant tool-call turn
    is re-sent (server chat-template bug, "can only concatenate str (not dict) to
    str"), so the model never sees the sentinel. This is a real, loud failure
    until the served template is fixed -- not a skip or xfail. See SPEC.md 7.3.
    """
    messages = [
        {"role": "user", "content": "Look up the code for 'alpha'. Use the tool."}
    ]
    resp = client.chat(messages, tools=[LOOKUP_TOOL], max_tokens=256)
    call = _tool_calls(resp)[0]
    msg = resp["choices"][0]["message"]
    messages.append(
        {
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": msg["tool_calls"],
        }
    )
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": "4827"})
    final = client.chat(messages, tools=[LOOKUP_TOOL], max_tokens=256)
    content = ChatClient.content(final) or ""
    assert "4827" in content, f"sentinel from tool result missing: {content!r}"


# Agentic system prompt that triggers the Apertus-1.5 `-tools` leak: instead of a
# structured tool call, the model emits `<info>...</info>` / `<bash>...</bash>`
# tool intent as plain content with EMPTY tool_calls -- so agents (opencode etc.)
# never execute anything. See module docstring and SPEC.md 7.3 / 7.7.
_AGENT_SYS = (
    "You are an autonomous coding agent operating in a terminal. You have a "
    "`bash` tool. When the user asks you to inspect or change the filesystem, "
    "CALL the bash tool with the command. Do not describe the command in prose."
)
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the shell and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"],
        },
    },
}


def test_tools_no_content_leak(client, tools_supported):
    """tools-leak: a tool call must not leak scaffolding into user content.

    When the model makes a tool call, user-visible `content` must be empty/None
    or genuine prose -- never the bare tool name or tool-orchestration markup
    (`<info>`/`<bash>`/`<|...|>`/`<think>`). The Apertus-1.5 `-tools` build
    deterministically returns content='bash' (the tool name) beside the call,
    and under opencode's protocol leaks `<info>...</info>` markup -- i.e. the
    served tool-call parser is not cleanly separating tool scaffolding from
    content, so agents render junk / execute nothing. Qwen3.5 returns
    content=None here. SPEC.md line 148-149 / 7.7.
    """
    resp = client.chat(
        [
            {"role": "system", "content": _AGENT_SYS},
            {"role": "user", "content": "Create a directory called rob-test here."},
        ],
        tools=[BASH_TOOL],
        max_tokens=256,
    )
    calls = _tool_calls(resp)
    assert calls, (
        "agentic action produced no structured tool_call; an agent (opencode) "
        f"would execute nothing. content={ChatClient.content(resp)!r}"
    )
    content = (ChatClient.content(resp) or "").strip()
    leak = TOOL_MARKUP_RE.search(content)
    assert not leak, (
        f"tool-orchestration markup {leak.group(0)!r} leaked into content: {content!r}"
    )
    leaked_names = {c["function"]["name"].lower() for c in calls}
    assert content.lower() not in leaked_names, (
        f"bare tool name leaked into content beside the call: {content!r}"
    )
