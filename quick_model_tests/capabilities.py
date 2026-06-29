"""Model capability report -- probe a served model and print what it supports.

This is a *diagnostic*, separate from the pass/fail test gate (the suites in
`suites/` hard-fail when something is broken). The report instead answers
"what does this endpoint actually do?" across the functional paths, with a
status per capability:

    yes      capability works (probe got the expected result)
    no       capability absent / not offered -- including the server *rejecting*
             the probe (e.g. tools 400 "requires --tool-call-parser", audio 500
             "install vllm[audio]"): the feature is not available on this build
    broken   capability is offered but errors / misbehaves mid-use (a server bug,
             e.g. the multi-turn tool round-trip 400)
    error    the probe itself could not run -- auth (401/403), connection, or an
             unexpected exception

This is the DEFAULT mode of `quick-model-tests` (the pass/fail gate is opt-in via
--test-framework). Run it:
    quick-model-tests                                   # default model
    quick-model-tests --model swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo-tools
    quick-model-tests --model A --model B               # comparison table
    quick-model-tests --json

All probes are deterministic and self-contained (temperature=0); see SPEC.md
section 6 and open question 4 (capability matrix).
"""

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .client import ApiError, ChatClient
from .config import Config

_ASSETS = Path(__file__).resolve().parent / "assets"

YES, NO, BROKEN, ERROR = "yes", "no", "broken", "error"

# Tool-orchestration / special-token markup that must never leak into content.
_TOOL_MARKUP_RE = re.compile(
    r"<\|[^>]*\|>"
    r"|</?(?:info|bash|tool_call|tool_calls|function_call|think"
    r"|inner_prefix|inner_suffix)\b",
    re.IGNORECASE,
)
_AGENT_SYS = (
    "You are an autonomous coding agent operating in a terminal. You have a "
    "`bash` tool. When the user asks you to inspect or change the filesystem, "
    "CALL the bash tool with the command. Do not describe the command in prose."
)
_BASH = {
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

_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
_LOOKUP = {
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


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""


def _calls(resp):
    return resp["choices"][0]["message"].get("tool_calls") or []


# -- probes: each takes a client and returns a Result --------------------------

# Generous budget: reasoning models (e.g. Qwen3.5 with --reasoning-parser) think
# ~150+ tokens before answering even trivial prompts, so a tight max_tokens would
# truncate the answer (finish_reason=length, empty content) and look "broken".
_CHAT_PROMPT = "What is the capital of France? Answer in one short sentence."


def probe_chat(client):
    resp = client.chat([{"role": "user", "content": _CHAT_PROMPT}], max_tokens=512)
    content = ChatClient.content(resp)
    if content and content.strip():
        return Result("chat.basic", YES, "returns non-empty content")
    fr = resp["choices"][0].get("finish_reason")
    detail = "empty content"
    if fr == "length":
        detail += (
            " (finish_reason=length -- model still generating; a reasoning "
            "model may need a larger max_tokens or enable_thinking=false)"
        )
    return Result("chat.basic", BROKEN, detail)


def probe_streaming(client):
    chunks = list(
        client.stream([{"role": "user", "content": _CHAT_PROMPT}], max_tokens=512)
    )
    text = ChatClient.stream_text(chunks)
    if len(chunks) >= 2 and text.strip():
        return Result("chat.streaming", YES, f"{len(chunks)} SSE chunks, [DONE] seen")
    return Result("chat.streaming", BROKEN, f"{len(chunks)} chunks, text={text!r}")


def probe_tools(client):
    resp = client.chat(
        [{"role": "user", "content": "Weather in Paris? Use the tool."}],
        tools=[_WEATHER],
        tool_choice="required",
        max_tokens=128,
    )
    calls = _calls(resp)
    if not calls:
        return Result("tools.basic", NO, "no tool_calls even when forced")
    fn = calls[0]["function"]
    try:
        args = json.loads(fn["arguments"])
    except (ValueError, KeyError) as e:
        return Result("tools.basic", BROKEN, f"arguments not JSON: {e}")
    if fn["name"] != "get_weather":
        return Result("tools.basic", BROKEN, f"wrong fn {fn['name']!r}")
    return Result("tools.basic", YES, f"get_weather({args})")


def probe_tools_named(client):
    resp = client.chat(
        [{"role": "user", "content": "Hello."}],
        tools=[_WEATHER, _LOOKUP],
        tool_choice={"type": "function", "function": {"name": "lookup_code"}},
        max_tokens=128,
    )
    calls = _calls(resp)
    if calls and calls[0]["function"]["name"] == "lookup_code":
        return Result("tools.choice_named", YES, "named tool_choice honored")
    got = calls[0]["function"]["name"] if calls else None
    return Result("tools.choice_named", NO, f"forced lookup_code, got {got!r}")


def probe_tools_streaming(client):
    name, args = None, ""
    for chunk in client.stream(
        [{"role": "user", "content": "Weather in Paris? Use the tool."}],
        tools=[_WEATHER],
        max_tokens=128,
    ):
        for choice in chunk.get("choices") or []:
            for tc in choice.get("delta", {}).get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name") or name
                args += fn.get("arguments") or ""
    if not args:
        return Result("tools.streaming", NO, "no streamed tool_call deltas")
    try:
        json.loads(args)
    except ValueError as e:
        return Result("tools.streaming", BROKEN, f"deltas not JSON: {e}")
    return Result("tools.streaming", YES, f"{name} args accumulate to valid JSON")


def probe_tools_parallel(client):
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "Get the weather in Paris AND in Tokyo. "
                "Call the tool separately for each city.",
            }
        ],
        tools=[_WEATHER],
        max_tokens=256,
    )
    n = len(_calls(resp))
    if n >= 2:
        return Result("tools.parallel", YES, f"{n} calls in one turn")
    return Result("tools.parallel", NO, f"only {n} call for a 2-target prompt")


def probe_tools_multiturn(client):
    """Round-trip: call -> echo assistant tool_calls + tool result -> final."""
    msgs = [{"role": "user", "content": "Look up the code for 'alpha'. Use the tool."}]
    resp = client.chat(msgs, tools=[_LOOKUP], max_tokens=128)
    calls = _calls(resp)
    if not calls:
        return Result("tools.multiturn_loop", NO, "model did not call the tool")
    m = resp["choices"][0]["message"]
    msgs.append(
        {
            "role": "assistant",
            "content": m.get("content"),
            "tool_calls": m["tool_calls"],
        }
    )
    msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "4827"})
    try:
        final = client.chat(msgs, tools=[_LOOKUP], max_tokens=128)
    except ApiError as e:
        return Result(
            "tools.multiturn_loop",
            BROKEN,
            f"re-sending tool_calls turn -> {e.status}: {e.body[:120]}",
        )
    content = ChatClient.content(final) or ""
    if "4827" in content:
        return Result("tools.multiturn_loop", YES, "sentinel survived the round-trip")
    return Result("tools.multiturn_loop", BROKEN, f"sentinel lost: {content!r}")


def probe_tools_no_leak(client):
    """Agentic prompt: does tool intent become a structured call, or leak as
    `<info>`/`<bash>` markup in content? (The opencode-breaking Apertus bug.)"""
    resp = client.chat(
        [
            {"role": "system", "content": _AGENT_SYS},
            {"role": "user", "content": "Create a directory called rob-test here."},
        ],
        tools=[_BASH],
        max_tokens=256,
    )
    content = (ChatClient.content(resp) or "").strip()
    leak = _TOOL_MARKUP_RE.search(content)
    if leak:
        return Result(
            "tools.no_markup_leak",
            BROKEN,
            f"{leak.group(0)!r} markup leaked into content: {content[:80]!r}",
        )
    calls = _calls(resp)
    if calls and content.lower() in {c["function"]["name"].lower() for c in calls}:
        return Result(
            "tools.no_markup_leak",
            BROKEN,
            f"bare tool name leaked into content beside the call: {content!r}",
        )
    if calls:
        return Result("tools.no_markup_leak", YES, "structured call, clean content")
    return Result("tools.no_markup_leak", NO, "no tool call for an agentic action")


def _data_url(name: str, mime: str) -> str:
    raw = (_ASSETS / name).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def probe_vision(client):
    """Can the model read a numeric sentinel out of an image?

    A rejection (4xx/5xx) means vision is not available on this deployment -> NO
    (capability absent), not ERROR (which is for probe-side failures).
    """
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What number is in this image? Digits only.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _data_url("image_4827.png", "image/png")
                            },
                        },
                    ],
                }
            ],
            max_tokens=32,
        )
    except ApiError as e:
        return Result("vision.image", NO, f"image rejected ({e.status}): {e.body[:60]}")
    content = ChatClient.content(resp) or ""
    if "4827" in content:
        return Result("vision.image", YES, "read the image sentinel (4827)")
    return Result("vision.image", NO, f"did not read the sentinel: {content[:60]!r}")


def probe_audio(client):
    """Can the model transcribe a short audio clip?

    A rejection (e.g. 500 'install vllm[audio]') means audio is not available on
    this deployment -> NO (capability absent), not ERROR.
    """
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the audio."},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": _data_url("audio_fox.wav", "audio/wav")
                            },
                        },
                    ],
                }
            ],
            max_tokens=64,
        )
    except ApiError as e:
        return Result(
            "audio.transcribe", NO, f"audio rejected ({e.status}): {e.body[:50]}"
        )
    content = (ChatClient.content(resp) or "").lower()
    if "fox" in content:
        return Result("audio.transcribe", YES, "transcribed the audio sentinel")
    return Result("audio.transcribe", NO, f"did not transcribe: {content[:60]!r}")


def probe_reasoning(client):
    """How is thinking surfaced -- separate field vs inline <think> tags?

    Checks both `reasoning_content` (OpenAI/vLLM non-stream) and `reasoning`
    (this stack's stream-delta key / some builds). A gateway that strips these
    fields (the serving-api ConfigDict bug) shows up here as `no` even though the
    model is thinking -- so this row is a clean before/after for that fix.
    """
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "Think step by step, then answer: what is 6 times 7?",
            }
        ],
        max_tokens=512,
    )
    msg = resp["choices"][0]["message"]
    field = next((k for k in ("reasoning_content", "reasoning") if msg.get(k)), None)
    if field:
        return Result("reasoning.surface", YES, f"separate `{field}` field")
    content = msg.get("content") or ""
    if "<think>" in content or "<|inner_prefix|>" in content:
        return Result(
            "reasoning.surface", BROKEN, "raw <think>/<|inner_*|> leaks into content"
        )
    return Result("reasoning.surface", NO, "no separate field, no inline tags")


# (canonical capability name, probe). The name is used for BOTH the success and
# the error path so rows line up across models in the comparison table.
PROBES = [
    ("chat.basic", probe_chat),
    ("chat.streaming", probe_streaming),
    ("tools.basic", probe_tools),
    ("tools.choice_named", probe_tools_named),
    ("tools.streaming", probe_tools_streaming),
    ("tools.parallel", probe_tools_parallel),
    ("tools.multiturn_loop", probe_tools_multiturn),
    ("tools.no_markup_leak", probe_tools_no_leak),
    ("vision.image", probe_vision),
    ("audio.transcribe", probe_audio),
    ("reasoning.surface", probe_reasoning),
]


def run(config: Config) -> list:
    client = ChatClient(config)
    results = []
    for name, probe in PROBES:
        try:
            results.append(probe(client))
        except ApiError as e:
            # The server *rejecting* a capability probe means the feature is not
            # available on this deployment -> NO. ERROR is reserved for the probe
            # being unable to run at all (auth, connection, unexpected).
            status = ERROR if e.status in (401, 403) else NO
            results.append(Result(name, status, f"HTTP {e.status}: {e.body[:100]}"))
        except Exception as e:  # noqa: BLE001 -- diagnostic must never crash the report
            results.append(Result(name, ERROR, f"{type(e).__name__}: {e}"))
    return results


_ICON = {YES: "✔", NO: "✗", BROKEN: "⚠", ERROR: "!"}


def report(config: Config, as_json: bool = False) -> int:
    results = run(config)
    if as_json:
        print(
            json.dumps(
                {
                    "model": config.model,
                    "api_base": config.api_base,
                    "capabilities": [r.__dict__ for r in results],
                },
                indent=2,
            )
        )
        return 0
    print(f"Capability report for {config.model}")
    print(f"  endpoint: {config.api_base}\n")
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  {_ICON[r.status]} {r.name:<{width}}  {r.status:<7}  {r.detail}")
    broken = [r for r in results if r.status in (BROKEN, ERROR)]
    print(f"\n  {len(results)} probes; {len(broken)} broken/error")
    return 0


def report_compare(configs: list, as_json: bool = False, detail: bool = False) -> int:
    """Compare >=2 models as a markdown table.

    Transposed (capabilities down, models across) with one status glyph per cell,
    and columns labelled M1/M2/... so long model ids never widen the table -- the
    full ids go in a legend below. With detail=True, the failure reasons (every
    non-`yes` cell) are listed as per-model footnotes, keeping the table narrow.
    """
    runs = [(cfg.model, {r.name: r for r in run(cfg)}) for cfg in configs]
    # capability order, preserved across models (union, first-seen order)
    caps = []
    for _, res in runs:
        for name in res:
            if name not in caps:
                caps.append(name)

    if as_json:
        print(
            json.dumps(
                {
                    "api_base": configs[0].api_base,
                    "models": [m for m, _ in runs],
                    "capabilities": {
                        cap: {
                            m: (res[cap].__dict__ if cap in res else None)
                            for m, res in runs
                        }
                        for cap in caps
                    },
                },
                indent=2,
            )
        )
        return 0

    cols = [f"M{i + 1}" for i in range(len(runs))]
    cap_w = max(len(c) for c in caps)
    print(f"Capability comparison ({configs[0].api_base})\n")
    print(f"| {'Capability':<{cap_w}} | " + " | ".join(cols) + " |")
    print(f"|{'-' * (cap_w + 2)}|" + "----|" * len(cols))
    for cap in caps:
        cells = []
        for _, res in runs:
            r = res.get(cap)
            cells.append(_ICON.get(r.status, " ") if r else " ")
        # pad each single-glyph cell to its column-label width for alignment
        cells = [f"{c:<{len(col)}}" for c, col in zip(cells, cols)]
        print(f"| {cap:<{cap_w}} | " + " | ".join(cells) + " |")
    print("\nLegend: ✔ yes · ✗ no · ⚠ broken · ! error")
    for i, (m, _) in enumerate(runs):
        print(f"M{i + 1} = {m}")

    if detail:
        print("\nFailure details (non-yes cells):")
        cap_w = max(len(c) for c in caps)
        for i, (m, res) in enumerate(runs):
            fails = [res[c] for c in caps if c in res and res[c].status != YES]
            print(f"\nM{i + 1} = {m}")
            if not fails:
                print("  (all probes passed)")
            for r in fails:
                print(f"  {_ICON[r.status]} {r.name:<{cap_w}}  {r.detail}")
    return 0
