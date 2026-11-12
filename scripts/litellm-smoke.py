"""Smoke test for the LiteLLM proxy.

Run:
    source scripts/dev-env.sh
    uv run --with anthropic --with openai python scripts/litellm-smoke.py [--models m1,m2] [--formats openai,anthropic]

Needs LLM_PROXY_URL and LLM_PROXY_KEY in the environment. Never prints the key.

Matrix: model x format x feature.
Features:
  plain        one short call
  stream       streaming call, text must arrive in chunks
  temperature  temperature=0.7 must be stripped by the proxy (else upstream 400)
  top_p_top_k  top_p=0.9 (+ top_k=40 on /v1/messages) must be stripped
  tools3       3-turn tool loop. Turn 1 must contain thinking, so the history test is not vacuous.
               OpenAI format replays history the way most frameworks do: content + tool_calls only,
               no thinking fields. Anthropic format replays the full content blocks (SDK-native).
  tools3_full  (openai only) same loop, but replays the full assistant message incl. thinking_blocks.
Rule probes (informational, expected outcome noted):
  sonnet_forced_tool_choice   tool_choice any -> upstream rejects on sonnet
  thinking_disabled           thinking {type: disabled}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import anthropic
import openai

URL = os.environ["LLM_PROXY_URL"].rstrip("/")
KEY = os.environ["LLM_PROXY_KEY"]
MAX_TOKENS = 4096

oai = openai.OpenAI(base_url=f"{URL}/v1", api_key=KEY, max_retries=0, timeout=600)
ant = anthropic.Anthropic(base_url=URL, api_key=KEY, max_retries=0, timeout=600)

USAGE: list[dict] = []

# ---------------------------------------------------------------- fake tools
TEMPS = {"paris": 14, "tokyo": 19, "oslo": 3, "cairo": 27}


def get_temperature(city: str) -> dict:
    return {"city": city, "celsius": TEMPS.get(city.lower(), 10)}


def get_forecast(city: str, days: int = 3) -> dict:
    base = TEMPS.get(city.lower(), 10)
    return {"city": city, "forecast_c": [base + d for d in range(int(days))]}


TOOL_IMPL = {"get_temperature": get_temperature, "get_forecast": get_forecast}

TOOLS_ANT = [
    {
        "name": "get_temperature",
        "description": "Current temperature in Celsius for one city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "get_forecast",
        "description": "Daily forecast (Celsius) for one city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"],
        },
    },
]
TOOLS_OAI = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS_ANT
]

LOOP_PROMPT = (
    "I am choosing between Paris and Tokyo for a trip and want the warmer place. "
    "Step 1: get the current temperature of both cities. "
    "Step 2: only after you have both, think about which one is warmer and get a 3-day forecast for that city only. "
    "Step 3: answer in two sentences with the city and the forecast numbers. "
    "Do not guess any number; use the tools."
)


# ---------------------------------------------------------------- helpers
def short_err(e: Exception) -> str:
    s = f"{type(e).__name__}: {getattr(e, 'status_code', '')} {e}"
    return s.replace("\n", " ")[:400]


def rec_usage(model: str, fmt: str, feature: str, inp: int, out: int) -> None:
    USAGE.append({"model": model, "format": fmt, "feature": feature, "in": inp or 0, "out": out or 0})


def oai_usage(r, model, feature):
    u = r.usage
    rec_usage(model, "openai", feature, u.prompt_tokens, u.completion_tokens)


def ant_usage(r, model, feature):
    rec_usage(model, "anthropic", feature, r.usage.input_tokens, r.usage.output_tokens)


# ---------------------------------------------------------------- openai format
def oai_plain(model, **extra):
    r = oai.chat.completions.create(
        model=model, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": "Reply with the word: pong"}], **extra
    )
    oai_usage(r, model, "plain")
    txt = r.choices[0].message.content or ""
    assert "pong" in txt.lower(), f"unexpected text {txt!r}"
    return f"ok text={txt.strip()[:30]!r}"


def oai_stream(model):
    chunks, text, usage = 0, "", None
    s = oai.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": "Count from 1 to 20, comma separated."}],
    )
    for ch in s:
        if ch.usage:
            usage = ch.usage
        if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
            chunks += 1
            text += ch.choices[0].delta.content
    if usage:
        rec_usage(model, "openai", "stream", usage.prompt_tokens, usage.completion_tokens)
    assert "20" in text and chunks >= 2, f"chunks={chunks} text={text[:60]!r}"
    return f"ok chunks={chunks}"


def oai_has_thinking(msg) -> bool:
    d = msg.model_dump()
    return bool(d.get("reasoning_content")) or bool(d.get("thinking_blocks"))


def oai_tools3(model, replay_full=False, **extra):
    msgs = [{"role": "user", "content": LOOP_PROMPT}]
    calls, thinking_turns, tool_calls_total = 0, [], 0
    feature = "tools3_full" if replay_full else "tools3"
    while calls < 6:
        r = oai.chat.completions.create(model=model, max_tokens=MAX_TOKENS, messages=msgs, tools=TOOLS_OAI, **extra)
        calls += 1
        oai_usage(r, model, feature)
        m = r.choices[0].message
        thinking_turns.append(oai_has_thinking(m))
        if not m.tool_calls:
            break
        tool_calls_total += len(m.tool_calls)
        if replay_full:
            am = m.model_dump(exclude_none=True)
        else:
            # What LangChain/CrewAI/most OpenAI-format frameworks send back: content + tool_calls only.
            am = {
                "role": "assistant",
                "content": m.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in m.tool_calls
                ],
            }
        msgs.append(am)
        for tc in m.tool_calls:
            out = TOOL_IMPL[tc.function.name](**json.loads(tc.function.arguments or "{}"))
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
    final = (m.content or "").strip()
    assert calls >= 3, f"loop ended after {calls} model calls (wanted >=3); final={final[:80]!r}"
    assert "tokyo" in final.lower(), f"final answer missing Tokyo: {final[:120]!r}"
    status = "ok" if thinking_turns[0] else "ok-BUT-NO-THINKING-turn1(vacuous)"
    return f"{status} calls={calls} tool_calls={tool_calls_total} thinking_per_turn={thinking_turns}"


# ---------------------------------------------------------------- anthropic format
def ant_plain(model, **extra):
    r = ant.messages.create(
        model=model, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": "Reply with the word: pong"}], **extra
    )
    ant_usage(r, model, "plain")
    txt = "".join(b.text for b in r.content if b.type == "text")
    assert "pong" in txt.lower(), f"unexpected text {txt!r}"
    return f"ok text={txt.strip()[:30]!r}"


def ant_stream(model):
    chunks, text = 0, ""
    with ant.messages.stream(
        model=model, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": "Count from 1 to 20, comma separated."}]
    ) as s:
        for t in s.text_stream:
            chunks += 1
            text += t
        final = s.get_final_message()
    ant_usage(final, model, "stream")
    assert "20" in text and chunks >= 2, f"chunks={chunks} text={text[:60]!r}"
    return f"ok chunks={chunks}"


def ant_tools3(model, **extra):
    msgs = [{"role": "user", "content": LOOP_PROMPT}]
    calls, thinking_turns, tool_calls_total = 0, [], 0
    while calls < 6:
        r = ant.messages.create(model=model, max_tokens=MAX_TOKENS, messages=msgs, tools=TOOLS_ANT, **extra)
        calls += 1
        ant_usage(r, model, "tools3")
        thinking_turns.append(any(b.type in ("thinking", "redacted_thinking") for b in r.content))
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            break
        tool_calls_total += len(uses)
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(TOOL_IMPL[u.name](**u.input))}
                    for u in uses
                ],
            }
        )
    final = "".join(b.text for b in r.content if b.type == "text").strip()
    assert calls >= 3, f"loop ended after {calls} model calls (wanted >=3); final={final[:80]!r}"
    assert "tokyo" in final.lower(), f"final answer missing Tokyo: {final[:120]!r}"
    status = "ok" if thinking_turns[0] else "ok-BUT-NO-THINKING-turn1(vacuous)"
    return f"{status} calls={calls} tool_calls={tool_calls_total} thinking_per_turn={thinking_turns}"


# ---------------------------------------------------------------- matrix
def cases(model: str, fmt: str):
    sonnet = "sonnet" in model
    if fmt == "openai":
        yield "plain", lambda: oai_plain(model), True
        yield "stream", lambda: oai_stream(model), True
        yield "temperature", lambda: oai_plain(model, temperature=0.7), True
        yield "top_p_top_k", lambda: oai_plain(model, top_p=0.9), True
        yield "tools3", lambda: oai_tools3(model), True
        yield "tools3_full", lambda: oai_tools3(model, replay_full=True), True
        yield "forced_tool_choice", lambda: oai_forced(model), not sonnet
        yield "thinking_disabled", lambda: oai_plain(model, extra_body={"thinking": {"type": "disabled"}}), None
    else:
        yield "plain", lambda: ant_plain(model), True
        yield "stream", lambda: ant_stream(model), True
        yield "temperature", lambda: ant_plain(model, temperature=0.7), True
        yield "top_p_top_k", lambda: ant_plain(model, top_p=0.9, top_k=40), True
        yield "tools3", lambda: ant_tools3(model), True
        yield "forced_tool_choice", lambda: ant_forced(model), not sonnet
        yield "thinking_disabled", lambda: ant_plain(model, thinking={"type": "disabled"}), None


def oai_forced(model):
    r = oai.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": "Temperature in Oslo?"}],
        tools=TOOLS_OAI,
        tool_choice="required",
    )
    oai_usage(r, model, "forced_tool_choice")
    return f"ok tool_calls={len(r.choices[0].message.tool_calls or [])}"


def ant_forced(model):
    r = ant.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": "Temperature in Oslo?"}],
        tools=TOOLS_ANT,
        tool_choice={"type": "any"},
    )
    ant_usage(r, model, "forced_tool_choice")
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="claude-haiku-5-5,claude-sonnet-5-5")
    ap.add_argument("--formats", default="openai,anthropic")
    ap.add_argument("--features", default="", help="comma list; empty = all")
    a = ap.parse_args()
    feats = set(filter(None, a.features.split(",")))
    rows, unexpected = [], 0
    for model in a.models.split(","):
        for fmt in a.formats.split(","):
            for feat, fn, expect_ok in cases(model, fmt):
                if feats and feat not in feats:
                    continue
                t0 = time.time()
                try:
                    detail, ok = fn(), True
                except Exception as e:  # noqa: BLE001
                    detail, ok = short_err(e), False
                    if os.environ.get("SMOKE_TRACE"):
                        traceback.print_exc()
                dt = time.time() - t0
                verdict = "PASS" if ok else "FAIL"
                if expect_ok is None:
                    tag = "info"
                elif ok == expect_ok:
                    tag = "as-expected"
                else:
                    tag = "UNEXPECTED"
                    unexpected += 1
                rows.append((model, fmt, feat, verdict, tag, f"{dt:.1f}s", detail))
                print(f"{model:18} {fmt:9} {feat:19} {verdict:4} {tag:11} {dt:5.1f}s  {detail}", flush=True)
    tin = sum(u["in"] for u in USAGE)
    tout = sum(u["out"] for u in USAGE)
    print(f"\nclient-side token usage: input={tin} output={tout} calls={len(USAGE)}")
    for m in sorted({u['model'] for u in USAGE}):
        i = sum(u["in"] for u in USAGE if u["model"] == m)
        o = sum(u["out"] for u in USAGE if u["model"] == m)
        print(f"  {m}: input={i} output={o}")
    print(f"unexpected results: {unexpected}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
