"""Export one agent's traces of one batch from Langfuse's ClickHouse as a Langfuse-style JSON file.

    python3 scripts/export_langfuse.py --batch smoke --agent lg-supervisor-team [--out f.json.gz]
        [--envelope array|data|object|jsonl] [--trace-ids id,id] [--exclude-ids-file ids.txt]

Langfuse v4 runs in events_only mode (no /api/public/traces), so this reads `events_full` (events_core truncates input and output to 200 chars) through
`kubectl exec` into clickhouse-0, the same way as scripts/inspect_trace.py. The password stays inside the pod.

Output: one trace object per Langfuse trace, in the Langfuse public-API shape, observations nested:
  {id, name, timestamp, input, output, tags, sessionId, userId, environment, release, metadata,
   level, statusMessage, latency, totalCost, observations: [
     {id, traceId, parentObservationId, type, name, startTime, endTime, completionStartTime, input, output,
      model, modelParameters, usage, usageDetails, costDetails, totalCost, level, statusMessage, metadata,
      toolCalls, toolCallNames, toolDefinitions}]}
Envelope: `array` = [trace, ...] (default), `data` = {"data": [...]}, `object` = one trace (needs exactly one).
Only traces whose ROOT span (parent_span_id = '') has both tags are selected. The root is exported last,
so a trace that is still being written never qualifies.
Prints a JSON summary line to stdout: {"agent", "batch", "traces", "observations", "trace_ids", "out"}.
"""
import argparse
import gzip
import json
import re
import subprocess
import sys
import time

K = ["kubectl", "--context", "gepton-base", "-n", "experiments"]
CH = ['clickhouse-client --password "$CLICKHOUSE_PASSWORD" --format JSONEachRow '
      '--output_format_json_quote_64bit_integers=0 --output_format_json_quote_decimals=0 -q "$(cat)"']
SAFE = re.compile(r"^[A-Za-z0-9_.:-]+$")
# Metadata keys that are transport noise or identify our Langfuse project. Drop them.
DROP_META = ("resourceAttributes.", "scope.")

# Credentials that can appear in payloads (e.g. a LiteLLM key in llm.invocation_parameters). Replaced before export.
SECRET_RES = [
    (re.compile(r"\b(sk-lf-|pk-lf-)[A-Za-z0-9-]{8,}"), r"\1<redacted>"),
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}"), "sk-<redacted>"),
    (re.compile(r"\b(xox[abposr]-)[A-Za-z0-9-]{8,}"), r"\1<redacted>"),
    (re.compile(r"(Bearer )[A-Za-z0-9._~+/=-]{12,}"), r"\1<redacted>"),
    (re.compile(r'((?:\\?["\'])(?:api_key|apiKey|api-key|x-api-key|authorization|password|secret_key)(?:\\?["\'])\s*:\s*\\?["\'])'
                r'(?!<redacted>)[^"\'\\]{6,}', re.I), r"\1<redacted>"),
]


def scrub(s):
    if not isinstance(s, str) or not s:
        return s
    for rx, rep in SECRET_RES:
        s = rx.sub(rep, s)
    return s


COLS = """trace_id, span_id, parent_span_id, start_time, end_time, completion_start_time, name, type, level,
status_message, tags, session_id, user_id, trace_name, environment, release, input, output, provided_model_name,
model_parameters, usage_details, cost_details, greatest(total_cost, calculated_total_cost) AS cost,
metadata_names, metadata_values, tool_calls, tool_call_names, tool_definitions"""


def ch(sql: str, timeout: int = 600) -> list:
    """Run one query and return parsed rows.

    `kubectl exec` drops the end of large streams (websocket close 1006) and can still exit 0. So the pod
    gzips the output: a cut stream then fails to decompress instead of silently losing rows."""
    p = subprocess.run(K + ["exec", "-i", "clickhouse-0", "--", "bash", "-c", CH[0] + " | gzip -1"],
                       input=sql.encode(), capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"clickhouse query failed: {p.stderr.decode(errors='replace').strip()[-500:]}")
    try:
        text = gzip.decompress(p.stdout).decode("utf-8")
    except (OSError, EOFError) as e:
        raise RuntimeError(f"truncated clickhouse stream ({len(p.stdout)} bytes): {e}") from None
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# Keep ClickHouse safe while the collector writes to it (it was OOM-killed at 6 GiB by full-table payload scans):
# a per-query memory cap, few threads, PREWHERE on trace_id and a start_time window that uses the primary key.
SETTINGS = " SETTINGS max_memory_usage = 1500000000, max_threads = 2"
WINDOW: dict = {}  # trace_id -> (root start_time, root end_time), filled by root_trace_ids / window_for


def root_trace_ids(batch: str, agent: str) -> list:
    sql = (f"SELECT trace_id, min(start_time) AS t, max(ifNull(end_time, start_time)) AS e FROM events_core FINAL "
           f"WHERE parent_span_id = '' AND has(tags, 'batch:{batch}') AND has(tags, 'agent:{agent}') "
           f"GROUP BY trace_id ORDER BY t" + SETTINGS)
    rows = ch(sql)
    for r in rows:
        WINDOW[r["trace_id"]] = (r["t"], r["e"])
    return [r["trace_id"] for r in rows]


def _in(ids: list) -> str:
    return ",".join(repr(t) for t in ids)


def window_for(ids: list) -> str:
    """SQL condition on start_time covering all spans of these traces (root start - 10 min .. root end + 10 min)."""
    miss = [t for t in ids if t not in WINDOW]
    if miss:
        for r in ch(f"SELECT trace_id, min(start_time) AS t, max(ifNull(end_time, start_time)) AS e FROM events_core "
                    f"WHERE parent_span_id = '' AND trace_id IN ({_in(miss)}) GROUP BY trace_id" + SETTINGS):
            WINDOW[r["trace_id"]] = (r["t"], r["e"])
    w = [WINDOW[t] for t in ids if t in WINDOW]
    if not w:
        return "1"
    lo, hi = min(x[0] for x in w), max(x[1] for x in w)
    return (f"start_time >= toDateTime64('{lo}', 6) - INTERVAL 10 MINUTE "
            f"AND start_time <= toDateTime64('{hi}', 6) + INTERVAL 10 MINUTE")


def iso(ts):
    if not ts:
        return None
    return ts.replace(" ", "T") + "Z"


def maybe_json(s: str):
    if s is None or s == "":
        return None
    if s[0] in "{[\"" or s in ("true", "false", "null") or s[0].isdigit() or s[0] == "-":
        try:
            return json.loads(s)
        except ValueError:
            pass
    return s


def meta(r) -> dict:
    return {k: maybe_json(v) for k, v in zip(r["metadata_names"], r["metadata_values"]) if not k.startswith(DROP_META)}


def num(x) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def _call(name, args, cid):
    if not isinstance(args, str):
        args = json.dumps(args if args is not None else {}, ensure_ascii=False)
    return {"id": cid or "", "type": "function", "function": {"name": name, "arguments": args}}


def _walk(x, acc):
    """Collect text, tool calls and reasoning from the many framework-specific generation output shapes."""
    if isinstance(x, list):
        for i in x:
            _walk(i, acc)
        return
    if not isinstance(x, dict):
        return
    t = x.get("type")
    if t == "text" and isinstance(x.get("text"), str):  # Anthropic block
        acc["text"].append(x["text"])
    elif t == "text" and isinstance(x.get("content"), str):  # pydantic-ai part
        acc["text"].append(x["content"])
    elif t in ("tool_use", "server_tool_use") and "name" in x:  # Anthropic block
        acc["calls"].append(_call(x["name"], x.get("input"), x.get("id")))
    elif t in ("tool-call", "tool_call") and "name" in x and ("arguments" in x or "args" in x):  # pydantic-ai
        acc["calls"].append(_call(x["name"], x.get("arguments", x.get("args")), x.get("id") or x.get("tool_call_id")))
    elif t in ("thinking", "redacted_thinking"):
        th = x.get("thinking") or x.get("content")
        if isinstance(th, str) and th:
            acc["reasoning"].append(th)
    elif "toolUse" in x and isinstance(x["toolUse"], dict):  # Bedrock / strands
        u = x["toolUse"]
        acc["calls"].append(_call(u.get("name"), u.get("input"), u.get("toolUseId")))
    elif "reasoningContent" in x:
        rt = (x["reasoningContent"].get("reasoningText") or {}).get("text") if isinstance(x["reasoningContent"], dict) else None
        if rt:
            acc["reasoning"].append(rt)
    elif "text" in x and isinstance(x["text"], str) and len(x) <= 3 and not x.get("thought"):  # ADK / bedrock text part
        acc["text"].append(x["text"])
    elif "function_call" in x and isinstance(x["function_call"], dict) and x["function_call"].get("name"):  # ADK part
        f = x["function_call"]
        acc["calls"].append(_call(f["name"], f.get("args"), f.get("id")))
    else:
        c = x.get("content")
        if isinstance(c, str):
            acc["text"].append(c)
        elif isinstance(c, (list, dict)):
            _walk(c, acc)
        if isinstance(x.get("parts"), list):
            _walk(x["parts"], acc)
        if isinstance(x.get("message"), str):
            m = maybe_json(x["message"])
            if isinstance(m, (list, dict)):
                _walk(m, acc)
            else:
                acc["text"].append(m)
        elif isinstance(x.get("message"), dict):
            _walk(x["message"], acc)
        if isinstance(x.get("choices"), list):
            _walk([ch.get("message") for ch in x["choices"] if isinstance(ch, dict)], acc)
        # tool_calls listed beside content blocks repeat the tool_use blocks; take them only if none were found.
        tcs = x.get("tool_calls")
        if isinstance(tcs, list) and not any(isinstance(b, dict) and b.get("type") == "tool_use" for b in
                                             (c if isinstance(c, list) else [])):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                if isinstance(tc.get("function"), dict):  # OpenAI
                    acc["calls"].append(_call(tc["function"].get("name"), tc["function"].get("arguments"), tc.get("id")))
                elif "name" in tc:  # LangChain
                    acc["calls"].append(_call(tc["name"], tc.get("args", tc.get("arguments")), tc.get("id")))


def normalize_generation_output(out):
    """Return an OpenAI-style assistant message {role, content, tool_calls[, reasoning]} or the input unchanged.
    The platform's Langfuse parser shows plain text and OpenAI tool_calls; raw Anthropic/ADK/strands shapes
    come through as unreadable JSON with no tool calls."""
    if not isinstance(out, (dict, list)):
        return out
    acc = {"text": [], "calls": [], "reasoning": []}
    _walk(out, acc)
    if not acc["text"] and not acc["calls"]:
        return out
    seen, calls = set(), []
    for c in acc["calls"]:
        k = (c["id"], c["function"]["name"], c["function"]["arguments"]) if c["id"] else id(c)
        if k not in seen:
            seen.add(k)
            calls.append(c)
    msg = {"role": "assistant", "content": "\n".join(t for t in acc["text"] if t) or None}
    if calls:
        msg["tool_calls"] = calls
    if acc["reasoning"]:
        msg["reasoning"] = "\n".join(acc["reasoning"])
    return msg


NORMALIZE = True


def observation(r) -> dict:
    u = {k: int(v) for k, v in (r["usage_details"] or {}).items()}
    o = {
        "id": r["span_id"],
        "traceId": r["trace_id"],
        "parentObservationId": r["parent_span_id"] or None,
        "type": r["type"],
        "name": r["name"],
        "startTime": iso(r["start_time"]),
        "endTime": iso(r["end_time"]),
        "completionStartTime": iso(r["completion_start_time"]),
        "input": maybe_json(r["input"]),
        "output": maybe_json(r["output"]),
        "level": r["level"] or "DEFAULT",
        "statusMessage": r["status_message"] or None,
        "metadata": meta(r),
        "environment": r["environment"] or None,
    }
    if r["type"] == "GENERATION" and NORMALIZE:
        o["output"] = normalize_generation_output(o["output"])
    if r["provided_model_name"] or u:
        o["model"] = r["provided_model_name"] or None
        o["modelParameters"] = maybe_json(r["model_parameters"]) or {}
        o["usageDetails"] = u
        o["usage"] = {"input": u.get("input", 0), "output": u.get("output", 0),
                      "total": u.get("total", u.get("input", 0) + u.get("output", 0)), "unit": "TOKENS"}
        o["costDetails"] = {k: num(v) for k, v in (r["cost_details"] or {}).items()}
        o["totalCost"] = num(r["cost"])
    if r["tool_calls"]:
        o["toolCalls"] = [maybe_json(t) for t in r["tool_calls"]]
        o["toolCallNames"] = r["tool_call_names"]
    if r["tool_definitions"]:
        o["toolDefinitions"] = {k: maybe_json(v) for k, v in r["tool_definitions"].items()}
    return o


def trace(rows: list) -> dict:
    roots = [r for r in rows if not r["parent_span_id"]]
    root = roots[0] if roots else rows[0]
    end = max((r["end_time"] or r["start_time"]) for r in rows)
    obs = [observation(r) for r in rows]
    from datetime import datetime
    f = "%Y-%m-%d %H:%M:%S.%f"
    latency = (datetime.strptime(end, f) - datetime.strptime(root["start_time"], f)).total_seconds()
    return {
        "id": root["trace_id"],
        "name": root["trace_name"] or root["name"],
        "timestamp": iso(root["start_time"]),
        "input": maybe_json(root["input"]),
        "output": maybe_json(root["output"]),
        "tags": sorted(root["tags"]),
        "sessionId": root["session_id"] or None,
        "userId": root["user_id"] or None,
        "environment": root["environment"] or None,
        "release": root["release"] or None,
        "metadata": meta(root),
        "level": root["level"] or "DEFAULT",
        "statusMessage": root["status_message"] or None,
        "latency": latency,
        "totalCost": sum(num(r["cost"]) for r in rows),
        "observations": obs,
    }


def _fetch(ids: list) -> dict:
    sql = (f"SELECT {COLS} FROM events_full FINAL PREWHERE trace_id IN ({_in(ids)}) WHERE {window_for(ids)} "
           f"ORDER BY trace_id, start_time, span_id" + SETTINGS)
    by: dict = {}
    for r in ch(sql):
        for c in ("input", "output", "status_message", "model_parameters"):
            r[c] = scrub(r[c])
        r["metadata_values"] = [scrub(v) for v in r["metadata_values"]]
        r["tool_calls"] = [scrub(v) for v in r["tool_calls"]]
        r["tool_definitions"] = {k: scrub(v) for k, v in r["tool_definitions"].items()}
        by.setdefault(r["trace_id"], []).append(r)
    return by


def trace_stats(trace_ids: list) -> dict:
    """{trace_id: {"n": rows, "b": input+output bytes}} for the given traces."""
    if not trace_ids:
        return {}
    return {r["trace_id"]: r for r in ch(
        f"SELECT trace_id, count() AS n, sum(input_length + output_length) AS b FROM events_full FINAL "
        f"PREWHERE trace_id IN ({_in(trace_ids)}) WHERE {window_for(trace_ids)} GROUP BY trace_id" + SETTINGS)}


def iter_traces(trace_ids: list, max_bytes: int = 12_000_000, max_traces: int = 20, attempts: int = 4):
    """Yield trace dicts. Traces are fetched in chunks of at most `max_bytes` of input/output, and every
    chunk is checked against a per-trace row count; a failed chunk is retried, then split to single traces."""
    if not trace_ids:
        return
    stats = trace_stats(trace_ids)
    chunks, cur, size = [], [], 0
    for t in trace_ids:
        if t not in stats:
            continue
        b = int(stats[t]["b"])
        if cur and (size + b > max_bytes or len(cur) >= max_traces):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += b
    if cur:
        chunks.append(cur)
    while chunks:
        ids = chunks.pop(0)
        by, err = None, None
        for attempt in range(attempts):
            try:
                got = _fetch(ids)
                short = [t for t in ids if len(got.get(t, [])) != int(stats[t]["n"])]
                if not short:
                    by = got
                    break
                err = f"row count mismatch for {short}"
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                err = e
            if len(ids) > 1 and attempt >= 1:
                break
            time.sleep(2 * (attempt + 1))
        if by is None:
            if len(ids) > 1:
                chunks[:0] = [[t] for t in ids]
                continue
            raise RuntimeError(f"could not fetch trace {ids[0]}: {err}")
        for t in ids:
            yield trace(by[t])


def export(batch: str, agent: str, out: str, envelope: str = "array", trace_ids=None, exclude=()) -> dict:
    for v in (batch, agent):
        if not SAFE.match(v):
            raise ValueError(f"unsafe name: {v!r}")
    ids = trace_ids if trace_ids is not None else root_trace_ids(batch, agent)
    ids = [t for t in ids if t.isalnum() and t not in set(exclude)]
    if envelope == "object" and len(ids) != 1:
        raise ValueError(f"envelope=object needs exactly one trace, got {len(ids)}")
    opener = gzip.open if out.endswith(".gz") else open
    n_obs, done = 0, []
    with opener(out, "wt", encoding="utf-8") as f:
        if envelope == "data":
            f.write('{"data": [')
        elif envelope == "array":
            f.write("[")
        for t in iter_traces(ids):
            if envelope == "jsonl":
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
            else:
                if done and envelope != "object":
                    f.write(",\n")
                json.dump(t, f, ensure_ascii=False)
            n_obs += len(t["observations"])
            done.append(t["id"])
        if envelope == "data":
            f.write("]}")
        elif envelope == "array":
            f.write("]")
    return {"agent": agent, "batch": batch, "traces": len(done), "observations": n_obs, "trace_ids": done, "out": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--out")
    ap.add_argument("--envelope", choices=["array", "data", "object", "jsonl"], default="array")
    ap.add_argument("--trace-ids", help="comma-separated subset (still must be given explicitly)")
    ap.add_argument("--exclude-ids-file", help="file with one trace_id per line to skip")
    ap.add_argument("--raw-generations", action="store_true", help="keep generation outputs in framework shape")
    a = ap.parse_args()
    global NORMALIZE
    NORMALIZE = not a.raw_generations
    out = a.out or f"{a.agent}.{a.batch}.json.gz"
    excl = [l.strip() for l in open(a.exclude_ids_file)] if a.exclude_ids_file else []
    tids = a.trace_ids.split(",") if a.trace_ids else None
    s = export(a.batch, a.agent, out, a.envelope, tids, excl)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
