"""Print Langfuse traces as trees and check them against the contract.

    python scripts/inspect_trace.py <trace_id> [<trace_id> ...]
    python scripts/inspect_trace.py --results results/smoke/<agent_id>.jsonl

Langfuse v4 (events_only mode) has no public trace-read endpoint and its v2 observations API does
not return tags, so this reads Langfuse's ClickHouse table `events_core` through `kubectl exec`.
Needs only kubectl access to the gepton-base context. Waits up to 90 s for ingestion.
Exit code 1 if any trace fails the checks (missing tags, no generation, no usage, no nesting).
"""
import json
import subprocess
import sys
import time

K = ["kubectl", "--context", "gepton-base", "-n", "experiments"]
REQUIRED_TAG_PREFIXES = ("framework:", "agent:", "family:", "complexity:", "kind:", "batch:")
COLS = ("trace_id, span_id, parent_span_id, start_time, name, type, level, status_message, tags, session_id, "
        "provided_model_name, usage_details, greatest(total_cost, calculated_total_cost) AS total_cost, scope_name")


def query(sql: str) -> list:
    out = subprocess.run(
        K + ["exec", "-i", "clickhouse-0", "--", "bash", "-c",
             'clickhouse-client --password "$CLICKHOUSE_PASSWORD" --format JSONEachRow --output_format_json_quote_64bit_integers=0 -q "$(cat)"'],
        input=sql, text=True, capture_output=True, check=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def fetch(trace_ids: list[str], wait_s: int = 180) -> dict:
    ids = ",".join(f"'{t}'" for t in trace_ids if t.isalnum())
    deadline = time.time() + wait_s
    while True:
        rows = query(f"SELECT {COLS} FROM events_core FINAL WHERE trace_id IN ({ids}) ORDER BY start_time")
        by: dict = {}
        for r in rows:
            by.setdefault(r["trace_id"], []).append(r)
        # The root span ends last, so it is exported last: wait until every trace has its root.
        complete = all(any(not o["parent_span_id"] for o in by.get(t, [])) for t in set(trace_ids))
        if complete or time.time() > deadline:
            return by
        time.sleep(5)


def check(obs: list) -> list[str]:
    problems = []
    root = [o for o in obs if not o["parent_span_id"]]
    tags = set().union(*(o["tags"] for o in obs)) if obs else set()
    for p in REQUIRED_TAG_PREFIXES:
        if not any(t.startswith(p) for t in tags):
            problems.append(f"missing tag {p}*")
    if len(root) != 1:
        problems.append(f"{len(root)} root observations (want 1)")
    gens = [o for o in obs if o["type"] == "GENERATION"]
    if not gens:
        problems.append("no GENERATION observations")
    elif not any(o["usage_details"] for o in gens):
        problems.append("generations have no token usage")
    if any(o["scope_name"].startswith("com.anthropic.sdk") for o in obs):
        problems.append("duplicate Anthropic SDK spans (com.anthropic.sdk)")
    unpriced = [o for o in gens if not o["provided_model_name"]]
    if unpriced:
        problems.append(f"{len(unpriced)} generations without a model name (unpriced)")
    if gens and not all(o["parent_span_id"] for o in gens):
        problems.append("some generations are not nested")
    if len(obs) < 3:
        problems.append(f"only {len(obs)} observations")
    return problems


def show(tid: str, obs: list) -> None:
    kids: dict = {}
    for o in obs:
        kids.setdefault(o["parent_span_id"] or None, []).append(o)
    root = (kids.get(None) or [{}])[0]
    print(f"trace {tid}  tags={sorted(set().union(*(o['tags'] for o in obs)))}  session={root.get('session_id')}")

    def walk(pid, depth):
        for o in kids.get(pid, []):
            extra = f" model={o['provided_model_name']} usage={o['usage_details']}" if o["type"] == "GENERATION" else ""
            lvl = f" [{o['level']}: {o['status_message'][:80]}]" if o["level"] not in ("DEFAULT", "") else ""
            print(f"  {'  ' * depth}- {o['type']}: {o['name'][:60]} ({o['scope_name']}){extra}{lvl}")
            walk(o["span_id"], depth + 1)

    walk(None, 0)
    counts: dict = {}
    for o in obs:
        counts[o["type"]] = counts.get(o["type"], 0) + 1
    cost = sum(float(o["total_cost"] or 0) for o in obs)
    print(f"  totals: {counts}  cost=${cost:.6f}")


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--results":
        ids = [json.loads(line)["trace_id"] for line in open(args[1]) if line.strip()]
    else:
        ids = args
    by = fetch(ids)
    bad = 0
    for tid in ids:
        obs = by.get(tid, [])
        if not obs:
            print(f"trace {tid}: NOT FOUND in events_core\n  CHECK: FAIL not ingested\n")
            bad += 1
            continue
        show(tid, obs)
        p = check(obs)
        print(f"  CHECK: {'PASS' if not p else 'FAIL ' + '; '.join(p)}\n")
        bad += bool(p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
