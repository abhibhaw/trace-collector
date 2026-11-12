"""Per-agent trace report for a batch, from Langfuse's ClickHouse.

    python scripts/report.py --batch b1 [--json out.json]

Columns: traces, error traces (root level ERROR), mean observations, mean generations, total cost.
"""
import argparse
import json
import subprocess
import sys

K = ["kubectl", "--context", "gepton-base", "-n", "experiments"]


def query(sql: str) -> list:
    out = subprocess.run(
        K + ["exec", "-i", "clickhouse-0", "--", "bash", "-c",
             'clickhouse-client --password "$CLICKHOUSE_PASSWORD" --format JSONEachRow --output_format_json_quote_64bit_integers=0 -q "$(cat)"'],
        input=sql, text=True, capture_output=True, check=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    batch_tag = f"batch:{a.batch}"
    if not batch_tag.replace(":", "").replace("-", "").replace("_", "").isalnum():
        print("bad batch name", file=sys.stderr)
        return 2
    sql = f"""
WITH t AS (
  SELECT trace_id,
         anyIf(arrayFirst(x -> startsWith(x, 'agent:'), tags), parent_span_id = '') AS agent,
         anyIf(arrayFirst(x -> startsWith(x, 'framework:'), tags), parent_span_id = '') AS framework,
         maxIf(level = 'ERROR', parent_span_id = '') AS is_error,
         count() AS obs,
         countIf(type = 'GENERATION') AS gens,
         countIf(type = 'TOOL' OR type = 'SPAN' AND positionCaseInsensitive(name, 'tool') > 0) AS tools,
         sum(greatest(total_cost, calculated_total_cost)) AS cost
  FROM events_core FINAL
  WHERE trace_id IN (SELECT DISTINCT trace_id FROM events_core WHERE has(tags, '{batch_tag}'))
  GROUP BY trace_id
)
SELECT replaceOne(agent, 'agent:', '') AS agent, replaceOne(framework, 'framework:', '') AS framework,
       count() AS traces, sum(is_error) AS errors, round(avg(obs), 1) AS mean_obs,
       round(avg(gens), 1) AS mean_gens, round(avg(tools), 1) AS mean_tools, round(sum(cost), 4) AS cost_usd
FROM t GROUP BY agent, framework ORDER BY framework, agent
"""
    rows = query(sql)
    print(f"{'agent':40} {'framework':16} {'traces':>6} {'errors':>6} {'obs':>6} {'gens':>5} {'tools':>5} {'cost$':>8}")
    for r in rows:
        print(f"{r['agent']:40} {r['framework']:16} {r['traces']:>6} {r['errors']:>6} {r['mean_obs']:>6} "
              f"{r['mean_gens']:>5} {r['mean_tools']:>5} {r['cost_usd']:>8}")
    tot = sum(r["traces"] for r in rows)
    err = sum(r["errors"] for r in rows)
    cost = sum(r["cost_usd"] for r in rows)
    print(f"\n{len(rows)} agents, {tot} traces, {err} error traces, ${cost:.2f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
