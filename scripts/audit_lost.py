"""Find runner result rows whose trace has no root observation in Langfuse, and optionally drop them.

    python3 scripts/audit_lost.py --batch full            # report only
    python3 scripts/audit_lost.py --batch full --apply    # back up results, drop lost rows, print --only list

A row is "lost" when its trace_id has no root span (parent_span_id = '') in events_core, for
example because ClickHouse restarted during ingestion. Dropping the row makes the runner's skip
logic re-run that task on the next dispatch of the same batch. Run it only when no job of the batch
is active, so no in-flight trace is mistaken for a lost one.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter

K = ["kubectl", "--context", "gepton-base", "-n", "experiments"]


def sh(args: list[str], stdin: str | None = None) -> str:
    return subprocess.run(args, input=stdin, text=True, capture_output=True, check=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not a.batch.replace("-", "").replace("_", "").isalnum():
        print("bad batch name", file=sys.stderr)
        return 2

    active = sh(K + ["get", "jobs", "-l", f"batch={a.batch}", "-o", "jsonpath={.items[*].status.active}"]).split()
    if any(x not in ("", "0") for x in active):
        print("a job of this batch is still active; run the audit after it finishes", file=sys.stderr)
        return 3

    pod = sh(K + ["get", "pod", "-l", "app=toolbox", "-o", "jsonpath={.items[0].metadata.name}"])
    raw = sh(K + ["exec", pod, "--", "bash", "-c", f"cd /work/results/{a.batch} && grep -H '' *.jsonl"])
    rows = []  # (agent_id, row)
    for line in raw.splitlines():
        fname, _, body = line.partition(":")
        if body.strip():
            rows.append((fname.removesuffix(".jsonl"), json.loads(body)))

    roots = set(sh(
        K + ["exec", "-i", "clickhouse-0", "--", "bash", "-c",
             'clickhouse-client --password "$CLICKHOUSE_PASSWORD" --max_memory_usage=1500000000 -q "$(cat)"'],
        stdin=f"SELECT DISTINCT trace_id FROM events_core WHERE parent_span_id = '' "
              f"AND has(tags, 'batch:{a.batch}') FORMAT TSV",
    ).split())

    lost = [(ag, r) for ag, r in rows if r["trace_id"] not in roots]
    by_agent = Counter(ag for ag, _ in lost)
    print(f"{len(rows)} result rows, {len(roots)} root traces, {len(lost)} lost rows in {len(by_agent)} agents")
    for ag, n in by_agent.most_common():
        print(f"  {ag}: {n}")
    if not lost or not a.apply:
        return 0

    lost_ids = {r["trace_id"] for _, r in lost}
    script = f"""set -euo pipefail
cd /work/results/{a.batch}
bak=../{a.batch}.bak-$(date +%s)
mkdir -p "$bak" && cp *.jsonl "$bak"/
python3 - <<'PY'
import json, glob
lost = set({json.dumps(sorted(lost_ids))})
for f in glob.glob('*.jsonl'):
    keep = [l for l in open(f) if l.strip() and json.loads(l)['trace_id'] not in lost]
    open(f, 'w').writelines(keep)
PY
"""
    sh(K + ["exec", "-i", pod, "--", "bash", "-s"], stdin=script)
    print(f"dropped {len(lost_ids)} rows; re-run with:")
    print(f"  python3 scripts/dispatch.py --batch {a.batch} --run p3 --only {','.join(sorted(by_agent))} "
          f"--parallelism {min(20, len(by_agent))} --concurrency 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
