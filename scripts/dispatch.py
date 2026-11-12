"""Launch an Indexed Job that runs agents in-cluster. One pod per agent, bounded parallelism.

    python3 scripts/dispatch.py --batch b1 [--parallelism 20] [--concurrency 4] [--limit N] [--only id,id] [--family f] [--dry-run]

Venv per agent: the content of agents/<family>/<id>/VENV if present, else <family>.
Run scripts/sync.sh and scripts/build-venvs.sh first.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = ["kubectl", "--context", "gepton-base", "-n", "experiments"]
IMAGE = "ghcr.io/astral-sh/uv:0.9.9-python3.12-bookworm-slim"


def agents(family: str | None, only: set[str] | None) -> list[tuple[str, str, str]]:
    out = []
    for a in sorted(ROOT.glob("agents/*/*/agent.py")):
        d = a.parent
        fam, aid = d.parent.name, d.name
        if family and fam != family or only and aid not in only:
            continue
        venv = (d / "VENV").read_text().strip() if (d / "VENV").exists() else fam
        out.append((venv, fam, aid))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--parallelism", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--family", default=None)
    ap.add_argument("--run", default="", help="suffix for the Job name, so one batch can be dispatched more than once")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = agents(a.family, set(a.only.split(",")) if a.only else None)
    if not rows:
        print("no agents matched", file=sys.stderr)
        return 1
    manifest = "\n".join("\t".join(r) for r in rows) + "\n"
    name = f"run-{a.batch}{'-' + a.run if a.run else ''}".lower().replace("_", "-")[:52]
    limit = f"--limit {a.limit}" if a.limit else ""
    script = f"""set -euo pipefail
line=$(sed -n "$((JOB_COMPLETION_INDEX+1))p" /work/manifests/{name}.tsv)
venv=$(echo "$line" | cut -f1); fam=$(echo "$line" | cut -f2); aid=$(echo "$line" | cut -f3)
echo "index=$JOB_COMPLETION_INDEX agent=$aid family=$fam venv=$venv"
cd /work/repo
exec /work/venvs/$venv/bin/python -u -m peira_harness.runner --agent agents/$fam/$aid --batch {a.batch} \\
  --concurrency {a.concurrency} --results-dir /work/results {limit}
"""
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": "experiments", "labels": {"app": "peira-run", "batch": a.batch}},
        "spec": {
            "completionMode": "Indexed",
            "completions": len(rows),
            "parallelism": a.parallelism,
            "backoffLimitPerIndex": 2,
            "maxFailedIndexes": len(rows),
            "ttlSecondsAfterFinished": 86400 * 3,
            "template": {
                "metadata": {
                    "labels": {"app": "peira-run", "batch": a.batch},
                    # Agent pods run for up to an hour; consolidation would kill in-flight tasks.
                    "annotations": {"karpenter.sh/do-not-disrupt": "true"},
                },
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {"peira.tech/pool": "experiments"},
                    "tolerations": [{"key": "peira.tech/experiments", "operator": "Equal", "value": "true", "effect": "NoSchedule"}],
                    "containers": [{
                        "name": "agent",
                        "image": IMAGE,
                        "command": ["bash", "-c", script],
                        "envFrom": [{"secretRef": {"name": "langfuse-client"}}, {"secretRef": {"name": "litellm-client"}}],
                        "env": [
                            {"name": "LLM_PROXY_URL", "value": "http://litellm.experiments.svc.cluster.local:4000"},
                            {"name": "TOOLS_URL", "value": "http://tools.experiments.svc.cluster.local:8080"},
                            {"name": "PEIRA_ENVIRONMENT", "value": "experiments"},
                            {"name": "CREWAI_DISABLE_TELEMETRY", "value": "true"},
                            {"name": "AGNO_TELEMETRY", "value": "false"},
                            {"name": "ANONYMIZED_TELEMETRY", "value": "false"},
                            {"name": "HF_HOME", "value": "/work/cache/hf"},
                            {"name": "DENO_DIR", "value": "/tmp/deno"},
                            {"name": "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "value": "4096"},
                            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                            # Harness comes from the synced repo, not the copy installed in the venv.
                            {"name": "PYTHONPATH", "value": "/work/repo/harness"},
                        ],
                        "resources": {"requests": {"cpu": "250m", "memory": "768Mi"}, "limits": {"memory": "3Gi"}},
                        "volumeMounts": [{"name": "work", "mountPath": "/work"}, {"name": "tmp", "mountPath": "/tmp"}],
                    }],
                    "volumes": [
                        {"name": "work", "persistentVolumeClaim": {"claimName": "work"}},
                        {"name": "tmp", "emptyDir": {"sizeLimit": "5Gi"}},
                    ],
                },
            },
        },
    }
    print(f"{len(rows)} agents -> job {name} (parallelism {a.parallelism}, concurrency {a.concurrency})")
    if a.dry_run:
        print(manifest)
        return 0
    pod = subprocess.check_output(K + ["get", "pod", "-l", "app=toolbox", "-o", "jsonpath={.items[0].metadata.name}"], text=True)
    subprocess.run(K + ["exec", "-i", pod, "--", "bash", "-c", f"mkdir -p /work/manifests && cat > /work/manifests/{name}.tsv"],
                   input=manifest, text=True, check=True)
    subprocess.run(K + ["apply", "-f", "-"], input=json.dumps(job), text=True, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
