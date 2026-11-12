# trace-collector

This repo runs 100 open-source-derived agents on 15 frameworks and collects their traces in a self-hosted Langfuse.
It was built for peira.tech, to get realistic agent traces.

## Contents

| Path | Content |
|---|---|
| `CONTRACT.md` | The rules every agent and service follows. |
| `harness/` | `peira_harness`: the runner, the tracing setup, the LLM proxy client and the tool client. |
| `agents/<family>/<agent_id>/` | One agent: `agent.py`, `tasks.jsonl` (100 tasks), `README.md` (source and smoke cost). |
| `infra/` | Kubernetes manifests and install scripts for Langfuse, LiteLLM, the tools service and the runtime. |
| `scripts/` | Sync, venv build, dispatch, trace check, report, audit, export, archive and upload. |
| `data/traces/` | The collected traces as gzipped JSONL, in Git LFS. See `data/traces/README.md`. |
| `results/` | Runner result rows from the smoke runs, and the platform upload state. |

## Run 2026-11-12 (batch `full`)

- 100 agents, 10,000 tasks, 9,902 tasks with an ok trace, 10,130 traces in total (failed tasks were retried).
- Models: Claude Haiku 5.5 for most agents, Claude Sonnet 5.5 for the complex parts of 18 agents.
- Spend: $261 through the LiteLLM proxy.
- Cluster: `gepton-base`, namespace `experiments`, on a dedicated Karpenter node pool.

## Run again

```
scripts/sync.sh
scripts/build-venvs.sh
python3 scripts/dispatch.py --batch <name> --parallelism 20 --concurrency 4
python3 scripts/report.py --batch <name>
python3 scripts/archive_traces.py --batch <name>
```
