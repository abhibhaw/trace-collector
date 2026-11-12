# Trace collector contract

This file is the single source of truth for every contributor.
Do not deviate from it.
If something in it is wrong, report it to the orchestrator instead of working around it.

## Goal

Run 100 open-source-derived agents across many frameworks in the `experiments` namespace of the `gepton-base` AKS cluster.
Each agent runs 100 tasks.
Every run produces one nested trace in a self-hosted Langfuse.
The traces are for peira.tech, an agent-observability product, so realistic and complex agent behaviour is the point.

## Cluster facts

- kube context: `gepton-base`. Always pass `--context gepton-base`. Never touch other namespaces.
- Namespace: `experiments` (has a ResourceQuota and LimitRange).
- Nodes are **arm64**. Every image must be multi-arch.
- Workloads must run on the dedicated Karpenter pool. Every pod spec needs:

```yaml
nodeSelector: {peira.tech/pool: experiments}
tolerations: [{key: peira.tech/experiments, operator: Equal, value: "true", effect: NoSchedule}]
```

- We do not build container images. There is no registry we can push to.
- Secrets are created with `kubectl create secret ... --from-literal` and random values from `openssl rand`. Never write a secret value to a file, a log, or a summary.
- Do not read secrets that you did not create.

## Services (in-cluster DNS)

| Service | URL | Owner |
|---|---|---|
| Langfuse web | `http://langfuse-web.experiments.svc.cluster.local:3000` | infra-langfuse |
| LiteLLM proxy | `http://litellm.experiments.svc.cluster.local:4000` | infra-litellm |
| Tools service | `http://tools.experiments.svc.cluster.local:8080` | infra-tools |

Secrets that agent pods load with `envFrom`:

| Secret | Keys |
|---|---|
| `langfuse-client` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` |
| `litellm-client` | `LLM_PROXY_KEY` (a LiteLLM virtual key) |

Local development: port-forward and set the same env vars.
`source scripts/dev-env.sh` starts the port-forwards and exports the variables silently. It never prints values.

## LLM access

- All LLM calls go through the LiteLLM proxy. Never call a provider directly.
- The proxy exposes two models under their real Anthropic IDs:
  - `claude-haiku-5-5` - the default for every agent.
  - `claude-sonnet-5-5` - only for agents marked `complexity: "complex"`, at most 15 of the 100.
- The proxy speaks both formats:
  - OpenAI format at `$LLM_PROXY_URL/v1` (chat completions).
  - Anthropic format at `$LLM_PROXY_URL` (`/v1/messages`). Set `base_url=$LLM_PROXY_URL` and `api_key=$LLM_PROXY_KEY` on the Anthropic SDK or `langchain_anthropic.ChatAnthropic`.
- Prefer the Anthropic-native client where the framework supports it. Use the OpenAI format or a LiteLLM model wrapper only when the framework has no Anthropic path.
- Model rules for these models (violations return HTTP 400):
  - Do not set `temperature`, `top_p`, or `top_k`.
  - Do not use assistant-message prefill.
  - On `claude-sonnet-5-5`, do not force tool choice (`tool_choice` `any` or a named tool). LangChain `with_structured_output` forces tool choice by default; use `method="json_schema"` or prompt for JSON.
  - Do not set `thinking` to `disabled` or set `budget_tokens`. Leave thinking at its default.
- Set `max_tokens` to 4096 unless the agent needs more.
- `harness/peira_harness/llm.py` gives the URLs, key, and ready-made clients. Use it.

## Tools

- Only free, keyless tools. Do not add any paid API, even with a free tier.
- All shared tools are plain typed Python functions with docstrings in `harness/peira_harness/tools.py`. They call the tools service over HTTP. Wrap them in each framework's tool type.
- Available tool families: web search, URL fetch, Wikipedia, arXiv, Python execution (sandboxed in the tools service), SQL over sample SQLite databases (Chinook, Northwind), and mock business APIs (CRM, orders, support tickets, inventory, travel, finance quotes, HR, calendar, email outbox).
- An agent may also define its own local tools (calculators, in-memory scratch files, parsers).

## Repository layout

```
harness/peira_harness/      shared package: runner, tracing, llm, tools
agents/<family>/requirements.txt     one venv per family
agents/<family>/<agent_id>/agent.py  the agent
agents/<family>/<agent_id>/tasks.jsonl   exactly 100 tasks
agents/<family>/<agent_id>/README.md     source URL, what was adapted, tools used
infra/                      Kubernetes manifests and Helm values
scripts/                    sync, venv build, dispatch, verify
```

`agent_id` format: `<family-prefix>-<slug>`, lowercase, hyphens, max 40 chars, unique across the repo.

## Agent module interface

```python
from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="lg-sql-analyst",
    framework="langgraph",            # see the family table
    source_url="https://github.com/...",  # the open-source example it is derived from
    description="One sentence.",
    complexity="simple" | "medium" | "complex",
    model="claude-haiku-5-5",
    tracing="langchain" | "openinference" | "native-otel" | "manual",
)

def setup() -> None: ...              # optional, runs once per process (instrumentors, indexes)
def run(task: dict) -> str: ...       # sync agents
async def arun(task: dict) -> str: ...  # async agents (define one of run or arun)
```

- `run` / `arun` receives the task dict and returns the final answer as a string.
- Raise on failure. The runner records the error on the trace. Do not swallow exceptions into a fake answer.
- Do not create root traces, call `flush`, or set trace tags. The runner does that.
- Get callbacks or instrumentors from `peira_harness.tracing`:
  - LangChain, LangGraph, deepagents: pass `config={"callbacks": tracing.langchain_callbacks()}`.
  - Other frameworks: call the matching OpenInference instrumentor in `setup()` with `tracing.instrument(<Instrumentor>())`.
  - Use exactly one source of LLM spans per agent. Do not instrument both the framework and the underlying SDK.
- The trace must show nested spans: agent or chain, tool calls, and LLM generations with token usage.

## Task file

`tasks.jsonl`, exactly 100 lines, each:

```json
{"id": "t001", "input": "the user request", "kind": "normal", "expected": null}
```

- `kind` is `normal` (about 85), `edge` (about 10: ambiguous, missing data, contradictory), or `adversarial` (about 5: prompt injection in a tool result or the request, impossible requests).
- Tasks must be realistic and varied. Do not produce 100 copies of one template with a swapped noun.
- Tasks must be answerable with the agent's tools and the tools service data.

## Runner

```
python -m peira_harness.runner --agent agents/<family>/<agent_id> --batch <batch> [--limit N] [--concurrency C]
```

- One root observation per task, `as_type="agent"`, named `agent_id`.
- Trace tags: `framework:<f>`, `agent:<id>`, `family:<family>`, `complexity:<c>`, `kind:<task kind>`, `batch:<b>`.
- Trace metadata: `task_id`, `source_url`, `model`. Session: `<agent_id>:<batch>`.
- Results go to `results/<batch>/<agent_id>.jsonl`. Re-running skips tasks that already succeeded.
- The runner flushes and shuts down Langfuse before exit.

## Family table

| Family dir | Framework tag | Prefix | Agents |
|---|---|---|---|
| `langgraph` | langgraph | `lg` | 10 |
| `deepagents` | deepagents | `da` | 10 |
| `google_adk` | google-adk | `adk` | 10 |
| `crewai` | crewai | `crew` | 10 |
| `smolagents` | smolagents | `smol` | 10 |
| `openai_agents` | openai-agents | `oai` | 10 |
| `pydantic_llamaindex` | pydantic-ai / llamaindex | `pyd` / `li` | 5 + 5 |
| `autogen_agno` | autogen / agno | `ag` / `agno` | 5 + 5 |
| `dspy_strands_haystack` | dspy / strands / haystack | `dspy` / `str` / `hay` | 4 + 3 + 3 |
| `raw_loops` | raw-anthropic / raw-litellm | `raw` | 10 |

## Definition of done for one agent

1. `agent.py`, `tasks.jsonl` (100 lines), and `README.md` exist.
2. A local smoke run with `--limit 3` succeeds against the in-cluster proxy and Langfuse (through port-forward).
3. In Langfuse, each of those 3 traces has the right tags, nested tool spans, and LLM generations with token usage.
4. The agent's per-run cost on the smoke tasks is recorded in its README.
