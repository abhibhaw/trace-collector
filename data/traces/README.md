# Agent traces

This folder holds the agent traces from the peira.tech trace collection run of 2026-11-12.
The files are in Git LFS.
Run `git lfs pull` to get the content.

## Layout

```
data/traces/<batch>/<family>/<agent_id>.jsonl.gz
data/traces/<batch>/manifest.json
```

- `batch` is `full`: 100 agents, 100 tasks each.
- `family` is the folder under `agents/` that holds the agent code.
- `manifest.json` gives, for each agent: trace count, observation count, file size and SHA-256.

## Record format

Each line is one trace, as one JSON object.
The shape follows the Langfuse public API trace, with the observations nested in the trace.

| Field | Meaning |
|---|---|
| `id` | Trace id (32 hex characters). The same id is in Langfuse. |
| `name` | The agent id. |
| `timestamp` | Start time of the root span, ISO 8601, UTC. |
| `input` | The task text. |
| `output` | The final answer, or `{"error": ...}` for a failed run. |
| `level`, `statusMessage` | `ERROR` and the exception text when the run failed. |
| `tags` | `framework:`, `agent:`, `family:`, `complexity:`, `kind:`, `batch:`. |
| `sessionId` | `<agent_id>:<batch>`. |
| `metadata` | `task_id`, `source_url`, `model`, `agent_id`. |
| `latency` | Seconds. |
| `observations[]` | All spans of the trace. |

Each observation has these fields:

| Field | Meaning |
|---|---|
| `id`, `traceId`, `parentObservationId` | The span tree. The root has `parentObservationId: null`. |
| `type` | `AGENT`, `CHAIN`, `TOOL`, `GENERATION`, `SPAN`, `GUARDRAIL`, `EVALUATOR`, `RETRIEVER`. |
| `name` | Span name from the framework or the instrumentor. |
| `startTime`, `endTime`, `completionStartTime` | ISO 8601, UTC. |
| `input`, `output` | Full payloads. JSON when the source was valid JSON. |
| `level`, `statusMessage` | Span errors. |
| `model`, `usageDetails`, `usage`, `costDetails`, `totalCost` | Generations only. Cost in USD from Langfuse prices. |
| `toolCalls`, `toolCallNames`, `toolDefinitions` | Present when the framework recorded them. |
| `metadata` | Framework attributes. Langfuse resource and scope keys are removed. |

Generation `output` is in the framework's own shape (Anthropic content blocks, LangChain messages, and so on).
`scripts/export_langfuse.py` without `--raw-generations` converts it to OpenAI-style messages.

## Known limits

- The usage keys differ by framework. Some frameworks report Anthropic cache reads in separate keys, so `input` can be small.
- 62 of 78,989 generations have no cost. 23 of them are LlamaIndex wrapper spans with no model call.
- Most `web_search` results come from a Wikipedia fallback (`source: wikipedia-fallback`), because DuckDuckGo was unreachable.
- Tool data is a fictional company, "Lumen & Co". Some KB articles and tickets contain prompt injection on purpose, for the adversarial tasks.
- Secrets are masked in payloads (`sk-`, `pk-lf-`, `sk-lf-`, `Bearer`, `api_key`, `password` values).

## Read the data

```python
import gzip, json
with gzip.open("data/traces/full/langgraph/lg-sql-analyst.jsonl.gz", "rt") as f:
    for line in f:
        trace = json.loads(line)
        gens = [o for o in trace["observations"] if o["type"] == "GENERATION"]
        print(trace["metadata"]["task_id"], trace["level"], len(gens))
```

## Rebuild

`python3 scripts/archive_traces.py --batch full --force` exports again from the Langfuse ClickHouse in namespace `experiments`.
