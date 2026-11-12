# Builder brief (for framework-family owners)

Read `CONTRACT.md` first. It overrides anything here.

## Your job

Build the 10 agents of one family directory under `agents/<family>/`.
Derive each agent from a real open-source example (official example repos, templates, cookbooks, well-known community repos).
Shallow-clone sources into your own new, empty directory under the scratchpad path given in your prompt.
Read the source, then adapt it.
Never run code from a cloned repo directly.
Record the exact source URL (repo path or file) in `AGENT.source_url` and the README.

Adaptation rules:

- Keep the source's architecture: its graph shape, sub-agents, handoffs, prompts, reflection loops, and tool roles.
  These shapes are what makes the traces valuable.
- Replace paid or keyed tools (Tavily, Serper, E2B, Exa, Firecrawl, OpenWeather, Polygon, browsers) with tools from `peira_harness.tools`.
- Replace the model with the proxy (see `peira_harness.llm`).
- Bound cost: at most about 25 LLM calls per run (recursion limits, max_steps, max_turns, max_iter).
- At most 2 agents in your family may use `claude-sonnet-5-5`, and only ones marked `complexity="complex"`.
- Aim for a spread: about 3 simple, 4 medium, 3 complex.
- Use as many distinct framework features as the family offers: multi-agent, handoffs, sub-agents, structured output, memory/state, guardrails, parallel branches, loops, human-in-the-loop simulated by an auto-approver, code execution, RAG.

## Tasks

- Write `tasks.jsonl` per agent, exactly 100 lines, following the contract's task format and kind mix.
- Read `infra/tools/README.md` and `infra/tools/catalog.json` (written by the tools owner) for real entity ids: customers, orders, tickers, cities, KB articles, injection-carrying KB articles, databases.
  If they do not exist yet, write tasks that do not depend on specific ids first, then come back.
- You may generate tasks with a script, but review them: they must read like real user requests and vary in length, tone, and difficulty.
  Some should be multi-part, some terse, some with typos, some long with context.

## Environment

- Python 3.12. Create `agents/<family>/.venv` with `uv venv -p 3.12 agents/<family>/.venv`.
- `agents/<family>/requirements.txt`: pinned versions (`==`) of every top-level package you need, including OpenInference instrumentors.
  It must resolve together with `langfuse==5.2.0` and install on linux/arm64.
  Install with `uv pip install -p agents/<family>/.venv -r agents/<family>/requirements.txt -e harness`.
- Before smoke runs: `source scripts/dev-env.sh` (port-forwards and env vars for the proxy, Langfuse, and tools).
  If the script or the secrets do not exist yet, the infra owners are still deploying.
  Build code and tasks first, then poll every 2 minutes (up to 40 minutes) for `kubectl --context gepton-base -n experiments get secret litellm-client langfuse-client` and `curl -s localhost:8080/health`.
- Never kill port-forwards you did not start. Other builders share them.
- Smoke run: `agents/<family>/.venv/bin/python -m peira_harness.runner --agent agents/<family>/<id> --batch smoke --limit 3 --concurrency 3`
- Verify: `agents/<family>/.venv/bin/python scripts/inspect_trace.py --results results/smoke/<id>.jsonl` must print `CHECK: PASS` for each trace.
  Read the printed tree too: it must show the agent's real structure (sub-agents, tool spans), not a flat list.
- Order each `tasks.jsonl` so that the first 3 tasks contain at least one `edge` or `adversarial` task. The smoke run then covers it.

## Boundaries

- Only edit files under `agents/<family>/` and `results/smoke*/`.
- Do not edit `harness/`, `scripts/`, `infra/`, or other families.
  If the harness blocks you, work around it inside your family directory if possible, and describe the harness problem in your final report.
- Do not deploy anything to the cluster. The orchestrator runs the full batch in-cluster.
- Budget: your family's smoke testing must stay under $10. Check spend with the proxy's `/key/info` if needed.

## Final report (keep it short)

A table: agent_id, framework feature shape (one line), source URL, model, complexity, smoke result (3/3 PASS?), mean LLM calls per run, mean cost per run.
Then: harness or infra problems found, dependency pins of note, anything the orchestrator must know to run the family in-cluster (system packages, env vars, startup cost such as index building).
