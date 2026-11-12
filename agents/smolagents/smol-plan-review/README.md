# smol-plan-review

Source: https://github.com/huggingface/smolagents/blob/main/examples/plan_customization/plan_customization.py

## Shape

`CodeAgent` with `planning_interval=3` and `step_callbacks={PlanningStep: AutoReviewer, ActionStep: log_action}`. The reviewer approves the plan, edits `memory_step.plan`, or calls `agent.interrupt()`. After an interrupt, the run resumes with `reset=False`.

## What was adapted

- The `input()` approve/modify/cancel menu becomes a deterministic auto-reviewer that simulates the human.
- Approve: a short plan that mentions sources.
- Modify: a plan with more than 5 steps, or no sources. The reviewer trims the plan and adds constraints.
- Cancel: a plan that mentions unsafe topics. The run resumes with `reset=False` and the reviewer's feedback, as in the source's resume demo.
- `DuckDuckGoSearchTool` becomes `web_search`, `fetch_url` and `wikipedia_search`.

## Tools

`web_search`, `fetch_url` (as `visit_webpage`), `wikipedia_search`.

## Cost bound

`max_steps=6`, `planning_interval=3`, and the resume uses `max_steps=2`. Worst case: about 12 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 4.0
- Mean cost per run: $0.0030 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.
