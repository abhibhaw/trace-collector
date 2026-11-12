# smol-math-solver

Source: https://github.com/huggingface/smolagents/blob/main/docs/source/en/guided_tour.md (final_answer_checks) and examples/smolagents_benchmark/run.py

## Shape

`CodeAgent` with `final_answer_checks=[answer_has_number]` (a variant of the guided tour's `is_integer`) and sympy. Local tools: `physical_constant` and `convert_units`.

## What was adapted

- The check accepts a number, or an explicit statement that the problem is ill-posed.
- Authorized imports: `math, cmath, fractions, decimal, statistics, itertools, sympy`. Code runs in the smolagents local executor in the pod. No file or network modules are allowed.

## Tools

Local tools only: CODATA constants and unit conversion.

## Cost bound

`max_steps=5`. Worst case: 6 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 1.7
- Mean cost per run: $0.0007 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.
