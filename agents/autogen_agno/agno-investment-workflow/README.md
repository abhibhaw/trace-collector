# agno-investment-workflow

Investment report workflow on Agno Workflows (function step, Router, Parallel, Condition, agent steps).

- Framework: agno
- Source: https://github.com/agno-agi/agno/blob/main/cookbook/04_workflows/02_conditional_execution/condition_with_parallel.py
- Complexity: complex
- Model: claude-sonnet-5-5
- Tracing: OpenInference Agno instrumentor (agent, team, workflow, tool and LLM spans)

## What was adapted

Derived from the Agno cookbook workflows "Condition With Parallel" (conditional research branches run in
parallel before writing steps) and "Router Basic" (selector function picks the research path), applied to
the cookbook's investment-report use case. Adapted: a function step parses and validates tickers against
the finance service; a Router sends ticker requests to parallel market-data / fundamentals / news research
and ticker-less requests to a sector research path; a Condition adds a risk-review step when the market
analyst flags high volatility or the user asks about risk; a Claude Sonnet writer produces the report.

## Tools

- `finance_history` (peira_harness.tools)
- `finance_quote` (peira_harness.tools)
- `python_exec` (peira_harness.tools)
- `web_search` (peira_harness.tools)
- `wikipedia_search` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [3, 7, 8] (mean 6.0).
Cost per run from Langfuse: ['$0.0113', '$0.0305', '$0.0587'] (mean $0.0335).
Projected cost for 100 tasks: about $3.35.
