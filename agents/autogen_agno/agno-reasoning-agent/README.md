# agno-reasoning-agent

Reasoning agent on Agno with ReasoningTools (think / analyze scratchpad tools).

- Framework: agno
- Source: https://github.com/agno-agi/agno/blob/main/cookbook/10_reasoning/tools/reasoning_tools.py
- Complexity: medium
- Model: claude-haiku-5-5
- Tracing: OpenInference Agno instrumentor (agent, team, workflow, tool and LLM spans)

## What was adapted

Derived from the Agno cookbook "Reasoning Tools" (Agent + ReasoningTools(add_instructions=True) with the
expert problem-solving instructions). Adapted: besides pure reasoning puzzles, the agent answers
company-policy and HR questions for Lumen & Co, so it also gets HR, policy, calendar and sandboxed
Python tools. The tool-call limit bounds the think/act loop.

## Tools

- `calendar_free_slots` (peira_harness.tools)
- `calendar_get_events` (peira_harness.tools)
- `hr_get_employee` (peira_harness.tools)
- `hr_policy_lookup` (peira_harness.tools)
- `hr_search_employees` (peira_harness.tools)
- `python_exec` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [2, 2, 4] (mean 2.7).
Cost per run from Langfuse: ['$0.0009', '$0.0011', '$0.0043'] (mean $0.0021).
Projected cost for 100 tasks: about $0.21.
