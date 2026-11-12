# agno-finance-agent

Finance agent on Agno with the FinanceTools toolkit and a custom provider.

- Framework: agno
- Source: https://github.com/agno-agi/agno/blob/main/cookbook/91_tools/finance/06_custom_provider.py
- Complexity: simple
- Model: claude-haiku-5-5
- Tracing: OpenInference Agno instrumentor (agent, team, workflow, tool and LLM spans)

## What was adapted

Derived from the Agno cookbook "Market Brief with FinanceTools" (one agent, one FinanceTools toolkit,
"lead with the answer, then show the evidence") and "Bring Your Own Provider" (FinanceProvider subclass
returning the normalized dataclasses). Adapted: Yahoo Finance is replaced by a provider backed by the
tools-service finance quotes (synthetic market data); news is not available from this provider.

## Tools

- `finance_history` (peira_harness.tools)
- `finance_quote` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [2, 2, 2] (mean 2.0).
Cost per run from Langfuse: ['$0.0006', '$0.0005', '$0.0006'] (mean $0.0006).
Projected cost for 100 tasks: about $0.06.
