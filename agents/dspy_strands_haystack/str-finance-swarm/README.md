# str-finance-swarm

Research-director agent that pulls a quote, runs a 3-agent Strands Swarm with handoffs, and writes an investment report.

- Framework: strands
- Source: https://github.com/strands-agents/samples/tree/main/python/04-industry-use-cases/finance/finance-assistant-swarm-agent
- Model: claude-sonnet-5-5
- Complexity: complex
- Tracing: native-otel (see `../dsh_common.py`)

## What was adapted

Derived from strands-agents/samples python/04-industry-use-cases/finance/finance-assistant-swarm-agent
(finance_assistant_swarm.py) and the Swarm tutorial (python/01-learn/11-swarm/swarm.ipynb).
Kept: the "senior research director" orchestrator with its WORKFLOW / CRITICAL RULES / REPORT STRUCTURE prompt,
the tools get_real_stock_data and analyze_company_with_collaborative_swarm, the 3-agent Swarm
(company_strategist -> financial_analyst -> market_analyst) with its prompts, max_handoffs=3, max_iterations=3,
and the ticker extraction helper.
Changed: yfinance -> tools-service finance quotes/history (synthetic prices, 50 fictional + 6 real tickers);
company news -> keyless web search; Bedrock Opus -> claude-sonnet-5-5 through the proxy; tools are async so
the swarm's spans stay inside the task trace.

## Tools

finance_quote, finance_history, web_search (tools service); metrics computed locally

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 7.0 (models seen: claude-sonnet-5-5).
- Mean cost per run: $0.0624 (Langfuse per-trace cost); smoke total $0.1872.
