# crew-stock-analysis

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis

Customer request: {request}\n\n

## What was adapted

Stock analysis crew, adapted from crewAI-examples/crews/stock_analysis.

Shape kept from the source: three agents (research analyst, financial analyst, investment advisor)
and four sequential tasks (research -> financial analysis -> filings analysis -> recommendation).
Adapted: SEC/EDGAR scraping and WebsiteSearchTool are replaced by the synthetic finance API and web
search; the source's CalculatorTool is kept as a local tool; the "filings" task works on the price
history (the only filing-like data available) with python_exec for computations.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: simple.
- Shared tools: fetch_url, finance_history, finance_quote, python_exec, web_search.
- Local tools: calculator.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 16.0 (per task: 13, 18, 17).
Mean cost per run: $0.0334 (Langfuse trace cost).
