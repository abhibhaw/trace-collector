# str-graph-workflow

Strands GraphBuilder workflow: a classifier routes to a technical branch or a finance->(tech, market) fan-out joined by a risk analyst.

- Framework: strands
- Source: https://github.com/strands-agents/samples/blob/main/python/01-learn/12-graph/graph.ipynb
- Model: claude-haiku-5-5
- Complexity: complex
- Tracing: native-otel (see `../dsh_common.py`)

## What was adapted

Derived from strands-agents/samples python/01-learn/12-graph/graph.ipynb, which shows three topologies:
basic chain, parallel processing (finance_expert -> tech_expert + market_expert -> risk_analyst), and
branching with conditions (classifier -> technical_report | business_report via is_technical/is_business).
This agent combines the two latter graphs into one GraphBuilder graph, keeping the notebook's agent names,
system prompts and condition functions:

    classifier --is_business--> finance_expert --> tech_expert   --\
                                              \--> market_expert --> risk_analyst
    classifier --is_technical-> technical_report ---------------------/

Changed: Bedrock -> Anthropic via the proxy; experts get tools (web search, Lumen & Co catalog,
finance quotes, sandboxed Python for ROI math); execution is bounded (max_node_executions, timeouts).

## Tools

web_search, python_exec, finance_quote, products_search (tools service)

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 7.3 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0041 (Langfuse per-trace cost); smoke total $0.0124.
