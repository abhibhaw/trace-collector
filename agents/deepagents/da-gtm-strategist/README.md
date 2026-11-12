# da-gtm-strategist

Source: https://github.com/langchain-ai/deepagents/tree/main/examples/deploy-gtm-agent

## What was adapted

Kept: AGENTS.md memory, competitor-analysis skill, market-researcher sub-agent (its AGENTS.md as prompt, analyze-market skill, structured response with full_report_path). The async content-writer (remote Agent Protocol server) became a synchronous sub-agent. Company context (Lumen & Co) added to memory. Caps: main 10, market-researcher 7, content-writer 5.

## Tools and features

web_search, wikipedia_search, products_search, crm_search_customers, response_format (pydantic), skills, memory, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 11.7. Mean cost per run: $0.0112 (Langfuse cost from scripts/inspect_trace.py).
