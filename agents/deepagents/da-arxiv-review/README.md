# da-arxiv-review

Source: https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py
Also: Docs: https://docs.langchain.com/oss/python/deepagents/subagents (CompiledSubAgent)

## What was adapted

Documented CompiledSubAgent pattern: a LangChain create_agent graph with response_format=PaperCard runs as the paper-analyst sub-agent, and the parent receives JSON. The orchestrator follows the deep_research workflow and writes /review.md. Caps: main 8, paper-analyst 4 per paper.

## Tools and features

arxiv_search, fetch_url, CompiledSubAgent, response_format, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 6.7. Mean cost per run: $0.0045 (Langfuse cost from scripts/inspect_trace.py).
