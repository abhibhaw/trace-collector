# da-deep-research

Source: https://github.com/langchain-ai/deepagents/tree/main/examples/deep_research
Also: Critique sub-agent: https://github.com/langchain-ai/deepagents/blob/69076db4282c26d853304a93cc1cac8c4fb0314a/examples/research/research_agent.py

## What was adapted

Kept the deep_research workflow, delegation and researcher prompts, think_tool, /research_request.md and /final_report.md. Added the critique-agent from the earlier examples/research/research_agent.py (same repo, cited commit). Tavily replaced by web_search + fetch_url. Orchestrator on claude-sonnet-5-5; researcher, critique and general-purpose sub-agents on claude-haiku-5-5. Caps: orchestrator 9 model calls, researcher 5, critique 3, general-purpose 3; at most 3 parallel researchers and 1 delegation round.

## Tools and features

web_search, fetch_url, think_tool (local), virtual file tools, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 8.7. Mean cost per run: $0.0831 (Langfuse cost from scripts/inspect_trace.py).
