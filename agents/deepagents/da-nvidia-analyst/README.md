# da-nvidia-analyst

Source: https://github.com/langchain-ai/deepagents/tree/main/examples/nvidia_deep_agent

## What was adapted

Kept the orchestrator, researcher-agent and data-processor-agent, the source AGENTS.md as memory, and the cudf-analytics skill verbatim. The pandas fallback runs because the sandbox has no GPU. Model mix: frontier orchestrator (claude-sonnet-5-5) with cheaper sub-agents (claude-haiku-5-5 instead of Nemotron). The Modal GPU sandbox and `execute` are replaced by `run_python_file`, which sends a virtual-FS script and its CSVs to the sandboxed python_exec tool. Removed: the data-visualization skill (PNG charts cannot come back), the dataset-download section and the cuML/document skills. Tavily is replaced by web_search and wikipedia_search. Data comes from finance_quote and finance_history. Caps: orchestrator 8, researcher 4, data-processor 8, general-purpose 3.

## Tools and features

finance_quote, finance_history, python_exec (via run_python_file), web_search, wikipedia_search, virtual file tools, skills, memory, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 15.0. Mean cost per run: $0.1111 (Langfuse cost from scripts/inspect_trace.py).
