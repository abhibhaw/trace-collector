# da-coding-agent

Source: https://github.com/langchain-ai/deepagents/tree/main/examples/deploy-coding-agent

## What was adapted

Kept: AGENTS.md phased workflow (plan, implement, review, deliver), planning, coding-prefs and code-review skills, and the researcher sub-agent. The LangSmith sandbox shell `execute` is replaced by `run_python_file` (virtual-FS script and modules sent to the sandboxed python_exec). Removed git, pytest and the bundled lint_check.py script. Caps: main 14, researcher 4, general-purpose 4.

## Tools and features

python_exec (via run_python_file), web_search, fetch_url, virtual file tools, skills, memory, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 6.3. Mean cost per run: $0.0047 (Langfuse cost from scripts/inspect_trace.py).
