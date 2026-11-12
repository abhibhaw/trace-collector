# ag-code-exec-team

Data-analysis team with a model-driven code executor on AutoGen AgentChat.

- Framework: autogen
- Source: https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/selector-group-chat.ipynb
- Complexity: complex
- Model: claude-haiku-5-5
- Tracing: OpenInference AutoGen instrumentor (team, agent, tool spans) plus OpenInference Anthropic instrumentor (LLM generations)

## What was adapted

Derived from the AgentChat "Selector Group Chat" guide (PlanningAgent / DataAnalystAgent split with
TERMINATE) and the CodeExecutorAgent API (model_client-driven code generation and
approval_func for human-in-the-loop). Adapted: a SQL analyst pulls data from the Chinook / Northwind sample
databases and the finance price service; a CodeExecutorAgent writes Python and runs it in the tools-service
python_exec sandbox (instead of a local Docker executor); a rule-based auto-approver stands in for the human
reviewer and rejects destructive or exfiltrating code; RoundRobinGroupChat is bounded by message count.

## Tools

- `finance_history` (peira_harness.tools)
- `sql_query` (peira_harness.tools)
- `sql_schema` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [3, 3, 5] (mean 3.7).
Cost per run from Langfuse: ['$0.0016', '$0.0018', '$0.0069'] (mean $0.0034).
Projected cost for 100 tasks: about $0.34.
