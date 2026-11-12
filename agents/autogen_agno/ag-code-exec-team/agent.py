"""Data-analysis team with a model-driven code executor on AutoGen AgentChat.

Derived from the AgentChat "Selector Group Chat" guide (PlanningAgent / DataAnalystAgent split with
TERMINATE) and the CodeExecutorAgent API (model_client-driven code generation and
approval_func for human-in-the-loop). Adapted: a SQL analyst pulls data from the Chinook / Northwind sample
databases and the finance price service; a CodeExecutorAgent writes Python and runs it in the tools-service
python_exec sandbox (instead of a local Docker executor); a rule-based auto-approver stands in for the human
reviewer and rejects destructive or exfiltrating code; RoundRobinGroupChat is bounded by message count.
"""
from __future__ import annotations

import re

import ag_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="ag-code-exec-team",
    framework="autogen",
    source_url="https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/selector-group-chat.ipynb",
    description="RoundRobin team where a SQL analyst gathers data and an auto-approved CodeExecutorAgent writes and runs Python in a sandbox.",
    complexity="complex",
    model=C.HAIKU,
    tracing="openinference",
)

ANALYST_SYSTEM = """You are a data analyst. Today is 2026-11-13.
Data sources: SQLite sample databases 'chinook' (music store) and 'northwind' (trading company) via the SQL tools,
and daily stock prices via finance_history.
On your first turn: inspect the schema if needed, run read-only queries to fetch exactly the data needed, then
hand the computation to code_executor by stating the data (inline, compact) and what to compute.
Never run write or DDL statements (INSERT, UPDATE, DELETE, DROP, ALTER); refuse such requests.
When code_executor has produced results, write the final answer for the user (numbers, short table, caveats about
data quality) and end with TERMINATE."""

EXECUTOR_SYSTEM = """You are a Python coding assistant. Write ONE self-contained ```python``` code block that
computes what the analyst asked for, using only the standard library (statistics, math, datetime, json, csv).
Embed the data you were given directly in the code and print the results clearly. Do not access files, the
network, environment variables or subprocesses."""

_BLOCKED = re.compile(
    r"(os\.environ|getenv|/etc/|subprocess|socket|requests\.|urllib|http\.client|shutil\.rmtree|os\.remove|"
    r"open\(\s*['\"]/|__import__|eval\(|exec\(|DROP\s+TABLE|DELETE\s+FROM)",
    re.IGNORECASE,
)


def auto_approver(request):
    """Simulated human reviewer for CodeExecutorAgent: approves analysis code, rejects risky code."""
    from autogen_agentchat.agents import ApprovalResponse

    hit = _BLOCKED.search(request.code)
    if hit:
        return ApprovalResponse(approved=False, reason=f"Rejected by reviewer: code uses {hit.group(0)!r}.")
    return ApprovalResponse(approved=True, reason="Approved by reviewer: read-only analysis code.")


def setup() -> None:
    C.setup_autogen_tracing()


async def arun(task: dict) -> str:
    from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat

    model_client = C.autogen_model(C.HAIKU)
    try:
        analyst = AssistantAgent(
            "data_analyst",
            model_client=model_client,
            description="Queries the SQL databases and price history, then writes the final answer.",
            tools=C.autogen_tools(T.sql_schema, T.sql_query, T.finance_history),
            system_message=ANALYST_SYSTEM,
            reflect_on_tool_use=True,
            max_tool_iterations=4,
        )
        code_executor = CodeExecutorAgent(
            "code_executor",
            code_executor=C.peira_python_executor(),
            model_client=model_client,
            system_message=EXECUTOR_SYSTEM,
            # max_retries_on_error needs structured output (RetryDecision); the Anthropic client has none.
            max_retries_on_error=0,
            approval_func=auto_approver,
            description="Writes and runs Python analysis code in a sandbox.",
        )
        termination = TextMentionTermination("TERMINATE", sources=["data_analyst"]) | MaxMessageTermination(6)
        team = RoundRobinGroupChat([analyst, code_executor], termination_condition=termination)
        result = await team.run(task=task["input"])
        return C.final_text(result, prefer_source="data_analyst")
    finally:
        await model_client.close()
