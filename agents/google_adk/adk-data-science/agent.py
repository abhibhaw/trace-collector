"""Data science multi-agent: root planner -> per-database NL2SQL agents + Python analytics agent.

Derived from google/adk-samples python/agents/data-science (removed from main in #2591; pinned).
Adaptations:
- The two configured datasets (BigQuery + AlloyDB in the source) become the two SQLite sample
  databases of the tools service (chinook + northwind), each with its own NL2SQL agent that is
  invoked through a function tool wrapping AgentTool.run_async (same as the source's
  call_bigquery_agent / call_alloydb_agent).
- BigQuery execute_sql -> peira_harness sql_query; the after_tool_callback still stores query
  results in session state for the analytics agent.
- VertexAiCodeExecutor -> SandboxCodeExecutor, an ADK BaseCodeExecutor that runs the model's
  ```python blocks in the tools-service sandbox (python_exec) and records a TOOL span per run.
- The source's LLM-inside-a-tool nl2sql step and the BQML sub-agent (Vertex only) are dropped;
  the DB agent writes SQL itself. temperature settings are removed (proxy rule).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.code_executors import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.agent_tool import AgentTool
from opentelemetry import trace

import adk_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

prompts = C.load_local(__file__)
log = logging.getLogger(__name__)

AGENT = AgentMeta(
    id="adk-data-science",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/data-science",
    description="Root data-science agent plans queries, delegates NL2SQL to per-database agents (Chinook, Northwind) and analysis to a code-executing analytics agent.",
    complexity="complex",
    model=C.SONNET,
    tracing="openinference",
)

DATASETS = {
    "chinook": "Chinook: a digital music store (artists, albums, tracks, genres, playlists, customers, invoices, invoice lines, employees as sales support reps).",
    "northwind": "Northwind: a food import/export trading company (customers, suppliers, products, categories, orders, order details, employees, shippers, territories).",
}
CROSS_DATASET_RELATIONS = (
    "There are NO foreign keys between chinook and northwind. They are unrelated businesses. "
    "Only compare them at an aggregate level (e.g. countries both serve), joining in Python if needed."
)

_schema_cache: dict[str, str] = {}


def _schema_text(db: str) -> str:
    if db not in _schema_cache:
        s = T.sql_schema(db)
        if "error" in s:
            raise RuntimeError(f"sql_schema({db}) failed: {s['error']}")
        lines = []
        for t in s.get("tables", []):
            cols = ", ".join(f"{c['name']} {c['type']}{' PK' if c.get('pk') else ''}" for c in t["columns"])
            fks = "; ".join(f"{f['column']} -> {f['references']}" for f in t.get("foreign_keys", []))
            size = "view" if t.get("type") == "view" else f"{t.get('row_count')} rows"
            lines.append(f"- {t['name']} ({size}): {cols}" + (f" | FK: {fks}" if fks else ""))
        _schema_cache[db] = "\n".join(lines) + ("\n" + s["note"] if s.get("note") else "")
    return _schema_cache[db]


# --- callbacks ----------------------------------------------------------------------------------
def load_database_settings_in_context(callback_context: CallbackContext) -> None:
    """before_agent_callback: put database settings (schemas) into session state on first use."""
    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = {db: {"description": d} for db, d in DATASETS.items()}


def make_store_results_in_context(db: str):
    def store_results_in_context(
        tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
    ) -> dict | None:
        # Lets the analytics agent use the SQL results as context.
        if tool.name == "sql_query" and isinstance(tool_response, dict) and "rows" in tool_response:
            tool_context.state[f"{db}_query_result"] = tool_response
            tool_context.state["sql_query"] = args.get("sql")
        return None

    return store_results_in_context


# --- sandbox code executor ------------------------------------------------------------------------
_tracer = trace.get_tracer("adk-data-science.code-executor")


class SandboxCodeExecutor(BaseCodeExecutor):
    """Runs model-written code blocks in the tools-service Python sandbox (stateless)."""

    stateful: bool = False
    error_retry_attempts: int = 2

    def execute_code(self, invocation_context: InvocationContext, code_execution_input: CodeExecutionInput) -> CodeExecutionResult:
        with _tracer.start_as_current_span(
            "code_execution",
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "python_exec",
                "input.value": code_execution_input.code,
                "input.mime_type": "text/x-python",
            },
        ) as span:
            res = T.python_exec(code=code_execution_input.code, timeout_s=30)
            stdout = res.get("stdout", "") if isinstance(res, dict) else str(res)
            stderr = (res.get("stderr") or res.get("error") or "") if isinstance(res, dict) else ""
            if isinstance(res, dict) and res.get("status") == "timeout":
                stderr = (stderr + "\nExecution timed out.").strip()
            span.set_attribute("output.value", json.dumps({"stdout": stdout[-4000:], "stderr": stderr[-2000:]}))
            return CodeExecutionResult(stdout=stdout, stderr=stderr)


# --- agents --------------------------------------------------------------------------------------
def _db_instruction(db: str):
    def provider(ctx: ReadonlyContext) -> str:
        return f"""
      You are an AI assistant serving as a SQL expert for the SQLite database '{db}'.
      Your job is to help users generate SQL answers from natural language questions.

      {DATASETS[db]}

      <SCHEMA>
{_schema_text(db)}
      </SCHEMA>

      1. Write SQLite SQL for the question using only the schema above.
      2. Use the sql_query tool (db="{db}") to validate and execute the SQL.
      3. If there are any syntax errors in the query, address the error and re-run (at most 3 attempts).
      4. Generate the final result in JSON format with four keys: "explain", "sql", "sql_results", "nl_results".
        * "explain": step-by-step reasoning for how the query follows from the schema and question.
        * "sql": the SQL you executed.
        * "sql_results": the raw rows from sql_query (at most 30 rows; say how many were omitted).
        * "nl_results": natural language summary of results, otherwise None if the SQL is invalid.
      Only read-only SELECT statements are allowed.
    """

    return provider


def _make_db_agent(db: str) -> LlmAgent:
    return LlmAgent(
        model=C.claude(C.HAIKU),
        name=f"{db}_db_agent",
        description=f"NL2SQL agent for the {db} database.",
        instruction=_db_instruction(db),
        tools=[C.AT.sql_query],
        after_tool_callback=make_store_results_in_context(db),
    )


chinook_agent = _make_db_agent("chinook")
northwind_agent = _make_db_agent("northwind")

analytics_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    name="analytics_agent",
    instruction=prompts.return_instructions_analytics()
    .replace(
        "**Statefulness:** All code snippets are executed and the variables stay in\n  the environment. You NEVER need to re-initialize variables. You NEVER need to\n  reload files. You NEVER need to re-import libraries.",
        "**Statefulness:** Each code block runs in a FRESH Python process. Always\n  re-create the DataFrame from the data in the prompt and import what you need.",
    )
    .replace("**Imported Libraries:** The following libraries are ALREADY imported and\n  should NEVER be imported again:", "**Available Libraries:** import these as needed (no plotting backend; describe charts in text):")
    .replace("import matplotlib.pyplot as plt\n", "")
    + "\nWrite code in ```python blocks. Run at most 3 code blocks, then give the final answer.\n",
    code_executor=SandboxCodeExecutor(),
)


async def _call_db(agent: LlmAgent, db: str, question: str, tool_context: ToolContext):
    agent_tool = AgentTool(agent=agent)
    output = await agent_tool.run_async(args={"request": question}, tool_context=tool_context)
    tool_context.state[f"{db}_agent_output"] = output
    return output


async def call_chinook_agent(question: str, tool_context: ToolContext):
    """Tool to call the chinook database (nl2sql) agent: music store sales, tracks, customers, invoices."""
    return await _call_db(chinook_agent, "chinook", question, tool_context)


async def call_northwind_agent(question: str, tool_context: ToolContext):
    """Tool to call the northwind database (nl2sql) agent: trading company orders, products, suppliers."""
    return await _call_db(northwind_agent, "northwind", question, tool_context)


async def call_analytics_agent(question: str, tool_context: ToolContext):
    """
    This tool can generate Python code to process and analyze a dataset.

    Some of the tasks it can do in Python include:
    * Statistics, correlations, regressions and simple forecasts;
    * Processing or filtering existing datasets;
    * Combining datasets to create a joined dataset for further analysis.

    The Python modules available to it are: io, math, re, numpy, pandas, scipy.

    The tool DOES NOT have the ability to retrieve additional data from
    a database. Only the data already retrieved will be analyzed.

    Args:
        question (str): Natural language question or analytics request.
        tool_context (ToolContext): The tool context.

    Returns:
        Response from the analytics agent.
    """
    chinook_data = tool_context.state.get("chinook_query_result", "")
    northwind_data = tool_context.state.get("northwind_query_result", "")
    question_with_data = f"""
  Question to answer: {question}

  Actual data to analyze this question is available in the following data
  tables:

  <CHINOOK>
  {json.dumps(chinook_data, default=str)[:12000] if chinook_data else ""}
  </CHINOOK>

  <NORTHWIND>
  {json.dumps(northwind_data, default=str)[:12000] if northwind_data else ""}
  </NORTHWIND>

  """
    agent_tool = AgentTool(agent=analytics_agent)
    output = await agent_tool.run_async(args={"request": question_with_data}, tool_context=tool_context)
    tool_context.state["analytics_agent_output"] = output
    return output


def _root_instruction(ctx: ReadonlyContext) -> str:
    base = (
        prompts.return_instructions_root()
        .replace("Use 'call_bigquery_agent' or 'call_alloydb_agent'", "Use 'call_chinook_agent' or 'call_northwind_agent'")
        .replace("    - If the user specifically wants to work on BQML, route to the bqml_agent.\n", "")
    )
    defs = "\n<DATASETS>\n"
    for db, desc in DATASETS.items():
        defs += f"<{db.upper()}>\n<DESCRIPTION>\n{desc}\n</DESCRIPTION>\n<SCHEMA>\n{_schema_text(db)}\n</SCHEMA>\n</{db.upper()}>\n"
    defs += f"</DATASETS>\n<CROSS_DATASET_RELATIONS>\n{CROSS_DATASET_RELATIONS}\n</CROSS_DATASET_RELATIONS>\n"
    defs += "\nBQML is not available in this deployment; if asked for BigQuery ML, explain that and offer a Python analysis instead.\n"
    defs += "Skip step 2 (reporting the plan and waiting); state the plan briefly and execute it in the same turn.\n"
    return base + defs


root_agent = LlmAgent(
    model=C.claude(C.SONNET),
    name="data_science_root_agent",
    instruction=_root_instruction,
    global_instruction=f"""
            You are a Data Science and Data Analytics Multi Agent System.
            Todays date: {date.today()}
            """,
    tools=[call_chinook_agent, call_northwind_agent, call_analytics_agent],
    before_agent_callback=load_database_settings_in_context,
)


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(root_agent, task, app_name="data_science", max_llm_calls=25)
