"""Text-to-SQL deep agent over the Chinook and Northwind sample databases.

Adapted from langchain-ai/deepagents examples/text-to-sql-agent. Kept: AGENTS.md memory, the
query-writing and schema-exploration skills (verbatim), the LangChain SQLDatabaseToolkit tool
names (sql_db_list_tables, sql_db_schema, sql_db_query), planning with write_todos for complex
questions, and subagents=[] (only the built-in general-purpose sub-agent). Changed: the local
chinook.db + SQLDatabaseToolkit -> the shared read-only SQL tools service (adds Northwind);
FilesystemBackend -> default StateBackend, so memory and skills are seeded as virtual files.
"""
import json

import da_common as C
from deepagents import create_deep_agent
from langchain_core.tools import tool

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="da-text-to-sql",
    framework="deepagents",
    source_url="https://github.com/langchain-ai/deepagents/tree/main/examples/text-to-sql-agent",
    description="Text-to-SQL deep agent with AGENTS.md memory and progressive-disclosure skills that explores schemas, plans complex joins with todos and runs read-only queries.",
    complexity="simple",
    model=C.HAIKU,
    tracing="langchain",
)


def _dump(x) -> str:
    return json.dumps(x, default=str)[:12000]


@tool
@C.safe
def sql_db_list_tables(db: str) -> str:
    """List the tables in a database, with row counts.

    Args:
        db: 'chinook' or 'northwind'.
    """
    s = T.sql_schema(db)
    return _dump([{"table": t["name"], "rows": t.get("row_count")} for t in s.get("tables", [])])


@tool
@C.safe
def sql_db_schema(db: str, table_names: str) -> str:
    """Get columns, primary keys and foreign keys for specific tables. Call sql_db_list_tables first.

    Args:
        db: 'chinook' or 'northwind'.
        table_names: Comma-separated table names, e.g. "Invoice, Customer".
    """
    wanted = {n.strip().strip('"').lower() for n in table_names.split(",") if n.strip()}
    s = T.sql_schema(db)
    found = [t for t in s.get("tables", []) if t["name"].lower() in wanted]
    missing = wanted - {t["name"].lower() for t in found}
    out = {"tables": found}
    if missing:
        out["error"] = f"tables not found: {sorted(missing)}"
    return _dump(out)


@tool
@C.safe
def sql_db_query(db: str, query: str) -> str:
    """Execute one read-only SELECT query and return columns and rows. On error, rewrite the query and try again.

    Args:
        db: 'chinook' or 'northwind'.
        query: A single SQLite SELECT (or WITH ... SELECT) statement.
    """
    return _dump(T.sql_query(db, query, max_rows=50))


_agent = None
_seed: dict[str, str] = {}


def setup() -> None:
    global _agent, _seed
    _seed = C.load_seed(__file__)
    _agent = create_deep_agent(
        model=C.model(C.HAIKU),
        tools=[sql_db_list_tables, sql_db_schema, sql_db_query],
        memory=["/memory/AGENTS.md"],
        skills=["/skills/"],
        subagents=[],
        middleware=[C.todos(), C.call_limit(14)],
        name="text-to-sql",
    )


def run(task: dict) -> str:
    result = C.invoke(_agent, task["input"], recursion_limit=200, files=_seed, run_name=AGENT.id)
    return C.final_answer(result)
