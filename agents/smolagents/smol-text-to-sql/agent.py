"""Text-to-SQL CodeAgent, adapted from the smolagents text_to_sql example and docs tutorial.

Source shape: one CodeAgent with a single `sql_engine` tool whose description embeds the table schema.
The docs tutorial then extends the tool description to several tables so the agent can write JOINs.
Adaptation: the in-memory `receipts` table is replaced by the Chinook and Northwind sample databases
served by the tools service. The tool description is built from the live schema, as in the tutorial.
"""
from __future__ import annotations

import uuid

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import CodeAgent, Tool

AGENT = AgentMeta(
    id="smol-text-to-sql",
    framework="smolagents",
    source_url="https://github.com/huggingface/smolagents/blob/main/examples/text_to_sql.py",
    description="CodeAgent that answers business questions by writing SQLite queries against Chinook and Northwind.",
    complexity="simple",
    model="claude-haiku-5-5",
    tracing="openinference",
)

_SCHEMA_TEXT: str | None = None


def _schema_text() -> str:
    """Build the 'Columns:' description used by the source example, for both databases."""
    global _SCHEMA_TEXT
    if _SCHEMA_TEXT is None:
        parts = []
        for db in ("chinook", "northwind"):
            s = T.sql_schema(db)
            if "error" in s:
                raise RuntimeError(f"sql_schema({db}) failed: {s['error']}")
            parts.append(f"Database '{db}':")
            for t in s["tables"]:
                cols = ", ".join(f"{c['name']} {c.get('type') or ''}".strip() for c in t["columns"])
                parts.append(f"  Table '{t['name']}' ({t.get('row_count', '?')} rows): {cols}")
        _SCHEMA_TEXT = "\n".join(parts)
    return _SCHEMA_TEXT


class SQLEngineTool(Tool):
    name = "sql_engine"
    inputs = {
        "database": {"type": "string", "description": "Which database to query: 'chinook' or 'northwind'."},
        "query": {"type": "string", "description": "The query to perform. This should be correct SQLite SQL."},
    }
    output_type = "string"

    def __init__(self, schema: str):
        self.description = (
            "Allows you to perform SQL queries on the tables. Returns a string representation of the result.\n"
            "Only single read-only SELECT statements are allowed. At most 100 rows are returned.\n"
            "The databases and tables are described as follows:\n" + schema
        )
        super().__init__()

    def forward(self, database: str, query: str) -> str:
        res = T.sql_query(database, query, max_rows=100)
        if "error" in res:
            return f"SQL error: {res['error']}"
        out = " | ".join(res["columns"])
        for row in res["rows"]:
            out += "\n" + str(tuple(row))
        if res.get("truncated"):
            out += f"\n... (truncated, {res.get('row_count')} rows total)"
        return out


def setup() -> None:
    C.setup_tracing()
    _schema_text()


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    agent = CodeAgent(
        tools=[SQLEngineTool(_schema_text())],
        model=C.make_model(AGENT.model),
        max_steps=6,
        verbosity_level=0,
        instructions=(
            "Pick the database that matches the question: chinook is a digital music store, "
            "northwind is a food trading company. If the question cannot be answered from either "
            "database, say so plainly instead of inventing data."
        ),
    )
    return C.last_text(agent.run(task["input"]))
