"""Text-to-SQL analyst, adapted from PydanticAI's sql_gen example.

Shape kept from the source: an Agent whose instructions embed the database schema and
few-shot examples, a union output type (Success | InvalidRequest), and an output validator
that rejects non-SELECT queries and test-runs the SQL, raising ModelRetry with the database
error so the model repairs its own query. The Postgres log table is replaced by the
Chinook and Northwind SQLite databases of the tools service. After validation, the agent
runs the final query and a second small agent writes the answer from the rows.
"""

import json
from dataclasses import dataclass, field
from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits, format_as_xml

import pl_common as pc
from peira_harness import AgentMeta, tools

AGENT = AgentMeta(
    id="pyd-sql-gen",
    framework="pydantic-ai",
    source_url="https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/sql_gen.py",
    description="Text-to-SQL agent with schema-in-instructions, union output, and an output validator that test-runs SQL and retries.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="native-otel",
)

SQL_EXAMPLES = [
    {"request": "top 5 artists by number of tracks", "db": "chinook",
     "response": "SELECT ar.Name, COUNT(*) AS tracks FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
                 "JOIN Track t ON t.AlbumId = al.AlbumId GROUP BY ar.ArtistId ORDER BY tracks DESC LIMIT 5"},
    {"request": "revenue per country in 2013", "db": "chinook",
     "response": "SELECT BillingCountry, ROUND(SUM(Total), 2) AS revenue FROM Invoice "
                 "WHERE strftime('%Y', InvoiceDate) = '2013' GROUP BY BillingCountry ORDER BY revenue DESC"},
    {"request": "products that are out of stock", "db": "northwind",
     "response": "SELECT ProductName, UnitsInStock FROM Products WHERE UnitsInStock = 0"},
    {"request": "which employee handled the most orders", "db": "northwind",
     "response": "SELECT e.FirstName || ' ' || e.LastName AS employee, COUNT(*) AS orders FROM Orders o "
                 "JOIN Employees e ON e.EmployeeID = o.EmployeeID GROUP BY e.EmployeeID ORDER BY orders DESC LIMIT 1"},
]


@dataclass
class Deps:
    schemas: dict[str, str]
    attempts: list[str] = field(default_factory=list)


class Success(BaseModel):
    """Response when SQL could be successfully generated."""

    db: Literal["chinook", "northwind"]
    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field("", description="Explanation of the SQL query, as markdown")


class InvalidRequest(BaseModel):
    """Response when the user input is not answerable from the databases or lacks information to generate SQL."""

    error_message: str


Response = Success | InvalidRequest

sql_agent = Agent[Deps, Response](
    pc.pyd_model(),
    name="sql_agent",
    output_type=Success | InvalidRequest,
    deps_type=Deps,
    retries=3,
)


@sql_agent.instructions
async def instructions(ctx: RunContext[Deps]) -> str:
    return f"""\
Given the following SQLite databases, your job is to write one SQL query that suits the user's request.
Pick the database that contains the data: 'chinook' (digital music store) or 'northwind' (food trading company).
Only write read-only SELECT (or WITH ... SELECT) queries. Quote table names with spaces, e.g. "Order Details".
If the request needs data that is not in either database, or asks to modify data, return InvalidRequest.

<chinook_schema>
{ctx.deps.schemas['chinook']}
</chinook_schema>

<northwind_schema>
{ctx.deps.schemas['northwind']}
</northwind_schema>

{format_as_xml(SQL_EXAMPLES, root_tag='examples')}
"""


@sql_agent.output_validator
async def validate_output(ctx: RunContext[Deps], output: Response) -> Response:
    if isinstance(output, InvalidRequest):
        return output
    output.sql_query = output.sql_query.replace("\\", "").strip().rstrip(";")
    head = output.sql_query.lstrip("( \n").upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        raise ModelRetry("Please create a SELECT query")
    ctx.deps.attempts.append(output.sql_query)
    # SQLite has no EXPLAIN through the tools service, so test-run the query with one row.
    res = await pc.safe_tool(tools.sql_query)(db=output.db, sql=output.sql_query, max_rows=1)
    if res.startswith("ERROR"):
        raise ModelRetry(f"Invalid query: {res}")
    return output


answer_agent = Agent(
    pc.pyd_model(),
    name="answer_agent",
    output_type=str,
    instructions=(
        "You turn SQL results into a short, direct answer to the user's question. "
        "Quote the key numbers. Mention if the result was truncated. Do not invent rows."
    ),
)

_SCHEMAS: dict[str, str] = {}


def _compact_schema(db: str) -> str:
    raw = tools.sql_schema(db)
    data = raw if isinstance(raw, dict) else json.loads(raw)
    lines = []
    for t in data.get("tables", []):
        cols = ", ".join(f"{c['name']} {c.get('type', '')}".strip() + (" PK" if c.get("pk") else "") for c in t.get("columns", []))
        fks = "; ".join(f"{f['column']} -> {f['references']}" for f in t.get("foreign_keys", []))
        lines.append(f"{t['name']}({cols})" + (f" FK: {fks}" if fks else "") + f" rows={t.get('row_count')}")
    return "\n".join(lines)


def setup() -> None:
    pc.pyd_setup()
    for db in ("chinook", "northwind"):
        _SCHEMAS[db] = _compact_schema(db)


async def arun(task: dict) -> str:
    pc.begin_task(AGENT.id, task)
    deps = Deps(schemas=_SCHEMAS)
    limits = UsageLimits(request_limit=8)
    result = await sql_agent.run(task["input"], deps=deps, usage_limits=limits)
    out = result.output
    if isinstance(out, InvalidRequest):
        return f"Cannot answer with SQL: {out.error_message}"
    rows = await pc.safe_tool(tools.sql_query)(db=out.db, sql=out.sql_query, max_rows=50)
    answer = await answer_agent.run(
        format_as_xml({"question": task["input"], "sql": out.sql_query, "db": out.db, "rows": rows}),
        usage=result.usage,
        usage_limits=limits,
    )
    return f"{answer.output}\n\nSQL ({out.db}):\n{out.sql_query}"
