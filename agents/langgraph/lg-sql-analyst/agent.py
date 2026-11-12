"""SQL agent with dedicated nodes (LangGraph tutorial "Build a SQL agent", section 3).

Kept from the source: the custom MessagesState graph with dedicated nodes list_tables
(predetermined tool call), call_get_schema (forced tool call), get_schema (ToolNode),
generate_query, check_query (LLM double-check with the tutorial's checklist) and run_query
(ToolNode), the should_continue loop back to generate_query, and the system prompts.
Changed: SQLDatabaseToolkit over a local Chinook file is replaced by the tools service's SQL
endpoints over Chinook and Northwind, so every tool takes a `database` argument.
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import lg_common as C
from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="lg-sql-analyst",
    framework="langgraph",
    source_url="https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/sql/sql-agent.md",
    description="Fixed SQL pipeline: list tables, forced schema call, generate, LLM query check, run, retry on error.",
    complexity="simple",
    model=C.HAIKU,
    tracing="langchain",
)

DATABASES = ("chinook", "northwind")
DIALECT = "SQLite"
TOP_K = 5


def _T():
    from peira_harness import tools as T

    return T


def _tables(database: str) -> list[dict]:
    res = _T().sql_schema(database)
    if "error" in res:
        raise ValueError(res["error"])
    return res["tables"]


@tool
def sql_db_list_tables(database: str = "") -> str:
    """List the tables of a database. Pass database='chinook' or 'northwind'; empty lists both."""
    dbs = [database] if database else list(DATABASES)
    lines = []
    for db in dbs:
        try:
            names = ", ".join(t["name"] for t in _tables(db))
        except ValueError as e:
            names = f"Error: {e}"
        lines.append(f"{db}: {names}")
    return "\n".join(lines)


@tool
def sql_db_schema(database: str, table_names: str) -> str:
    """Input is the database ('chinook' or 'northwind') and a comma-separated list of tables.
    Output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling sql_db_list_tables first!
    Example Input: database='chinook', table_names='Track, Genre'"""
    wanted = {t.strip().strip('"').lower() for t in table_names.split(",") if t.strip()}
    try:
        by_name = {t["name"].lower(): t for t in _tables(database)}
    except ValueError as e:
        return f"Error: {e}"
    missing = sorted(w for w in wanted if w not in by_name)
    parts = []
    for w in sorted(wanted - set(missing)):
        t = by_name[w]
        cols = ",\n    ".join(f'"{c["name"]}" {c["type"]}{" PRIMARY KEY" if c["pk"] else ""}' for c in t["columns"])
        fks = "".join(f',\n    FOREIGN KEY("{f["column"]}") REFERENCES {f["references"]}' for f in t["foreign_keys"])
        q = t["name"].replace('"', '""')
        try:
            sample = _T().sql_query(database, f'SELECT * FROM "{q}" LIMIT 3', max_rows=3)
            if "error" in sample:
                raise ValueError(sample["error"])
            rows = "\n".join("\t".join(str(v)[:40] for v in r) for r in sample["rows"])
            sample_txt = f"/*\n3 rows from {t['name']} table:\n" + "\t".join(sample["columns"]) + f"\n{rows}\n*/"
        except Exception as e:
            sample_txt = f"/* sample rows unavailable: {e} */"
        parts.append(f'CREATE TABLE "{t["name"]}" (\n    {cols}{fks}\n)  -- {t["row_count"]} rows\n{sample_txt}')
    if missing:
        parts.append(f"Error: table(s) not found in {database}: {', '.join(missing)}")
    return "\n\n".join(parts)


@tool
def sql_db_query(database: str, query: str) -> str:
    """Input is the database ('chinook' or 'northwind') and a detailed and correct read-only SQL query.
    Output is a result from the database. If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue
    with an unknown column, use sql_db_schema to query the correct table fields."""
    try:
        res = _T().sql_query(database, query, max_rows=50)
    except Exception as e:  # the tutorial surfaces DB errors to the model so it can correct the query
        return f"Error: {e}"
    if "error" in res:
        return f"Error: {res['error']}"
    return C.to_json({k: res[k] for k in ("columns", "rows", "row_count", "truncated") if k in res}, 6000)


generate_query_system_prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {DIALECT} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {TOP_K} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the user asks you to modify data, refuse and explain that you have read-only access.
Available databases: chinook (digital media store) and northwind (food trading company).
"""

check_query_system_prompt = f"""
You are a SQL expert with a strong attention to detail.
Double check the {DIALECT} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
"""

MAX_QUERIES = 4
graph = None


def list_tables(state: MessagesState):
    tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "list_tables_0", "type": "tool_call"}
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    tool_message = sql_db_list_tables.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}


def call_get_schema(state: MessagesState):
    llm_with_tools = C.model().bind_tools([sql_db_schema], tool_choice="any")
    # The proxy models reject a conversation that ends with an assistant turn (prefill).
    response = llm_with_tools.invoke(
        state["messages"] + [{"role": "user", "content": "Fetch the schema of the tables relevant to my question."}]
    )
    return {"messages": [response]}


def generate_query(state: MessagesState):
    system_message = {"role": "system", "content": generate_query_system_prompt}
    llm_with_tools = C.model().bind_tools([sql_db_query])
    queries_run = sum(1 for m in state["messages"] if getattr(m, "name", None) == "sql_db_query")
    if queries_run >= MAX_QUERIES:
        # Budget spent: answer from the results so far (no tools bound).
        response = C.model().invoke(
            [system_message] + state["messages"]
            + [{"role": "user", "content": "Query budget reached. Answer now from the results above."}]
        )
    else:
        response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def check_query(state: MessagesState):
    system_message = {"role": "system", "content": check_query_system_prompt}
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {
        "role": "user",
        "content": f"database: {tool_call['args'].get('database')}\nquery:\n{tool_call['args']['query']}",
    }
    llm_with_tools = C.model().bind_tools([sql_db_query], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "check_query"


def setup() -> None:
    global graph
    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(ToolNode([sql_db_schema], name="get_schema"))
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(ToolNode([sql_db_query], name="run_query"))
    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    graph = builder.compile(name="sql_agent")


def run(task: dict) -> str:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": task["input"]}]}, config=C.run_config(recursion_limit=25)
    )
    return C.text(result["messages"][-1])
