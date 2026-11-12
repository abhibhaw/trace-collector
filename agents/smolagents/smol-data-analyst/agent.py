"""Data analyst CodeAgent with pandas, adapted from the Hugging Face cookbook "agent_data_analyst".

Source shape (huggingface/cookbook notebooks/en/agent_data_analyst.ipynb): a single CodeAgent with
additional_authorized_imports=["numpy", "pandas", ...] that is run with
`agent.run(prompt, additional_args=dict(additional_notes=..., source_file=...))`: the analyst
loads data with pandas, explores it, and reports numeric findings.
Adaptation: instead of reading a CSV from disk, the agent loads data through tools that return pandas
DataFrames (SQL over Northwind/Chinook, synthetic market prices, the mock order system). The local
executor only authorizes pandas, numpy and a few stdlib modules; it has no file or network access.
Plotting is dropped. A step callback records which datasets were loaded, and a final-answer check
requires the answer to contain at least one concrete number.
"""
from __future__ import annotations

import re
import uuid

import pandas as pd

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import ActionStep, CodeAgent, tool

AGENT = AgentMeta(
    id="smol-data-analyst",
    framework="smolagents",
    source_url="https://github.com/huggingface/cookbook/blob/main/notebooks/en/agent_data_analyst.ipynb",
    description="Pandas data-analyst CodeAgent that loads SQL, market and order data as DataFrames and reports numeric findings.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="openinference",
)

AUTHORIZED_IMPORTS = ["pandas", "numpy", "math", "statistics", "datetime", "collections"]

ADDITIONAL_NOTES = """
### Variable Notes
- `northwind` is a food trading company database (Orders, "Order Details", Products, Categories, Customers,
  Employees, Suppliers, Shippers). Revenue = UnitPrice * Quantity * (1 - Discount) in "Order Details".
- `chinook` is a digital music store database (Invoice, InvoiceLine, Track, Album, Artist, Genre, Customer, Employee).
- Market data is synthetic daily OHLCV for fictional tickers (and a few real tickers with synthetic prices).
- The order system belongs to Lumen & Co, a smart-home device seller. Its "today" is 2026-11-13.
- Load only what you need. Aggregate in SQL when that is simpler, then finish the analysis in pandas.
"""


@tool
def load_sql(database: str, query: str) -> pd.DataFrame:
    """Runs one read-only SQLite SELECT query and returns the result as a pandas DataFrame (max 200 rows).
    Aggregate in SQL (GROUP BY) when the raw table would exceed 200 rows.

    Args:
        database: 'northwind' or 'chinook'.
        query: A single SELECT statement in SQLite syntax.
    """
    res = T.sql_query(database, query, max_rows=200)
    if "error" in res:
        raise ValueError(f"SQL error: {res['error']}")
    df = pd.DataFrame(res["rows"], columns=res["columns"])
    df.attrs["truncated"] = bool(res.get("truncated"))
    return df


@tool
def describe_database(database: str) -> str:
    """Lists the tables of a database with their columns and row counts.

    Args:
        database: 'northwind' or 'chinook'.
    """
    res = T.sql_schema(database)
    if "error" in res:
        return f"Error: {res['error']}"
    return "\n".join(
        f"{t['name']} ({t.get('row_count')} rows): " + ", ".join(c["name"] for c in t["columns"]) for t in res["tables"]
    )


@tool
def load_price_history(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Loads historical OHLCV price bars for one ticker as a DataFrame with a `date` column.

    Args:
        symbol: Ticker symbol, for example 'LMNC'.
        start_date: First date, 'YYYY-MM-DD'. Use '' for the start of the data.
        end_date: Last date, 'YYYY-MM-DD'. Use '' for the latest date.
        interval: 'daily', 'weekly' or 'monthly'.
    """
    res = T.finance_history(symbol, start_date=start_date, end_date=end_date, interval=interval)
    if "error" in res:
        raise ValueError(res["error"])
    bars = res.get("bars") or res.get("rows") or []
    df = pd.DataFrame(bars, columns=res.get("columns")) if bars and isinstance(bars[0], list) else pd.DataFrame(bars)
    for k, v in res.items():
        if k not in ("bars", "rows", "columns") and isinstance(v, (str, int, float)):
            df.attrs[k] = v
    return df


@tool
def load_orders(status: str, date_from: str, date_to: str, sku: str) -> pd.DataFrame:
    """Loads up to 100 Lumen & Co orders (newest first) as a DataFrame, filtered by status, date range and SKU.

    Args:
        status: Order status filter, or '' for any status.
        date_from: Earliest order date 'YYYY-MM-DD', or ''.
        date_to: Latest order date 'YYYY-MM-DD', or ''.
        sku: Only orders containing this SKU, or ''.
    """
    res = T.orders_list(status=status, date_from=date_from, date_to=date_to, sku=sku, limit=100)
    if "error" in res:
        raise ValueError(res["error"])
    return pd.json_normalize(res.get("orders", []))


def record_datasets(memory_step: ActionStep, agent: CodeAgent) -> None:
    """Step callback: keep a running list of the tool calls that loaded data, in agent.state."""
    for tc in memory_step.tool_calls or []:
        if tc.name == "python_interpreter":
            for fn in ("load_sql", "load_price_history", "load_orders"):
                if fn + "(" in str(tc.arguments):
                    agent.state.setdefault("datasets_loaded", []).append(fn)


def has_number(final_answer, memory, agent=None) -> bool:
    if re.search(r"\d", str(final_answer)):
        return True
    raise ValueError("An analyst answer must report concrete numbers (counts, totals, percentages). Add them.")


def setup() -> None:
    C.setup_tracing()


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    agent = CodeAgent(
        tools=[load_sql, describe_database, load_price_history, load_orders],
        model=C.make_model(AGENT.model),
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        max_steps=7,
        verbosity_level=0,
        step_callbacks=[record_datasets],
        final_answer_checks=[has_number],
        max_print_outputs_length=4000,
    )
    prompt = (
        "You are an expert data analyst. Answer the request below by loading the data you need with your tools "
        "and analysing it with pandas. Report your findings with concrete numbers, and state any data caveats "
        "(truncated loads, missing values, ambiguous definitions).\n\nRequest: " + task["input"]
    )
    result = agent.run(prompt, additional_args=dict(additional_notes=ADDITIONAL_NOTES))
    return C.last_text(result)
