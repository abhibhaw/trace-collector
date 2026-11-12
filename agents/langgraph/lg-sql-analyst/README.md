# lg-sql-analyst

Fixed SQL pipeline: list tables, forced schema call, generate, LLM query check, run, retry on error.

- Source: https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/sql/sql-agent.md
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: simple.

## Derived from

SQL agent with dedicated nodes (LangGraph tutorial "Build a SQL agent", section 3).

## What was adapted

Kept from the source: the custom MessagesState graph with dedicated nodes list_tables
(predetermined tool call), call_get_schema (forced tool call), get_schema (ToolNode),
generate_query, check_query (LLM double-check with the tutorial's checklist) and run_query
(ToolNode), the should_continue loop back to generate_query, and the system prompts.
Changed: SQLDatabaseToolkit over a local Chinook file is replaced by the tools service's SQL
endpoints over Chinook and Northwind, so every tool takes a `database` argument.

## Tools

sql_schema and sql_query (wrapped as sql_db_list_tables, sql_db_schema with 3 sample rows, sql_db_query), databases chinook and northwind

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [6, 2, 4] (mean 4.0). Cost per run from Langfuse: mean $0.0013 (total $0.0040 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.
