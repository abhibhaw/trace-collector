# da-text-to-sql

Source: https://github.com/langchain-ai/deepagents/tree/main/examples/text-to-sql-agent

## What was adapted

Kept: AGENTS.md (extended to Chinook + Northwind and an untrusted-text rule), query-writing and schema-exploration skills verbatim, the SQLDatabaseToolkit tool names (sql_db_list_tables, sql_db_schema, sql_db_query), and subagents=[]. The local chinook.db and toolkit are replaced by the shared read-only SQL tools service. sql_db_query_checker is dropped (it needs an extra LLM call). FilesystemBackend -> StateBackend. Cap: 14 model calls.

## Tools and features

sql_schema, sql_query (wrapped as toolkit-named tools), skills, memory, write_todos

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 2.3. Mean cost per run: $0.0017 (Langfuse cost from scripts/inspect_trace.py).
