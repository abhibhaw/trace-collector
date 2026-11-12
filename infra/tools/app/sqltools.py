"""Read-only SQL over the public Chinook and Northwind SQLite sample databases."""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import time

from registry import ToolError, tool

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DBS = {"chinook": "chinook.sqlite", "northwind": "northwind.sqlite"}
MAX_ROWS = 200
TIME_LIMIT_S = 8.0

_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED.add(sqlite3.SQLITE_RECURSIVE)
_SCHEMA_CACHE: dict[str, dict] = {}


def db_path(db: str) -> str:
    name = DBS.get(db.strip().lower())
    if not name:
        raise ToolError(f"Unknown database '{db}'. Available: {', '.join(DBS)}")
    return os.path.join(DATA_DIR, name)


def _connect(db: str) -> sqlite3.Connection:
    path = db_path(db)
    if not os.path.exists(path):
        raise ToolError(f"Database '{db}' is not loaded on this replica.")
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = 1")

    def auth(action, a1, a2, dbname, src):
        if action == sqlite3.SQLITE_PRAGMA and a1 in ("table_info", "foreign_key_list", "index_list", "table_xinfo"):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY
    conn.set_authorizer(auth)
    return conn


def ready() -> bool:
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in DBS.values())


def _run(db: str, sql: str, max_rows: int) -> dict:
    conn = _connect(db)
    deadline = time.monotonic() + TIME_LIMIT_S
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise ToolError(f"Query exceeded the {TIME_LIMIT_S:.0f}s time limit. Add filters or a LIMIT.") from None
        if "not authorized" in str(e):
            raise ToolError("Only read-only SELECT queries are allowed.") from None
        raise ToolError(f"SQL error: {e}") from None
    except (sqlite3.Warning, sqlite3.ProgrammingError) as e:
        raise ToolError(f"SQL error: {e}. Send exactly one SELECT statement.") from None
    except sqlite3.DatabaseError as e:
        raise ToolError(f"SQL error: {e}") from None
    finally:
        conn.close()
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return {"db": db, "columns": cols, "rows": [list(r) for r in rows], "row_count": len(rows), "truncated": truncated}


@tool("sql")
async def sql_query(db: str, sql: str, max_rows: int = 100) -> dict:
    """Run one read-only SELECT on 'chinook' or 'northwind'."""
    s = re.sub(r"(--[^\n]*|/\*.*?\*/)", " ", sql, flags=re.S).strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", s):
        raise ToolError("Only read-only SELECT (or WITH ... SELECT) queries are allowed.")
    return await asyncio.to_thread(_run, db.strip().lower(), s, max(1, min(max_rows, MAX_ROWS)))


def _schema(db: str) -> dict:
    if db in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db]
    path = db_path(db)
    if not os.path.exists(path):
        raise ToolError(f"Database '{db}' is not loaded on this replica.")
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        tables = []
        for (name, kind) in conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            q = name.replace('"', '""')
            cols = [{"name": c[1], "type": c[2], "pk": bool(c[5])} for c in conn.execute(f'PRAGMA table_info("{q}")')]
            fks = [{"column": f[3], "references": f"{f[2]}.{f[4]}"} for f in conn.execute(f'PRAGMA foreign_key_list("{q}")')]
            n = conn.execute(f'SELECT COUNT(*) FROM "{q}"').fetchone()[0] if kind == "table" else None
            tables.append({"name": name, "type": kind, "row_count": n, "columns": cols, "foreign_keys": fks})
    finally:
        conn.close()
    out = {"db": db, "tables": tables, "note": "Quote table names that contain spaces, e.g. \"Order Details\"."}
    _SCHEMA_CACHE[db] = out
    return out


@tool("sql")
async def sql_schema(db: str) -> dict:
    """Tables, columns, keys, and row counts of 'chinook' or 'northwind'."""
    return await asyncio.to_thread(_schema, db.strip().lower())
