"""Tools service: FastAPI app exposing every tool at POST /v1/tools/{name}.

Response contract:
  200 {"ok": true, "result": ...}           success
  200 {"ok": false, "error": "..."}          readable tool error (not found, invalid input, upstream unavailable)
  404 / 422 {"ok": false, "error": "..."}    unknown tool / bad arguments
  500 {"ok": false, "error": "..."}          server bug (the client retries, then raises)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import business  # noqa: F401  (registers tools)
import sandbox
import sqltools
import store
import web  # noqa: F401
from catalog import build_catalog
from mockdata import data
from registry import TOOLS, ToolError

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("tools")
app = FastAPI(title="peira tools service", version="0.1")
STATS: Counter = Counter()
LAT: dict[str, deque] = {}
MAX_BODY = 256 * 1024


@app.on_event("startup")
async def startup() -> None:
    t = time.time()
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=32, thread_name_prefix="tools"))
    data()
    await store.init()
    await sandbox.probe()
    log.info("startup done in %.1fs; %d tools", time.time() - t, len(TOOLS))


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/ready")
async def ready() -> JSONResponse:
    checks = {"data": data() is not None, "sqlite": sqltools.ready(), "valkey": await store.ping()}
    ok = checks["data"] and checks["sqlite"]  # valkey down degrades to local cache; still serve
    return JSONResponse({"ok": ok, "checks": checks}, status_code=200 if ok else 503)


@app.get("/v1/tools")
async def list_tools() -> dict:
    return {"tools": [{"name": n, "group": t["group"], "write": t["write"], "params": t["params"], "doc": t["doc"]} for n, t in sorted(TOOLS.items())]}


@app.get("/v1/catalog")
async def catalog() -> dict:
    return build_catalog()


@app.get("/v1/stats")
async def stats() -> dict:
    lat = {}
    for k, d in LAT.items():
        s = sorted(d)
        if s:
            lat[k] = {"n": len(s), "p50_ms": round(s[len(s) // 2] * 1000), "p95_ms": round(s[int(len(s) * 0.95) - 1 if len(s) > 1 else 0] * 1000)}
    return {"pid": os.getpid(), "counters": dict(STATS), "latency_recent": lat, "journal": await store.journal_counts()}


@app.post("/v1/tools/{name}")
async def call_tool(name: str, request: Request) -> JSONResponse:
    t = TOOLS.get(name)
    if not t:
        return JSONResponse({"ok": False, "error": f"Unknown tool '{name}'."}, status_code=404)
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return JSONResponse({"ok": False, "error": "Request body too large."}, status_code=413)
    try:
        args = await request.json() if raw else {}
    except ValueError:
        return JSONResponse({"ok": False, "error": "Body must be a JSON object of arguments."}, status_code=422)
    if not isinstance(args, dict):
        return JSONResponse({"ok": False, "error": "Body must be a JSON object of arguments."}, status_code=422)
    store.RUN_ID.set(request.headers.get("x-peira-run-id", "")[:128])
    store.IDEM_KEY.set(request.headers.get("idempotency-key", "")[:128])
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(t["fn"](**args), timeout=90)
        STATS[f"{name}:ok"] += 1
        body, code = {"ok": True, "result": result}, 200
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(str(x) for x in er['loc']) or 'args'}: {er['msg']}" for er in e.errors())
        STATS[f"{name}:invalid"] += 1
        body, code = {"ok": False, "error": f"Invalid arguments for {name}: {msgs}"}, 422
    except ToolError as e:
        STATS[f"{name}:tool_error"] += 1
        body, code = {"ok": False, "error": str(e)}, 200
    except asyncio.TimeoutError:
        STATS[f"{name}:timeout"] += 1
        body, code = {"ok": False, "error": f"{name} timed out on the server."}, 504
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s failed", name)
        STATS[f"{name}:exception"] += 1
        body, code = {"ok": False, "error": f"Internal error in {name}: {type(e).__name__}"}, 500
    LAT.setdefault(name, deque(maxlen=500)).append(time.monotonic() - t0)
    return JSONResponse(body, status_code=code)
