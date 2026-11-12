"""Shared state in valkey: cache, global rate limiters, write journal, per-run overlays.

Every helper degrades to an in-process fallback when valkey is unreachable, so the service keeps answering.
Cache keys carry a TTL; journal keys do not (valkey runs with volatile-lru, so only cache entries are evicted).
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable

import redis.asyncio as aioredis

log = logging.getLogger("tools.store")

RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
IDEM_KEY: contextvars.ContextVar[str] = contextvars.ContextVar("idem_key", default="")

_r: aioredis.Redis | None = None
_local_cache: dict[str, tuple[float, Any]] = {}
_local_next: dict[str, float] = {}
_local_journal: list[dict] = []
_local_kv: dict[str, Any] = {}
_inflight: dict[str, asyncio.Future] = {}

_LIMIT_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local interval = tonumber(ARGV[1])
local maxwait = tonumber(ARGV[2])
local nxt = tonumber(redis.call('GET', KEYS[1]) or '0')
local start = math.max(now, nxt)
local wait = start - now
if wait > maxwait then return -1 end
redis.call('SET', KEYS[1], tostring(start + interval), 'PX', wait + interval + 1000)
return wait
"""
_limit_script = None


def redis() -> aioredis.Redis | None:
    return _r


async def init() -> None:
    global _r, _limit_script
    url = os.environ.get("VALKEY_URL", "redis://tools-valkey:6379/0")
    try:
        r = aioredis.from_url(url, password=os.environ.get("VALKEY_PASSWORD") or None, socket_timeout=2,
                              socket_connect_timeout=2, health_check_interval=15, max_connections=200, decode_responses=True)
        await r.ping()
        _r = r
        _limit_script = r.register_script(_LIMIT_LUA)
        log.info("valkey connected")
    except Exception as e:  # noqa: BLE001
        log.warning("valkey unavailable, using in-process fallback: %s", e)
        _r = None


async def ping() -> bool:
    if _r is None:
        await init()
    if _r is None:
        return False
    try:
        return bool(await _r.ping())
    except Exception:  # noqa: BLE001
        return False


def key(tool: str, *parts: Any) -> str:
    norm = json.dumps(parts, sort_keys=True, default=str).lower()
    return f"c:{tool}:{hashlib.sha256(norm.encode()).hexdigest()[:32]}"


async def cache_get(k: str) -> Any:
    if _r is not None:
        try:
            v = await _r.get(k)
            return json.loads(v) if v is not None else None
        except Exception as e:  # noqa: BLE001
            log.warning("cache_get failed: %s", e)
    hit = _local_cache.get(k)
    if hit and hit[0] > time.time():
        return hit[1]
    return None


async def cache_set(k: str, v: Any, ttl: int) -> None:
    if _r is not None:
        try:
            await _r.set(k, json.dumps(v, default=str), ex=ttl)
            return
        except Exception as e:  # noqa: BLE001
            log.warning("cache_set failed: %s", e)
    if len(_local_cache) > 5000:
        _local_cache.clear()
    _local_cache[k] = (time.time() + ttl, v)


async def cached(k: str, ttl: int, producer: Callable[[], Awaitable[Any]]) -> tuple[Any, bool]:
    """Return (value, cache_hit). Concurrent identical calls in this process share one producer run."""
    v = await cache_get(k)
    if v is not None:
        return v, True
    fut = _inflight.get(k)
    if fut is not None:
        return await asyncio.shield(fut), True
    fut = asyncio.get_running_loop().create_future()
    _inflight[k] = fut
    try:
        v = await producer()
        if v is not None:
            await cache_set(k, v, ttl)
        fut.set_result(v)
        return v, False
    except BaseException as e:
        fut.set_exception(e)
        fut.exception()  # mark retrieved
        raise
    finally:
        _inflight.pop(k, None)


async def reserve_slot(name: str, interval_s: float, max_wait_s: float) -> float | None:
    """Global spacing limiter. Returns seconds to wait before calling, or None if the wait would exceed max_wait_s."""
    interval_ms, max_ms = int(interval_s * 1000), int(max_wait_s * 1000)
    if _limit_script is not None and _r is not None:
        try:
            w = int(await _limit_script(keys=[f"rl:{name}"], args=[interval_ms, max_ms]))
            return None if w < 0 else w / 1000
        except Exception as e:  # noqa: BLE001
            log.warning("limiter failed, local fallback: %s", e)
    now = time.time() * 1000
    start = max(now, _local_next.get(name, 0))
    if start - now > max_ms:
        return None
    _local_next[name] = start + interval_ms
    return (start - now) / 1000


async def flag_set(name: str, ttl_s: int) -> None:
    if _r is not None:
        try:
            await _r.set(f"flag:{name}", "1", ex=ttl_s)
            return
        except Exception:  # noqa: BLE001
            pass
    _local_kv[f"flag:{name}"] = time.time() + ttl_s


async def flag_on(name: str) -> bool:
    if _r is not None:
        try:
            return bool(await _r.exists(f"flag:{name}"))
        except Exception:  # noqa: BLE001
            pass
    return _local_kv.get(f"flag:{name}", 0) > time.time()


async def idem_get() -> Any:
    k = IDEM_KEY.get()
    if not k:
        return None
    if _r is not None:
        try:
            v = await _r.get(f"idem:{k}")
            return json.loads(v) if v else None
        except Exception:  # noqa: BLE001
            return None
    return _local_kv.get(f"idem:{k}")


async def journal(kind: str, entry: dict) -> None:
    """Append a write to the journal and remember its result under the idempotency key."""
    entry = {"kind": kind, "run_id": RUN_ID.get() or None, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    k = IDEM_KEY.get()
    if _r is not None:
        try:
            pipe = _r.pipeline()
            pipe.rpush(f"journal:{kind}", json.dumps(entry, default=str))
            if k:
                pipe.set(f"idem:{k}", json.dumps(entry["result"], default=str), ex=86400 * 3)
            await pipe.execute()
            return
        except Exception as e:  # noqa: BLE001
            log.warning("journal write failed: %s", e)
    _local_journal.append(entry)
    if k:
        _local_kv[f"idem:{k}"] = entry["result"]


async def overlay_put(kind: str, entity: str, value: dict) -> None:
    """Per-run state so a later read in the same run sees its own writes. No run id -> nothing stored."""
    run = RUN_ID.get()
    if not run:
        return
    if _r is not None:
        try:
            k = f"ov:{run}:{kind}"
            cur = await _r.hget(k, entity)
            lst = json.loads(cur) if cur else []
            lst.append(value)
            await _r.hset(k, entity, json.dumps(lst, default=str))
            await _r.expire(k, 86400 * 2)
            return
        except Exception:  # noqa: BLE001
            pass
    _local_kv.setdefault(f"ov:{run}:{kind}", {}).setdefault(entity, []).append(value)


async def overlay_get(kind: str, entity: str | None = None) -> Any:
    run = RUN_ID.get()
    if not run:
        return [] if entity else {}
    if _r is not None:
        try:
            k = f"ov:{run}:{kind}"
            if entity:
                v = await _r.hget(k, entity)
                return json.loads(v) if v else []
            return {e: json.loads(v) for e, v in (await _r.hgetall(k)).items()}
        except Exception:  # noqa: BLE001
            pass
    d = _local_kv.get(f"ov:{run}:{kind}", {})
    return d.get(entity, []) if entity else d


async def journal_counts() -> dict:
    if _r is None:
        return {"local": len(_local_journal)}
    try:
        out = {}
        for kind in ("refund", "ticket_update", "booking", "calendar_event", "email"):
            out[kind] = await _r.llen(f"journal:{kind}")
        return out
    except Exception:  # noqa: BLE001
        return {}
