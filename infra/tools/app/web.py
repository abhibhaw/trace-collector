"""Web tools: web_search (ddgs with fallbacks), fetch_url, Wikipedia, arXiv."""

from __future__ import annotations

import asyncio
import html as htmllib
import io
import ipaddress
import logging
import os
import random
import re
import socket
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlsplit

import httpx

import store
from registry import ToolError, tool

log = logging.getLogger("tools.web")
UA = "peira-trace-collector/0.1 (research experiment; https://peira.tech)"  # Wikimedia rejects UAs without a URL or contact
BROWSER_UA = "Mozilla/5.0 (compatible; peira-trace-collector/0.1; research experiment; +https://peira.tech)"
MAX_FETCH_BYTES = 5 * 1024 * 1024
CACHE_TTL_SEARCH = 7 * 86400
CACHE_TTL_FETCH = 86400

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(headers={"User-Agent": UA}, timeout=httpx.Timeout(15, connect=5),
                                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))
    return _client


async def _get(url: str, params: dict | None = None, attempts: int = 3, **kw) -> httpx.Response:
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = await client().get(url, params=params, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
            await asyncio.sleep((2 ** i) * 0.5 + random.uniform(0, 0.5))
    raise ToolError(f"upstream unavailable after {attempts} attempts: {last}")


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


# --- Wikipedia --------------------------------------------------------------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"


async def _wiki_api(params: dict) -> dict:
    wait = await store.reserve_slot("wikipedia", 0.05, 15)
    if wait is None:
        raise ToolError("Wikipedia is busy; try again in a few seconds.")
    if wait:
        await asyncio.sleep(wait)
    r = await _get(WIKI_API, params={**params, "format": "json", "formatversion": 2})
    if r.status_code != 200:
        raise ToolError(f"Wikipedia returned HTTP {r.status_code}")
    return r.json()


def _strip_tags(s: str) -> str:
    return htmllib.unescape(re.sub(r"<[^>]+>", "", s or ""))


async def _wiki_search_raw(query: str, limit: int) -> list[dict]:
    async def produce():
        j = await _wiki_api({"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
                             "srprop": "snippet|wordcount|timestamp"})
        return [{"title": h["title"], "url": "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
                 "snippet": _strip_tags(h.get("snippet", "")), "wordcount": h.get("wordcount"),
                 "last_edited": h.get("timestamp")} for h in j.get("query", {}).get("search", [])]
    v, _ = await store.cached(store.key("wiki_search", _norm(query), limit), CACHE_TTL_SEARCH, produce)
    return v


@tool("research")
async def wikipedia_search(query: str, max_results: int = 5) -> dict:
    """Search English Wikipedia; returns titles, URLs, and snippets."""
    if not query.strip():
        raise ToolError("query must not be empty")
    res = await _wiki_search_raw(query, max(1, min(max_results, 20)))
    return {"query": query, "results": res, **({} if res else {"note": "No matching articles. Try different keywords."})}


@tool("research")
async def wikipedia_page(title: str, max_chars: int = 8000) -> dict:
    """Plain-text content of a Wikipedia article (follows redirects)."""
    if not title.strip():
        raise ToolError("title must not be empty")
    max_chars = max(200, min(max_chars, 50000))

    async def produce():
        j = await _wiki_api({"action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
                             "inprop": "url", "titles": title.strip()})
        pages = j.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return {"missing": True}
        p = pages[0]
        return {"title": p["title"], "url": p.get("fullurl"), "text": p.get("extract", "")[:50000],
                "length": len(p.get("extract", ""))}
    v, _ = await store.cached(store.key("wiki_page", _norm(title)), CACHE_TTL_SEARCH, produce)
    if v.get("missing"):
        hits = await _wiki_search_raw(title, 5)
        raise ToolError(f"No Wikipedia article titled '{title}'. Close matches: " + ", ".join(h["title"] for h in hits) if hits else f"No Wikipedia article titled '{title}'.")
    text = v["text"]
    return {"title": v["title"], "url": v["url"], "content": text[:max_chars], "truncated": len(text) > max_chars,
            "total_chars": v["length"]}


# --- arXiv ------------------------------------------------------------------------------------
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@tool("research")
async def arxiv_search(query: str, max_results: int = 5, sort_by: str = "relevance") -> dict:
    """Search arXiv papers; returns id, title, authors, abstract, dates, and links."""
    if not query.strip():
        raise ToolError("query must not be empty")
    if sort_by not in ("relevance", "submittedDate", "lastUpdatedDate"):
        raise ToolError("sort_by must be one of: relevance, submittedDate, lastUpdatedDate")
    max_results = max(1, min(max_results, 25))
    q = query.strip()
    if not re.search(r"\b(ti|au|abs|cat|all|id):", q):
        q = "all:" + " AND all:".join(w for w in re.findall(r"[\w\-\.]+", q)[:12])

    async def produce():
        wait = await store.reserve_slot("arxiv", 3.1, 60)
        if wait is None:
            raise ToolError("arXiv rate limit queue is full; try again in a minute.")
        if wait:
            await asyncio.sleep(wait)
        r = await _get("https://export.arxiv.org/api/query", params={"search_query": q, "max_results": max_results,
                                                                     "sortBy": sort_by, "sortOrder": "descending"}, timeout=30)
        if r.status_code != 200:
            raise ToolError(f"arXiv returned HTTP {r.status_code}")
        root = ET.fromstring(r.text)
        out = []
        for e in root.findall("a:entry", ATOM):
            aid = (e.findtext("a:id", "", ATOM) or "").rsplit("/abs/", 1)[-1]
            out.append({
                "arxiv_id": aid,
                "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ATOM)).strip(),
                "authors": [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)],
                "summary": re.sub(r"\s+", " ", e.findtext("a:summary", "", ATOM)).strip(),
                "published": e.findtext("a:published", "", ATOM)[:10],
                "updated": e.findtext("a:updated", "", ATOM)[:10],
                "primary_category": (e.find("arxiv:primary_category", ATOM).get("term") if e.find("arxiv:primary_category", ATOM) is not None else None),
                "abs_url": f"https://arxiv.org/abs/{aid}",
                "pdf_url": f"https://arxiv.org/pdf/{aid}",
            })
        return out
    v, hit = await store.cached(store.key("arxiv", _norm(q), max_results, sort_by), CACHE_TTL_SEARCH, produce)
    return {"query": q, "results": v, "cached": hit, **({} if v else {"note": "No matching papers. Try broader keywords."})}


# --- web_search -------------------------------------------------------------------------------
DDGS_INTERVAL_S = float(os.environ.get("DDGS_INTERVAL_S", "0.5"))  # global spacing per stage, across all replicas
DDGS_MAX_WAIT_S = float(os.environ.get("DDGS_MAX_WAIT_S", "4"))  # longer queue -> go straight to the next stage
DDGS_STAGE_TIMEOUT_S = 8.0
WEB_SEARCH_DDGS_BUDGET_S = 6.0  # if stage 1 took longer than this, skip stage 2 and fall back to Wikipedia
# ddgs is blocking; give it its own pool so it cannot starve fetch_url / SQL threads in the default executor.
_DDGS_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="ddgs")
# duckduckgo.com is unreachable from the gepton-base egress (TCP connect timeout, verified 2026-11-12), so it is not in the
# default chain; it only added 6-9 s per call. Override with DDGS_STAGES="name=backends;name2=backends2".
DDG_STAGES = [tuple(x.split("=", 1)) for x in os.environ.get("DDGS_STAGES", "ddgs=yahoo,brave,startpage,mojeek").split(";") if "=" in x]


def _ddgs_sync(query: str, n: int, backend: str) -> list[dict]:
    from ddgs import DDGS
    return DDGS(timeout=8).text(query, max_results=n, backend=backend)


async def _ddgs(query: str, n: int, backend: str) -> list[dict]:
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
    try:
        loop = asyncio.get_running_loop()
        res = await asyncio.wait_for(loop.run_in_executor(_DDGS_POOL, _ddgs_sync, query, n, backend), timeout=DDGS_STAGE_TIMEOUT_S)
    except RatelimitException:
        await store.flag_set(f"ddgs-cooldown:{backend}", 45)
        return []
    except (DDGSException, TimeoutException, asyncio.TimeoutError) as e:
        log.info("ddgs %s failed: %s", backend, e)
        return []
    except Exception as e:  # noqa: BLE001
        log.warning("ddgs %s error: %r", backend, e)
        return []
    return [{"title": x.get("title", ""), "url": x.get("href", ""), "snippet": x.get("body", "")} for x in res if x.get("href")]


_STOP = set("the and for with from what who whom whose which when where why how are was were is be been being this that these those "
             "into onto about over under than then there their they them you your our does did do can could would should "
             "will shall may might must not but all any some most more less very best top list facts info information".split())


@tool("web")
async def web_search(query: str, max_results: int = 5) -> dict:
    """Web search with cache and fallbacks; every result carries `source`."""
    q = _norm(query)
    if not q:
        raise ToolError("query must not be empty")
    n = max(1, min(max_results, 10))

    async def produce():
        attempts = []
        t0 = asyncio.get_running_loop().time()
        for stage, backend in DDG_STAGES:
            if asyncio.get_running_loop().time() - t0 > WEB_SEARCH_DDGS_BUDGET_S:
                attempts.append(f"{stage}:skipped-budget")
                continue
            if await store.flag_on(f"ddgs-cooldown:{backend}"):
                attempts.append(f"{stage}:cooldown")
                continue
            wait = await store.reserve_slot(f"ddgs:{stage}", DDGS_INTERVAL_S, DDGS_MAX_WAIT_S)
            if wait is None:
                attempts.append(f"{stage}:busy")
                continue
            await asyncio.sleep(wait + random.uniform(0, 0.2))
            res = await _ddgs(query.strip(), n, backend)
            if res:
                return {"query": query.strip(), "source": stage, "results": [{**x, "source": stage} for x in res[:n]]}
            attempts.append(f"{stage}:empty")
        hits = await _wiki_search_raw(query.strip(), n)
        if not hits:
            # Relax the query: keep the most specific words and OR them, so the fallback still returns something useful.
            words = [w for w in re.findall(r"[^\W\d_]{3,}", query.lower()) if w not in _STOP]
            words = sorted(set(words), key=lambda w: (-len(w), w))[:4]
            if words:
                hits = await _wiki_search_raw(" OR ".join(words), n)
        if hits:
            return {"query": query.strip(), "source": "wikipedia-fallback", "fallback_reason": ", ".join(attempts),
                    "results": [{"title": h["title"], "url": h["url"], "snippet": h["snippet"], "source": "wikipedia-fallback"} for h in hits]}
        return {"query": query.strip(), "source": "none", "fallback_reason": ", ".join(attempts), "results": [],
                "note": "No results from any backend. Try a shorter or different query."}

    # Only cache real results; empty answers expire quickly so a later retry can succeed.
    k = store.key("web_search", q, n)
    v = await store.cache_get(k)
    if v is not None:
        return {**v, "cached": True}
    v, hit = await store.cached(k + ":sf", 1, produce)
    ttl = CACHE_TTL_SEARCH if v["source"] in {st for st, _ in DDG_STAGES} else (3600 if v["results"] else 60)
    await store.cache_set(k, v, ttl)
    return {**v, "cached": hit}


# --- fetch_url ------------------------------------------------------------------------------
_EXTRA_BLOCKED = [ipaddress.ip_network(n) for n in ("168.63.129.16/32", "100.64.0.0/10", "169.254.0.0/16", "0.0.0.0/8")]


def _ip_blocked(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return (not a.is_global) or a.is_multicast or any(a in n for n in _EXTRA_BLOCKED if a.version == n.version)


async def _check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError("Only http and https URLs are allowed.")
    host = parts.hostname
    if not host:
        raise ToolError("URL has no host.")
    if host.endswith((".local", ".internal", ".cluster.local", ".svc")) or host in ("localhost", "metadata", "metadata.google.internal"):
        raise ToolError("This host is not allowed (internal address).")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ToolError(f"Could not resolve host '{host}'.") from None
    for info in infos:
        if _ip_blocked(info[4][0]):
            raise ToolError("This host resolves to a private or internal address and is blocked.")


def _extract_html(body: str, url: str) -> tuple[str, str]:
    import trafilatura
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
    if m:
        title = htmllib.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    text = trafilatura.extract(body, url=url, include_tables=True, include_comments=False, favor_recall=True) or ""
    if not text:
        text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", body)
        text = htmllib.unescape(re.sub(r"<[^>]+>", " ", text))
        text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()
    return title, text


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    rd = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(rd.pages[:30]):
        parts.append(page.extract_text() or "")
        if sum(len(p) for p in parts) > 100000:
            break
    return "\n\n".join(parts)


async def _fetch(url: str) -> dict:
    cur = url
    for _ in range(6):
        await _check_url(cur)
        try:
            async with client().stream("GET", cur, follow_redirects=False, headers={"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5"}, timeout=httpx.Timeout(20, connect=6)) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    cur = urljoin(cur, resp.headers["location"])
                    continue
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > MAX_FETCH_BYTES:
                        break
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                status = resp.status_code
                enc = resp.encoding or "utf-8"
        except httpx.TransportError as e:
            raise ToolError(f"Could not fetch URL: {type(e).__name__}: {e}") from None
        data = bytes(buf)
        title, text = "", ""
        if ctype == "application/pdf" or data[:5] == b"%PDF-":
            text = await asyncio.to_thread(_extract_pdf, data)
            ctype = "application/pdf"
        elif "html" in ctype or (not ctype and b"<html" in data[:2000].lower()):
            title, text = await asyncio.to_thread(_extract_html, data.decode(enc, "replace"), cur)
        elif ctype.startswith("text/") or ctype in ("application/json", "application/xml", "application/rss+xml", "application/atom+xml"):
            text = data.decode(enc, "replace")
        else:
            raise ToolError(f"Unsupported content type '{ctype}' (only HTML, text, JSON, XML, and PDF are supported).")
        return {"url": url, "final_url": cur, "status": status, "content_type": ctype, "title": title,
                "text": text[:200000], "truncated_download": len(data) > MAX_FETCH_BYTES}
    raise ToolError("Too many redirects.")


@tool("web")
async def fetch_url(url: str, max_chars: int = 8000) -> dict:
    """Fetch a public web page or PDF and return its readable text."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    max_chars = max(200, min(max_chars, 50000))
    await _check_url(url)
    k = store.key("fetch2", url)
    v, hit = await store.cached(k, CACHE_TTL_FETCH, lambda: _fetch(url))
    if v["status"] >= 400:
        if not hit:
            await store.cache_set(k, v, 600)  # errors may be transient; keep them only briefly
        raise ToolError(f"HTTP {v['status']} for {v['final_url']}" + (f": {v['text'][:300]}" if v["text"] else ""))
    text = v["text"]
    return {"url": v["final_url"], "title": v["title"], "content_type": v["content_type"], "content": text[:max_chars],
            "truncated": len(text) > max_chars, "total_chars": len(text), "cached": hit}
