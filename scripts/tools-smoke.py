#!/usr/bin/env python3
"""Smoke test for the tools service: call every function in peira_harness.tools, then optionally run a burst.

Usage:
  TOOLS_URL=http://localhost:8080 python scripts/tools-smoke.py              # every tool + negative checks
  TOOLS_URL=http://localhost:8080 python scripts/tools-smoke.py --burst 200 --concurrency 40
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import httpx  # noqa: E402

from peira_harness import tools as T  # noqa: E402

FAILS: list[str] = []
CALLED: set[str] = set()


def check(fn, kwargs: dict, expect_error: bool = False, pred=None, label: str = ""):
    CALLED.add(fn.__name__)
    t0 = time.time()
    try:
        r = fn(**kwargs)
    except Exception as e:  # noqa: BLE001
        FAILS.append(f"{fn.__name__}{label}: raised {e!r}")
        print(f"FAIL {fn.__name__}{label}: raised {e!r}")
        return None
    dt = time.time() - t0
    is_err = isinstance(r, dict) and "error" in r
    ok = is_err == expect_error and (pred is None or pred(r))
    short = json.dumps(r, default=str)[:160]
    print(f"{'ok  ' if ok else 'FAIL'} {fn.__name__}{label} {dt:5.2f}s {short}")
    if not ok:
        FAILS.append(f"{fn.__name__}{label}: {short}")
    return r


def smoke() -> None:
    cat = httpx.get(f"{T.TOOLS_URL}/v1/catalog", timeout=30).json()
    cust = cat["crm"]["customers_with_5plus_orders"][0]
    order = cat["commerce"]["recent_delivered_refundable"][0]
    sku = cat["commerce"]["sample_products"][1]["sku"]
    ticket = cat["support"]["sample_open_tickets"][0]
    emp = [e["employee_id"] for e in cat["hr"]["employees"] if e["status"] == "active"][:3]
    T.set_run_id(f"smoke-{uuid.uuid4().hex[:8]}")

    # Web and research
    check(T.web_search, {"query": "python asyncio gather vs wait"}, pred=lambda r: len(r["results"]) > 0)
    check(T.fetch_url, {"url": "https://example.com", "max_chars": 500}, pred=lambda r: "Example Domain" in r["content"] + r["title"])
    check(T.fetch_url, {"url": "http://169.254.169.254/metadata/instance"}, expect_error=True, label="[ssrf-imds]")
    check(T.fetch_url, {"url": "http://langfuse-web.experiments.svc.cluster.local:3000"}, expect_error=True, label="[ssrf-cluster]")
    check(T.fetch_url, {"url": "http://10.0.0.1/"}, expect_error=True, label="[ssrf-private]")
    check(T.wikipedia_search, {"query": "Ada Lovelace"}, pred=lambda r: len(r["results"]) > 0)
    check(T.wikipedia_page, {"title": "Ada Lovelace", "max_chars": 1000}, pred=lambda r: "Lovelace" in r["content"])
    check(T.wikipedia_page, {"title": "Zzqx nonexistent page 12345"}, expect_error=True, label="[missing]")
    check(T.arxiv_search, {"query": "retrieval augmented generation", "max_results": 3}, pred=lambda r: len(r["results"]) > 0)

    # Code and SQL
    check(T.python_exec, {"code": "import numpy as np, pandas as pd\nprint(np.arange(5).sum(), pd.DataFrame({'a':[1,2]}).a.mean())"},
          pred=lambda r: r["status"] == "ok" and "10 1.5" in r["stdout"])
    check(T.python_exec, {"code": "import os\nprint(sorted(os.environ))"}, pred=lambda r: "VALKEY_PASSWORD" not in r["stdout"], label="[env]")
    check(T.python_exec, {"code": "print(open('/proc/1/environ','rb').read()[:50])"}, pred=lambda r: r["status"] == "error", label="[proc-env]")
    check(T.python_exec, {"code": "import urllib.request\nprint(urllib.request.urlopen('http://169.254.169.254/metadata/instance', timeout=3).read()[:20])"},
          pred=lambda r: r["status"] == "error", label="[net]")
    check(T.python_exec, {"code": "while True: pass", "timeout_s": 2}, pred=lambda r: r["status"] == "timeout", label="[timeout]")
    check(T.python_exec, {"code": "x = bytearray(3 * 1024**3)"}, pred=lambda r: r["status"] == "error", label="[mem]")
    check(T.sql_schema, {"db": "chinook"}, pred=lambda r: any(t["name"] == "Track" for t in r["tables"]))
    check(T.sql_query, {"db": "chinook", "sql": "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 3"}, pred=lambda r: r["row_count"] == 3)
    check(T.sql_query, {"db": "northwind", "sql": 'SELECT COUNT(*) FROM "Order Details"'}, pred=lambda r: r["rows"][0][0] > 0)
    check(T.sql_query, {"db": "chinook", "sql": "DELETE FROM Artist"}, expect_error=True, label="[write]")
    check(T.sql_query, {"db": "chinook", "sql": "SELECT 1; DROP TABLE Artist"}, expect_error=True, label="[multi]")
    check(T.sql_query, {"db": "chinook", "sql": "WITH x AS (SELECT 1) INSERT INTO Artist(Name) SELECT 'x'"}, expect_error=True, label="[with-insert]")

    # Support
    check(T.crm_get_customer, {"customer_id": cust}, pred=lambda r: r["customer_id"] == cust)
    check(T.crm_get_customer, {"customer_id": "CUST-99999"}, expect_error=True, label="[missing]")
    check(T.crm_search_customers, {"query": "smith", "limit": 3}, pred=lambda r: r["total_matches"] > 0)
    check(T.tickets_search, {"status": "open", "limit": 3}, pred=lambda r: r["total_matches"] > 0)
    check(T.tickets_get, {"ticket_id": ticket}, pred=lambda r: r["ticket_id"] == ticket)
    check(T.tickets_update, {"ticket_id": ticket, "status": "pending", "note": "smoke test"}, pred=lambda r: r["confirmation_id"].startswith("TU-"))
    check(T.tickets_get, {"ticket_id": ticket}, pred=lambda r: r["status"] == "pending", label="[read-your-write]")
    check(T.kb_search, {"query": "factory reset"}, pred=lambda r: len(r["results"]) > 0)
    check(T.kb_get_article, {"article_id": "KB-1005"}, pred=lambda r: r["article_id"] == "KB-1005")

    # Commerce
    check(T.orders_list, {"customer_id": cust, "limit": 5}, pred=lambda r: r["total_matches"] >= 5)
    check(T.orders_get, {"order_id": order}, pred=lambda r: r["order_id"] == order)
    check(T.orders_refund, {"order_id": order, "amount_usd": 5.0, "reason": "smoke test"}, pred=lambda r: r["status"] in ("approved", "pending_approval"))
    check(T.orders_get, {"order_id": order}, pred=lambda r: r["refunded_usd"] >= 5.0, label="[read-your-write]")
    check(T.orders_refund, {"order_id": order, "amount_usd": 999999}, expect_error=True, label="[too-much]")
    check(T.products_search, {"category": "cameras", "limit": 3}, pred=lambda r: r["total_matches"] > 0)
    check(T.inventory_check, {"sku": sku}, pred=lambda r: "stock_by_warehouse" in r)

    # Travel
    fl = check(T.travel_search_flights, {"origin": "New York", "destination": "LHR", "date": "2026-12-03"}, pred=lambda r: len(r["flights"]) > 0)
    ht = check(T.travel_search_hotels, {"city": "London", "check_in": "2026-12-03", "check_out": "2026-12-06"}, pred=lambda r: len(r["hotels"]) > 0)
    if fl and "flights" in fl:
        offer = next((f["offer_id"] for f in fl["flights"] if f["seats_left"] > 0), None)
        if offer:
            check(T.travel_book, {"offer_id": offer, "traveler_name": "Smoke Test"}, pred=lambda r: r["confirmation_id"].startswith("BK-"))
    if ht and "hotels" in ht:
        hid = next((h["hotel_id"] for h in ht["hotels"] if h["available"]), None)
        if hid:
            check(T.travel_book, {"offer_id": hid, "traveler_name": "Smoke Test", "check_in": "2026-12-03", "check_out": "2026-12-06"},
                  pred=lambda r: r["type"] == "hotel", label="[hotel]")
    check(T.travel_search_flights, {"origin": "Atlantis", "destination": "LHR", "date": "2026-12-03"}, expect_error=True, label="[bad-city]")

    # Finance
    check(T.finance_quote, {"symbol": "LMNC"}, pred=lambda r: r["price"] > 0)
    check(T.finance_history, {"symbol": "AAPL", "start_date": "2026-01-01", "interval": "monthly"}, pred=lambda r: r["count"] >= 10)

    # HR
    check(T.hr_get_employee, {"employee_id": emp[0]}, pred=lambda r: r["employee_id"] == emp[0])
    check(T.hr_search_employees, {"department": "Support"}, pred=lambda r: r["total_matches"] > 0)
    check(T.hr_policy_lookup, {"query": "hotel cap"}, pred=lambda r: len(r["policies"]) > 0)

    # Productivity
    check(T.calendar_get_events, {"employee_id": emp[0], "date": "2026-11-18"}, pred=lambda r: "events" in r)
    slots = check(T.calendar_free_slots, {"attendees": emp, "date": "2026-11-18", "duration_minutes": 30}, pred=lambda r: "free_slots" in r)
    start = slots["free_slots"][0]["start"] if slots and slots.get("free_slots") else "2026-11-18T16:00"
    check(T.calendar_create_event, {"title": "Smoke sync", "attendees": emp, "start": start, "duration_minutes": 30},
          pred=lambda r: r["confirmation_id"].startswith("EVT-") and not r["conflicts"])
    check(T.email_send, {"to": "ops@lumenco.example", "subject": "smoke", "body": "hello"}, pred=lambda r: r["status"] == "queued_to_outbox")
    check(T.email_send, {"to": "not-an-email", "subject": "x", "body": "y"}, expect_error=True, label="[bad-addr]")

    missing = {f.__name__ for f in T.ALL_TOOLS} - CALLED
    if missing:
        FAILS.append(f"not exercised: {sorted(missing)}")
    print(f"\n{len(CALLED)}/{len(T.ALL_TOOLS)} tools exercised, {len(FAILS)} failures")
    for f in FAILS:
        print("  -", f)


WORDS = ["solar", "battery", "climate", "jazz", "volcano", "neural", "compiler", "marathon", "tea", "glacier", "satellite", "vaccine",
         "bridge", "quantum", "orchid", "railway", "bitcoin", "coral", "opera", "drone", "algebra", "desert", "viking", "robot"]


TOPICS = ["history", "economics", "recipe", "museum", "festival", "species", "engine", "language", "treaty", "telescope",
          "migration", "architecture", "tournament", "dynasty", "pigment", "earthquake", "protocol", "novel", "harbor", "vaccine trial"]


def burst(n: int, conc: int) -> None:
    cat = httpx.get(f"{T.TOOLS_URL}/v1/catalog", timeout=30).json()
    custs = [c["customer_id"] for c in cat["crm"]["sample_customers"]]
    orders = [o["order_id"] for o in cat["commerce"]["sample_orders"]]
    tickers = [t["symbol"] for t in cat["finance"]["tickers"]]
    tickets = cat["support"]["sample_open_tickets"]
    rnd = random.Random(7)
    qrnd = random.Random(uuid.uuid4().int)  # fresh query mix per run, so the burst does not just hit a warm cache
    mix = [
        ("web_search", 25, lambda i: (T.web_search, {"query": " ".join(qrnd.sample(WORDS + TOPICS, 3))})),
        ("wikipedia_search", 8, lambda i: (T.wikipedia_search, {"query": f"{rnd.choice(WORDS)} {rnd.choice(WORDS)}"})),
        ("wikipedia_page", 5, lambda i: (T.wikipedia_page, {"title": rnd.choice(["Python (programming language)", "Kubernetes", "Volcano", "Coral reef", "Marathon"]), "max_chars": 2000})),
        ("fetch_url", 5, lambda i: (T.fetch_url, {"url": rnd.choice(["https://example.com", "https://www.python.org", "https://en.wikipedia.org/wiki/Glacier", "https://httpbin.org/html"]), "max_chars": 2000})),
        ("arxiv_search", 2, lambda i: (T.arxiv_search, {"query": f"{rnd.choice(['graph neural network', 'diffusion model', 'reinforcement learning', 'speculative decoding'])}", "max_results": 3})),
        ("python_exec", 10, lambda i: (T.python_exec, {"code": f"import numpy as np\nprint(np.random.default_rng({i}).normal(size=1000).mean())"})),
        ("sql_query", 10, lambda i: (T.sql_query, {"db": rnd.choice(["chinook", "northwind"]), "sql": "SELECT name FROM sqlite_master WHERE type='table' LIMIT 5"})),
        ("crm_get_customer", 8, lambda i: (T.crm_get_customer, {"customer_id": rnd.choice(custs)})),
        ("orders_get", 8, lambda i: (T.orders_get, {"order_id": rnd.choice(orders)})),
        ("tickets_search", 5, lambda i: (T.tickets_search, {"query": rnd.choice(["refund", "wifi", "login", "firmware"])})),
        ("kb_search", 5, lambda i: (T.kb_search, {"query": rnd.choice(["reset", "matter", "invoice", "camera privacy"])})),
        ("finance_quote", 5, lambda i: (T.finance_quote, {"symbol": rnd.choice(tickers)})),
        ("tickets_update", 3, lambda i: (T.tickets_update, {"ticket_id": rnd.choice(tickets), "note": f"burst note {i}"})),
        ("travel_search_flights", 4, lambda i: (T.travel_search_flights, {"origin": "JFK", "destination": rnd.choice(["LHR", "CDG", "NRT", "SFO"]), "date": "2026-12-10"})),
    ]
    names, weights, makers = zip(*mix)
    jobs = []
    for i in range(n):
        k = rnd.choices(range(len(mix)), weights)[0]
        jobs.append((names[k],) + makers[k](i))

    def run(job):
        name, fn, kw = job
        T.set_run_id(f"burst-{uuid.uuid4().hex[:8]}")  # contextvars do not cross into pool threads; set it per job
        t0 = time.time()
        try:
            r = fn(**kw)
            err = r.get("error") if isinstance(r, dict) else None
            src = r.get("source") if isinstance(r, dict) else None
            if name == "web_search" and not err and not r.get("results"):
                err = "no results"
            return name, time.time() - t0, ("tool_error: " + err[:80]) if err else None, src
        except Exception as e:  # noqa: BLE001
            return name, time.time() - t0, f"raised: {e!r}"[:120], None

    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(run, jobs))
    wall = time.time() - t0
    lat = sorted(r[1] for r in res)
    errs = [r for r in res if r[2]]
    q = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))]  # noqa: E731
    print(f"\nburst: {n} calls, concurrency {conc}, wall {wall:.1f}s")
    print(f"latency p50 {q(0.5):.2f}s  p95 {q(0.95):.2f}s  max {lat[-1]:.2f}s")
    print(f"errors {len(errs)}/{n} = {100 * len(errs) / n:.1f}%")
    by = {}
    for name, dt, err, src in res:
        b = by.setdefault(name, {"n": 0, "err": 0, "lat": [], "src": {}})
        b["n"] += 1
        b["err"] += bool(err)
        b["lat"].append(dt)
        if src:
            b["src"][src] = b["src"].get(src, 0) + 1
    for name, b in sorted(by.items()):
        print(f"  {name:22s} n={b['n']:3d} err={b['err']:2d} p50={statistics.median(b['lat']):.2f}s max={max(b['lat']):.2f}s {b['src'] or ''}")
    for e in errs[:15]:
        print("  error:", e[0], e[2])
    ws = by.get("web_search")
    if ws:
        print(f"web_search hard-failure rate: {100 * ws['err'] / ws['n']:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--skip-smoke", action="store_true")
    a = ap.parse_args()
    print("TOOLS_URL =", T.TOOLS_URL)
    if not a.skip_smoke:
        smoke()
    if a.burst:
        burst(a.burst, a.concurrency)
    sys.exit(1 if FAILS else 0)
