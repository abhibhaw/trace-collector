"""Shared agent tools backed by the in-cluster tools service.

Every function is a plain typed Python function with a Google-style docstring, so each framework can wrap it as a tool.
Every function returns a JSON-serialisable dict.

Error convention:
- A normal failure (not found, invalid input, upstream site unavailable) returns ``{"error": "<readable message>"}``.
  The model can read it and recover. Nothing is raised.
- A transport failure (service unreachable, 5xx) is retried with backoff. After the last retry it raises ``ToolsServiceError``.

The "today" of the mock business world is 2026-11-13. Calendar times are America/Chicago.

Optional: call ``set_run_id(...)`` per task. Then reads in that run see the run's own writes (a refund shows on the
order, a ticket update shows on the ticket). Runs never see each other's writes.
The run id is a contextvar: call ``set_run_id`` in the same context (thread or task) where the tools execute.
Thread pools do not copy contextvars, so a framework that runs tools in worker threads loses the id; writes are then still
journaled but not visible to later reads. Env ``PEIRA_RUN_ID`` is a process-wide fallback; use it only with one run per process.

Only the standard library and httpx are used.
"""

import contextvars
import os
import random
import threading
import time
import uuid
from typing import Any

import httpx

TOOLS_URL = os.environ.get("TOOLS_URL", "http://tools.experiments.svc.cluster.local:8080").rstrip("/")

_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("peira_tools_run_id", default="")
_client: httpx.Client | None = None
_client_lock = threading.Lock()
_RETRY_STATUS = {429, 500, 502, 503, 504}
_ATTEMPTS = 4


class ToolsServiceError(RuntimeError):
    """The tools service could not be reached or failed repeatedly."""


def set_run_id(run_id: str) -> None:
    """Scope writes and read-your-writes to one run (task). Not a tool.

    Args:
        run_id: Any unique string for the current task run, for example "<agent_id>:<task_id>:<batch>".
    """
    _RUN_ID.set(run_id)


def _http() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
                                       timeout=httpx.Timeout(60, connect=10))
    return _client


def _call(name: str, args: dict[str, Any], timeout: float = 45, write: bool = False) -> dict:
    headers = {"User-Agent": "peira-harness-tools/0.1"}
    run_id = _RUN_ID.get() or os.environ.get("PEIRA_RUN_ID", "")
    if run_id:
        headers["X-Peira-Run-Id"] = run_id
    if write:
        headers["Idempotency-Key"] = uuid.uuid4().hex  # same key across retries: no double refunds or bookings
    last = ""
    for attempt in range(_ATTEMPTS):
        try:
            r = _http().post(f"{TOOLS_URL}/v1/tools/{name}", json=args, headers=headers, timeout=timeout)
        except httpx.TransportError as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code not in _RETRY_STATUS:
                try:
                    body = r.json()
                except ValueError:
                    body = {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
                if body.get("ok"):
                    return body["result"]
                return {"error": str(body.get("error") or f"HTTP {r.status_code}")}
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        if attempt < _ATTEMPTS - 1:
            time.sleep(min(8.0, 0.5 * 2 ** attempt) + random.uniform(0, 0.5))
    raise ToolsServiceError(f"tools service call {name} failed after {_ATTEMPTS} attempts: {last}")


# ---------------------------------------------------------------------------------------------
# Web and research
# ---------------------------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web and return the top results.

    Results come from a metasearch backend. If that is rate-limited, results fall back to Wikipedia search.
    Each result has a `source` field that says where it came from.

    Args:
        query: The search query, for example "kubernetes readiness probe best practices".
        max_results: Number of results to return (1-10).

    Returns:
        A dict with `query`, `source`, and `results` (a list of {title, url, snippet, source}),
        or {"error": message}.
    """
    return _call("web_search", {"query": query, "max_results": max_results}, timeout=75)


def fetch_url(url: str, max_chars: int = 8000) -> dict:
    """Download a public web page or PDF and return its main readable text.

    Private, internal, and cluster addresses are blocked.

    Args:
        url: The http(s) URL to fetch.
        max_chars: Maximum number of characters of text to return (200-50000).

    Returns:
        A dict with `url`, `title`, `content_type`, `content`, `truncated`, and `total_chars`, or {"error": message}.
    """
    return _call("fetch_url", {"url": url, "max_chars": max_chars}, timeout=75)


def wikipedia_search(query: str, max_results: int = 5) -> dict:
    """Search English Wikipedia for articles.

    Args:
        query: Keywords to search for.
        max_results: Number of results to return (1-20).

    Returns:
        A dict with `results`, a list of {title, url, snippet, wordcount, last_edited}, or {"error": message}.
    """
    return _call("wikipedia_search", {"query": query, "max_results": max_results})


def wikipedia_page(title: str, max_chars: int = 8000) -> dict:
    """Get the plain-text content of a Wikipedia article.

    Args:
        title: The exact article title, for example "Alan Turing". Redirects are followed.
        max_chars: Maximum number of characters to return (200-50000).

    Returns:
        A dict with `title`, `url`, `content`, `truncated`, and `total_chars`, or {"error": message}
        (the error lists close title matches when the article does not exist).
    """
    return _call("wikipedia_page", {"title": title, "max_chars": max_chars})


def arxiv_search(query: str, max_results: int = 5, sort_by: str = "relevance") -> dict:
    """Search arXiv for research papers.

    Calls are globally rate-limited to one every 3 seconds, so this tool can take a while under load.

    Args:
        query: Keywords, or an arXiv query such as "ti:transformer AND cat:cs.CL".
        max_results: Number of papers to return (1-25).
        sort_by: One of "relevance", "submittedDate", or "lastUpdatedDate".

    Returns:
        A dict with `results`, a list of {arxiv_id, title, authors, summary, published, updated, primary_category,
        abs_url, pdf_url}, or {"error": message}.
    """
    return _call("arxiv_search", {"query": query, "max_results": max_results, "sort_by": sort_by}, timeout=120)


# ---------------------------------------------------------------------------------------------
# Code and SQL
# ---------------------------------------------------------------------------------------------
def python_exec(code: str, timeout_s: int = 20) -> dict:
    """Run Python 3.12 code in an isolated sandbox and return its output.

    numpy, pandas, scipy, and sympy are installed. There is no network access and no state between calls.
    Print what you want to see; the value of the last expression is not returned.

    Args:
        code: A complete Python program.
        timeout_s: Time limit in seconds (1-60).

    Returns:
        A dict with `status` ("ok", "error", or "timeout"), `exit_code`, `stdout`, `stderr`, and `duration_s`,
        or {"error": message}.
    """
    return _call("python_exec", {"code": code, "timeout_s": timeout_s}, timeout=max(30, timeout_s + 30))


def sql_query(db: str, sql: str, max_rows: int = 100) -> dict:
    """Run one read-only SQL SELECT query on a sample SQLite database.

    Args:
        db: "chinook" (digital music store) or "northwind" (food trading company).
        sql: A single SELECT (or WITH ... SELECT) statement in SQLite syntax.
        max_rows: Maximum number of rows to return (1-200).

    Returns:
        A dict with `columns`, `rows` (a list of lists), `row_count`, and `truncated`, or {"error": message}.
    """
    return _call("sql_query", {"db": db, "sql": sql, "max_rows": max_rows})


def sql_schema(db: str) -> dict:
    """Describe the tables of a sample SQLite database.

    Args:
        db: "chinook" or "northwind".

    Returns:
        A dict with `tables`, each with name, row_count, columns (name, type, pk), and foreign_keys,
        or {"error": message}.
    """
    return _call("sql_schema", {"db": db})


# ---------------------------------------------------------------------------------------------
# Support: CRM, tickets, knowledge base
# ---------------------------------------------------------------------------------------------
def crm_get_customer(customer_id: str) -> dict:
    """Get a customer's full profile, with order and open-ticket summary.

    Args:
        customer_id: The customer id, for example "CUST-10042".

    Returns:
        A dict with contact details, tier, cloud_plan, status, order_count, lifetime_value_usd, recent_order_ids,
        and open_ticket_ids, or {"error": message}.
    """
    return _call("crm_get_customer", {"customer_id": customer_id})


def crm_search_customers(query: str = "", city: str = "", tier: str = "", status: str = "", limit: int = 10) -> dict:
    """Search customers by name, email, phone, company, or id.

    Args:
        query: Text to match (substring, case-insensitive). Empty matches all.
        city: Exact city name filter, for example "Chicago". Empty means any.
        tier: "bronze", "silver", "gold", or "platinum". Empty means any.
        status: "active" or "closed". Empty means any.
        limit: Maximum number of customers to return (1-50).

    Returns:
        A dict with `total_matches` and `customers` (id, name, email, company, city, country, tier, status),
        or {"error": message}.
    """
    return _call("crm_search_customers", {"query": query, "city": city, "tier": tier, "status": status, "limit": limit})


def tickets_search(query: str = "", status: str = "", customer_id: str = "", priority: str = "", category: str = "", limit: int = 10) -> dict:
    """Search support tickets, newest first.

    Args:
        query: Text to match in subject, body, ticket id, or order id. Empty matches all.
        status: "open", "pending", "resolved", or "closed". Empty means any.
        customer_id: Only tickets from this customer, for example "CUST-10042".
        priority: "low", "normal", "high", or "urgent". Empty means any.
        category: For example "shipping", "defect", "billing", "refund", "account", "firmware". Empty means any.
        limit: Maximum number of tickets to return (1-50).

    Returns:
        A dict with `total_matches` and `tickets` (summaries without the body), or {"error": message}.
    """
    return _call("tickets_search", {"query": query, "status": status, "customer_id": customer_id, "priority": priority,
                                    "category": category, "limit": limit})


def tickets_get(ticket_id: str) -> dict:
    """Get a support ticket, including the customer's full message.

    Args:
        ticket_id: The ticket id, for example "TCK-40017".

    Returns:
        A dict with customer_id, order_id, subject, body, priority, status, assignee, timestamps, and tags,
        or {"error": message}.
    """
    return _call("tickets_get", {"ticket_id": ticket_id})


def tickets_update(ticket_id: str, status: str = "", priority: str = "", assignee: str = "", note: str = "") -> dict:
    """Update a support ticket and/or add an internal note. Empty arguments are left unchanged.

    Args:
        ticket_id: The ticket id, for example "TCK-40017".
        status: New status: "open", "pending", "resolved", or "closed".
        priority: New priority: "low", "normal", "high", or "urgent".
        assignee: Employee id of the new assignee, for example "EMP-007".
        note: Internal note to add to the ticket.

    Returns:
        A dict with `confirmation_id` and the applied `changes`, or {"error": message}.
    """
    return _call("tickets_update", {"ticket_id": ticket_id, "status": status, "priority": priority, "assignee": assignee,
                                    "note": note}, write=True)


def kb_search(query: str, max_results: int = 5) -> dict:
    """Search the Lumen & Co product knowledge base (devices, app, billing, orders, security).

    Args:
        query: Keywords, for example "factory reset smart lock".
        max_results: Number of articles to return (1-20).

    Returns:
        A dict with `results`, a list of {article_id, title, category, updated, snippet, score}, or {"error": message}.
    """
    return _call("kb_search", {"query": query, "max_results": max_results})


def kb_get_article(article_id: str) -> dict:
    """Get the full text of a knowledge-base article.

    Args:
        article_id: The article id, for example "KB-1005".

    Returns:
        A dict with article_id, title, category, applies_to, updated, and body, or {"error": message}.
    """
    return _call("kb_get_article", {"article_id": article_id})


# ---------------------------------------------------------------------------------------------
# Commerce: orders, products, inventory
# ---------------------------------------------------------------------------------------------
def orders_list(customer_id: str = "", status: str = "", date_from: str = "", date_to: str = "", sku: str = "", limit: int = 20) -> dict:
    """List orders, newest first, with optional filters.

    Args:
        customer_id: Only this customer's orders, for example "CUST-10042".
        status: "processing", "shipped", "delivered", "cancelled", "returned", or "refunded". Empty means any.
        date_from: Earliest order date, ISO format "YYYY-MM-DD". Empty means no lower bound.
        date_to: Latest order date, ISO format "YYYY-MM-DD". Empty means no upper bound.
        sku: Only orders that contain this product SKU, for example "SKU-2010".
        limit: Maximum number of orders to return (1-100).

    Returns:
        A dict with `total_matches` and `orders` (order_id, customer_id, order_date, status, total_usd, item_count),
        or {"error": message}.
    """
    return _call("orders_list", {"customer_id": customer_id, "status": status, "date_from": date_from, "date_to": date_to,
                                 "sku": sku, "limit": limit})


def orders_get(order_id: str) -> dict:
    """Get full order details: items, totals, shipping, tracking, and refunds.

    Args:
        order_id: The order id, for example "ORD-300123".

    Returns:
        A dict with the order fields, or {"error": message}.
    """
    return _call("orders_get", {"order_id": order_id})


def orders_refund(order_id: str, amount_usd: float = 0.0, reason: str = "") -> dict:
    """Issue a refund for an order. Refunds over USD 500 are created as pending approval.

    Args:
        order_id: The order id, for example "ORD-300123".
        amount_usd: Amount to refund in USD. 0 refunds the full remaining balance.
        reason: Short reason for the refund.

    Returns:
        A dict with `confirmation_id`, `amount_usd`, `status` ("approved" or "pending_approval"), and `warnings`,
        or {"error": message}.
    """
    return _call("orders_refund", {"order_id": order_id, "amount_usd": amount_usd, "reason": reason}, write=True)


def products_search(query: str = "", category: str = "", max_price_usd: float = 0.0, include_discontinued: bool = False, limit: int = 10) -> dict:
    """Search the product catalog.

    Args:
        query: Text to match in product name, SKU, or category. Empty matches all.
        category: One of "hubs", "cameras", "sensors", "lighting", "locks", "climate", "plugs", "accessories".
        max_price_usd: Only products at or below this price. 0 means no limit.
        include_discontinued: Include discontinued products.
        limit: Maximum number of products to return (1-50).

    Returns:
        A dict with `total_matches` and `products` (sku, name, category, price_usd, discontinued, total_stock),
        or {"error": message}.
    """
    return _call("products_search", {"query": query, "category": category, "max_price_usd": max_price_usd,
                                     "include_discontinued": include_discontinued, "limit": limit})


def inventory_check(sku: str) -> dict:
    """Check stock for a product in each warehouse (WH-EAST, WH-WEST, WH-EU).

    Args:
        sku: The product SKU, for example "SKU-2010".

    Returns:
        A dict with `stock_by_warehouse`, `total_available`, `needs_reorder`, and product details, or {"error": message}.
    """
    return _call("inventory_check", {"sku": sku})


# ---------------------------------------------------------------------------------------------
# Travel
# ---------------------------------------------------------------------------------------------
def travel_search_flights(origin: str, destination: str, date: str, cabin: str = "economy", max_stops: int = 1, max_price_usd: float = 0.0) -> dict:
    """Search flights between two supported cities on a date.

    Args:
        origin: City name or airport code, for example "New York" or "JFK".
        destination: City name or airport code, for example "London" or "LHR".
        date: Departure date, ISO format "YYYY-MM-DD", on or after 2026-11-13.
        cabin: "economy", "premium_economy", "business", or "first".
        max_stops: Maximum number of stops (0 for nonstop only).
        max_price_usd: Only flights at or below this price. 0 means no limit.

    Returns:
        A dict with `flights`, each with offer_id, airline, flight_number, departure_local, duration_minutes, stops,
        price_usd, seats_left, and refundable, or {"error": message}.
    """
    return _call("travel_search_flights", {"origin": origin, "destination": destination, "date": date, "cabin": cabin,
                                           "max_stops": max_stops, "max_price_usd": max_price_usd})


def travel_search_hotels(city: str, check_in: str, check_out: str, guests: int = 1, max_price_per_night_usd: float = 0.0) -> dict:
    """Search hotels in a supported city.

    Args:
        city: City name or airport code, for example "Paris" or "CDG".
        check_in: Check-in date, ISO format "YYYY-MM-DD".
        check_out: Check-out date, ISO format "YYYY-MM-DD".
        guests: Number of guests (1-8).
        max_price_per_night_usd: Only hotels at or below this average nightly price. 0 means no limit.

    Returns:
        A dict with `hotels`, each with hotel_id, name, stars, guest_rating, avg_nightly_usd, total_usd, and available,
        or {"error": message}.
    """
    return _call("travel_search_hotels", {"city": city, "check_in": check_in, "check_out": check_out, "guests": guests,
                                          "max_price_per_night_usd": max_price_per_night_usd})


def travel_book(offer_id: str, traveler_name: str, check_in: str = "", check_out: str = "", traveler_email: str = "") -> dict:
    """Book a flight offer or a hotel stay.

    Args:
        offer_id: A flight offer_id from travel_search_flights (for example "FL-JFK-LHR-20261201-2Y")
            or a hotel_id from travel_search_hotels (for example "HT-LHR-03").
        traveler_name: Full name of the traveler.
        check_in: Hotel check-in date "YYYY-MM-DD". Required for hotels, ignored for flights.
        check_out: Hotel check-out date "YYYY-MM-DD". Required for hotels, ignored for flights.
        traveler_email: Optional email for the confirmation.

    Returns:
        A dict with `confirmation_id`, `status`, and booking details, or {"error": message}.
    """
    return _call("travel_book", {"offer_id": offer_id, "traveler_name": traveler_name, "check_in": check_in,
                                 "check_out": check_out, "traveler_email": traveler_email}, write=True)


# ---------------------------------------------------------------------------------------------
# Finance (synthetic market data)
# ---------------------------------------------------------------------------------------------
def finance_quote(symbol: str) -> dict:
    """Get the latest stock quote for a ticker. Prices are synthetic, as of 2026-11-13.

    Args:
        symbol: Ticker symbol, for example "LMNC", "FLUX", or "AAPL".

    Returns:
        A dict with price, change, change_pct, day range, volume, 52-week range, market_cap_usd, pe_ratio,
        and dividend_yield_pct, or {"error": message}.
    """
    return _call("finance_quote", {"symbol": symbol})


def finance_history(symbol: str, start_date: str = "", end_date: str = "", interval: str = "daily") -> dict:
    """Get historical price bars for a ticker. Data covers 2023-01-02 to 2026-11-13.

    Args:
        symbol: Ticker symbol, for example "LMNC".
        start_date: First date "YYYY-MM-DD". Empty means 30 days before end_date.
        end_date: Last date "YYYY-MM-DD". Empty means 2026-11-13.
        interval: "daily", "weekly", or "monthly".

    Returns:
        A dict with `bars` (date, open, high, low, close, volume; at most 400) and `count`, or {"error": message}.
    """
    return _call("finance_history", {"symbol": symbol, "start_date": start_date, "end_date": end_date, "interval": interval})


# ---------------------------------------------------------------------------------------------
# HR
# ---------------------------------------------------------------------------------------------
def hr_get_employee(employee_id: str) -> dict:
    """Get an employee record: title, department, manager, office, status, and PTO balance.

    Args:
        employee_id: The employee id, for example "EMP-014".

    Returns:
        A dict with the employee fields, manager_name, and direct_reports, or {"error": message}.
    """
    return _call("hr_get_employee", {"employee_id": employee_id})


def hr_search_employees(query: str = "", department: str = "", office: str = "", limit: int = 20) -> dict:
    """Find employees by name, title, email, or id.

    Args:
        query: Text to match (substring, case-insensitive). Empty matches all.
        department: For example "Engineering", "Support", "Sales", "HR". Empty means any.
        office: "Austin", "Berlin", "Toronto", or "Remote". Empty means any.
        limit: Maximum number of employees to return (1-80).

    Returns:
        A dict with `total_matches` and `employees` (id, name, title, department, office, email, status),
        or {"error": message}.
    """
    return _call("hr_search_employees", {"query": query, "department": department, "office": office, "limit": limit})


def hr_policy_lookup(query: str) -> dict:
    """Look up company policies by policy id or keywords and return their full text.

    Args:
        query: A policy id such as "pto" or "travel-2026", or keywords such as "hotel cap" or "parental leave".

    Returns:
        A dict with `policies`, up to 3 of {policy_id, title, effective, owner, text}, or {"error": message}.
    """
    return _call("hr_policy_lookup", {"query": query})


# ---------------------------------------------------------------------------------------------
# Productivity: calendar and email
# ---------------------------------------------------------------------------------------------
def calendar_get_events(employee_id: str, date: str) -> dict:
    """List an employee's meetings on a date. Times are America/Chicago.

    Args:
        employee_id: Employee id (for example "EMP-014"), work email, or exact full name.
        date: The date, ISO format "YYYY-MM-DD".

    Returns:
        A dict with `events`, a list of {start, end, title}, or {"error": message}.
    """
    return _call("calendar_get_events", {"employee_id": employee_id, "date": date})


def calendar_free_slots(attendees: list[str], date: str, duration_minutes: int = 30) -> dict:
    """Find times when all attendees are free on a date, within 09:00-17:00 America/Chicago.

    Args:
        attendees: Employee ids (for example ["EMP-014", "EMP-022"]), work emails, or exact full names.
        date: The date, ISO format "YYYY-MM-DD".
        duration_minutes: Meeting length in minutes (15-480).

    Returns:
        A dict with `free_slots`, a list of {start, end} on a 30-minute grid, or {"error": message}.
    """
    return _call("calendar_free_slots", {"attendees": list(attendees), "date": date, "duration_minutes": duration_minutes})


def calendar_create_event(title: str, attendees: list[str], start: str, duration_minutes: int = 30, description: str = "") -> dict:
    """Create a calendar event. The event is created even with conflicts; conflicts are listed in the result.

    Args:
        title: Event title.
        attendees: Employee ids, work emails, or exact full names.
        start: Start date-time in America/Chicago, ISO format "YYYY-MM-DDTHH:MM".
        duration_minutes: Length in minutes (5-600).
        description: Optional event description.

    Returns:
        A dict with `confirmation_id`, `start`, `end`, and `conflicts`, or {"error": message}.
    """
    return _call("calendar_create_event", {"title": title, "attendees": list(attendees), "start": start,
                                           "duration_minutes": duration_minutes, "description": description}, write=True)


def email_send(to: str, subject: str, body: str, cc: str = "") -> dict:
    """Send an email. Messages go to an outbox; nothing is delivered outside the system.

    Args:
        to: Recipient address, or several separated by commas.
        subject: Subject line.
        body: Plain-text body.
        cc: Optional CC addresses, separated by commas.

    Returns:
        A dict with `message_id`, `status`, and `external` (recipients outside lumenco.example), or {"error": message}.
    """
    return _call("email_send", {"to": to, "subject": subject, "body": body, "cc": cc}, write=True)


# ---------------------------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------------------------
WEB_TOOLS = [web_search, fetch_url]
RESEARCH_TOOLS = [web_search, fetch_url, wikipedia_search, wikipedia_page, arxiv_search]
CODE_TOOLS = [python_exec]
SQL_TOOLS = [sql_schema, sql_query]
SUPPORT_TOOLS = [crm_get_customer, crm_search_customers, tickets_search, tickets_get, tickets_update, kb_search, kb_get_article]
COMMERCE_TOOLS = [orders_list, orders_get, orders_refund, products_search, inventory_check]
TRAVEL_TOOLS = [travel_search_flights, travel_search_hotels, travel_book]
FINANCE_TOOLS = [finance_quote, finance_history]
HR_TOOLS = [hr_get_employee, hr_search_employees, hr_policy_lookup]
PRODUCTIVITY_TOOLS = [calendar_get_events, calendar_free_slots, calendar_create_event, email_send]

ALL_TOOLS = list(dict.fromkeys(RESEARCH_TOOLS + CODE_TOOLS + SQL_TOOLS + SUPPORT_TOOLS + COMMERCE_TOOLS + TRAVEL_TOOLS
                               + FINANCE_TOOLS + HR_TOOLS + PRODUCTIVITY_TOOLS))

__all__ = [f.__name__ for f in ALL_TOOLS] + [
    "WEB_TOOLS", "RESEARCH_TOOLS", "CODE_TOOLS", "SQL_TOOLS", "SUPPORT_TOOLS", "COMMERCE_TOOLS", "TRAVEL_TOOLS",
    "FINANCE_TOOLS", "HR_TOOLS", "PRODUCTIVITY_TOOLS", "ALL_TOOLS", "TOOLS_URL", "ToolsServiceError", "set_run_id",
]
