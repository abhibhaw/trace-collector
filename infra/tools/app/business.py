"""Mock business APIs over deterministic data (see mockdata.py).

Base data never changes. Writes go to the valkey journal and return confirmation ids.
If the caller sends X-Peira-Run-Id, later reads in that run see the run's own writes (per-run overlay).
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from datetime import date, datetime, timedelta

import store
from kb import tokens
from mockdata import AS_OF, CABIN_CODES, CITIES, CITY_BY_CODE, TIMEZONE, busy_blocks, data, flights_for, hotel_nightly, hotel_sold_out, hotels_in
from registry import ToolError, tool


def _conf(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def _date(s: str, field: str) -> date:
    try:
        return date.fromisoformat(s.strip()[:10])
    except ValueError:
        raise ToolError(f"{field} must be an ISO date like 2026-11-20 (got '{s}')") from None


def _match(q: str, *fields) -> bool:
    q = q.lower().strip()
    return any(q in (f or "").lower() for f in fields)


async def _idem():
    prev = await store.idem_get()
    if prev is not None:
        return {**prev, "idempotent_replay": True}
    return None


# --- CRM ------------------------------------------------------------------------------------
def _cust_summary(c: dict) -> dict:
    return {k: c[k] for k in ("customer_id", "name", "email", "company", "city", "country", "tier", "status")}


@tool("support")
async def crm_get_customer(customer_id: str) -> dict:
    """Full customer profile with order and ticket summary."""
    d = data()
    c = d.customer_by_id.get(customer_id.strip().upper())
    if not c:
        raise ToolError(f"Customer '{customer_id}' not found. Use crm_search_customers to look up ids.")
    orders = d.orders_by_customer.get(c["customer_id"], [])
    tickets = [t for t in d.tickets if t["customer_id"] == c["customer_id"]]
    return {**c, "order_count": len(orders),
            "lifetime_value_usd": round(sum(o["total_usd"] - o["refunded_usd"] for o in orders if o["status"] != "cancelled"), 2),
            "last_order_date": max((o["order_date"] for o in orders), default=None),
            "recent_order_ids": [o["order_id"] for o in sorted(orders, key=lambda o: o["order_date"], reverse=True)[:5]],
            "open_ticket_ids": [t["ticket_id"] for t in tickets if t["status"] in ("open", "pending")]}


@tool("support")
async def crm_search_customers(query: str = "", city: str = "", tier: str = "", status: str = "", limit: int = 10) -> dict:
    """Search customers by name, email, phone, or company, with optional filters."""
    d = data()
    res = []
    for c in d.customers:
        if query and not _match(query, c["name"], c["email"], c["company"], c["phone"], c["customer_id"]):
            continue
        if city and c["city"].lower() != city.lower().strip():
            continue
        if tier and c["tier"] != tier.lower().strip():
            continue
        if status and c["status"] != status.lower().strip():
            continue
        res.append(_cust_summary(c))
    limit = max(1, min(limit, 50))
    return {"total_matches": len(res), "customers": res[:limit], "truncated": len(res) > limit}


# --- Orders -----------------------------------------------------------------------------------
async def _order_view(o: dict) -> dict:
    refunds = await store.overlay_get("refund", o["order_id"])
    out = dict(o)
    if refunds:
        out["refunded_usd"] = round(o["refunded_usd"] + sum(r["amount_usd"] for r in refunds), 2)
        out["refunds_this_session"] = refunds
        if out["refunded_usd"] >= o["total_usd"] - 0.005:
            out["status"] = "refunded"
    return out


@tool("commerce")
async def orders_list(customer_id: str = "", status: str = "", date_from: str = "", date_to: str = "", sku: str = "", limit: int = 20) -> dict:
    """List orders filtered by customer, status, date range, or SKU (newest first)."""
    d = data()
    src = d.orders_by_customer.get(customer_id.strip().upper(), []) if customer_id else d.orders
    if customer_id and customer_id.strip().upper() not in d.customer_by_id and not src:
        raise ToolError(f"Customer '{customer_id}' not found.")
    a = _date(date_from, "date_from") if date_from else None
    b = _date(date_to, "date_to") if date_to else None
    res = []
    for o in src:
        od = date.fromisoformat(o["order_date"])
        if status and o["status"] != status.lower().strip():
            continue
        if (a and od < a) or (b and od > b):
            continue
        if sku and not any(it["sku"] == sku.strip().upper() for it in o["items"]):
            continue
        res.append(o)
    res.sort(key=lambda o: (o["order_date"], o["order_id"]), reverse=True)
    limit = max(1, min(limit, 100))
    return {"total_matches": len(res), "truncated": len(res) > limit,
            "orders": [{"order_id": o["order_id"], "customer_id": o["customer_id"], "order_date": o["order_date"], "status": o["status"],
                        "total_usd": o["total_usd"], "item_count": sum(i["qty"] for i in o["items"])} for o in res[:limit]]}


@tool("commerce")
async def orders_get(order_id: str) -> dict:
    """Full order detail: items, totals, shipping, tracking, refunds."""
    o = data().order_by_id.get(order_id.strip().upper())
    if not o:
        raise ToolError(f"Order '{order_id}' not found.")
    return await _order_view(o)


@tool("commerce", write=True)
async def orders_refund(order_id: str, amount_usd: float = 0.0, reason: str = "") -> dict:
    """Issue a refund (amount 0 = full remaining balance)."""
    if (prev := await _idem()) is not None:
        return prev
    o = data().order_by_id.get(order_id.strip().upper())
    if not o:
        raise ToolError(f"Order '{order_id}' not found.")
    v = await _order_view(o)
    if v["status"] in ("cancelled",):
        raise ToolError(f"Order {o['order_id']} was cancelled before shipping and was already refunded automatically.")
    if v["status"] == "processing":
        raise ToolError(f"Order {o['order_id']} has not shipped yet; cancel it instead of refunding.")
    remaining = round(v["total_usd"] - v["refunded_usd"], 2)
    if remaining <= 0:
        raise ToolError(f"Order {o['order_id']} is already fully refunded (USD {v['refunded_usd']}).")
    amt = round(amount_usd, 2) if amount_usd and amount_usd > 0 else remaining
    if amt > remaining + 0.005:
        raise ToolError(f"Refund amount USD {amt} exceeds the refundable balance USD {remaining}.")
    warnings = []
    if v["delivery_date"] and (AS_OF - date.fromisoformat(v["delivery_date"])).days > 30:
        warnings.append("Delivered more than 30 days ago: outside the standard return window (policy customer-refunds).")
    if o["notes"] and "purged" in o["notes"]:
        warnings.append("Customer record for this order no longer exists.")
    status = "pending_approval" if amt > 500 else "approved"
    result = {"confirmation_id": _conf("RF"), "order_id": o["order_id"], "amount_usd": amt, "currency": "USD", "status": status,
              "reason": reason or None, "refund_method": o["payment_method"], "remaining_refundable_usd": round(remaining - amt, 2),
              "warnings": warnings,
              "note": "Refunds over USD 500 need Support Manager approval." if status == "pending_approval" else "Refund issued; 5-7 business days to appear."}
    await store.journal("refund", {"entity": o["order_id"], "args": {"amount_usd": amt, "reason": reason}, "result": result})
    await store.overlay_put("refund", o["order_id"], {"confirmation_id": result["confirmation_id"], "amount_usd": amt, "status": status})
    return result


@tool("commerce")
async def products_search(query: str = "", category: str = "", max_price_usd: float = 0.0, include_discontinued: bool = False, limit: int = 10) -> dict:
    """Search the product catalog by name, SKU, or category."""
    d = data()
    res = []
    for p in d.products:
        if query and not _match(query, p["name"], p["sku"], p["category"]):
            continue
        if category and p["category"] != category.lower().strip():
            continue
        if max_price_usd and (p["price_usd"] is None or p["price_usd"] > max_price_usd):
            continue
        if p["discontinued"] and not include_discontinued:
            continue
        res.append({k: p[k] for k in ("sku", "name", "category", "price_usd", "discontinued")} | {"total_stock": sum(p["stock"].values())})
    limit = max(1, min(limit, 50))
    return {"total_matches": len(res), "products": res[:limit], "truncated": len(res) > limit,
            "categories": sorted({p["category"] for p in d.products})}


@tool("commerce")
async def inventory_check(sku: str) -> dict:
    """Stock by warehouse for one SKU, with reorder status."""
    p = data().product_by_sku.get(sku.strip().upper())
    if not p:
        raise ToolError(f"SKU '{sku}' not found. Use products_search to find SKUs.")
    total = sum(p["stock"].values())
    out = {**{k: p[k] for k in ("sku", "name", "category", "price_usd", "discontinued", "reorder_point")},
           "stock_by_warehouse": p["stock"], "total_available": max(0, sum(max(0, v) for v in p["stock"].values())),
           "needs_reorder": total <= p["reorder_point"], "as_of": AS_OF.isoformat()}
    if any(v < 0 for v in p["stock"].values()):
        out["data_warning"] = "Negative stock recorded in a warehouse (likely an unreconciled return or count error)."
    return out


# --- Tickets ----------------------------------------------------------------------------------
async def _ticket_view(t: dict, ups: list | None = None) -> dict:
    if ups is None:
        ups = await store.overlay_get("ticket", t["ticket_id"])
    out = dict(t)
    if ups:
        notes = []
        for u in ups:
            for k in ("status", "priority", "assignee"):
                if u.get(k):
                    out[k] = u[k]
            if u.get("note"):
                notes.append(u["note"])
            out["updated_at"] = u["at"]
        if notes:
            out["internal_notes"] = notes
    return out


@tool("support")
async def tickets_search(query: str = "", status: str = "", customer_id: str = "", priority: str = "", category: str = "", limit: int = 10) -> dict:
    """Search support tickets by text and filters (newest first)."""
    d = data()
    res = []
    overlay = await store.overlay_get("ticket")  # one round trip for the whole run overlay
    for t in d.tickets:
        if customer_id and t["customer_id"] != customer_id.strip().upper():
            continue
        if query and not _match(query, t["subject"], t["body"], t["ticket_id"], t["order_id"]):
            continue
        if priority and t["priority"] != priority.lower().strip():
            continue
        if category and t["category"] != category.lower().strip():
            continue
        tv = await _ticket_view(t, overlay.get(t["ticket_id"], []))
        if status and tv["status"] != status.lower().strip():
            continue
        res.append(tv)
    res.sort(key=lambda t: t["created_at"], reverse=True)
    limit = max(1, min(limit, 50))
    return {"total_matches": len(res), "truncated": len(res) > limit,
            "tickets": [{k: t[k] for k in ("ticket_id", "customer_id", "order_id", "subject", "category", "priority", "status", "created_at")} for t in res[:limit]]}


@tool("support")
async def tickets_get(ticket_id: str) -> dict:
    """Full ticket including the customer's message."""
    t = data().ticket_by_id.get(ticket_id.strip().upper())
    if not t:
        raise ToolError(f"Ticket '{ticket_id}' not found.")
    return await _ticket_view(t)


@tool("support", write=True)
async def tickets_update(ticket_id: str, status: str = "", priority: str = "", assignee: str = "", note: str = "") -> dict:
    """Change ticket status/priority/assignee and/or add an internal note."""
    if (prev := await _idem()) is not None:
        return prev
    d = data()
    t = d.ticket_by_id.get(ticket_id.strip().upper())
    if not t:
        raise ToolError(f"Ticket '{ticket_id}' not found.")
    status, priority, assignee = status.lower().strip(), priority.lower().strip(), assignee.strip().upper()
    if status and status not in ("open", "pending", "resolved", "closed"):
        raise ToolError("status must be one of: open, pending, resolved, closed")
    if priority and priority not in ("low", "normal", "high", "urgent"):
        raise ToolError("priority must be one of: low, normal, high, urgent")
    if assignee:
        e = d.employee_by_id.get(assignee)
        if not e:
            raise ToolError(f"Assignee '{assignee}' is not an employee id (EMP-xxx).")
        if e["status"] != "active":
            raise ToolError(f"Employee {assignee} is {e['status']} and cannot be assigned tickets.")
    if not any([status, priority, assignee, note.strip()]):
        raise ToolError("Nothing to update: give status, priority, assignee, or note.")
    before = await _ticket_view(t)
    # Timestamps stay inside the simulated world: AS_OF plus the real time of day.
    at = datetime.combine(AS_OF, datetime.utcnow().time()).isoformat(timespec="seconds")
    upd = {"status": status or None, "priority": priority or None, "assignee": assignee or None, "note": note.strip() or None, "at": at}
    result = {"confirmation_id": _conf("TU"), "ticket_id": t["ticket_id"],
              "changes": {k: {"from": before.get(k), "to": v} for k, v in upd.items() if v and k in ("status", "priority", "assignee")},
              "note_added": bool(upd["note"]), "updated_at": at}
    await store.journal("ticket_update", {"entity": t["ticket_id"], "args": upd, "result": result})
    await store.overlay_put("ticket", t["ticket_id"], upd)
    return result


_KB_INDEX: tuple | None = None


def _kb_index() -> tuple:
    global _KB_INDEX
    if _KB_INDEX is None:
        df, toks = Counter(), {}
        for a in data().kb:
            tt = tokens(a["title"]) * 3 + tokens(a["body"])
            toks[a["article_id"]] = Counter(tt)
            df.update(set(tt))
        _KB_INDEX = (len(data().kb), df, toks)
    return _KB_INDEX


@tool("support")
async def kb_search(query: str, max_results: int = 5) -> dict:
    """Search the Lumen & Co product knowledge base."""
    d = data()
    q = [w for w in tokens(query) if len(w) > 1]
    if not q:
        raise ToolError("query must contain at least one word")
    n_docs, df, toks = _kb_index()
    scored = []
    for a in d.kb:
        c = toks[a["article_id"]]
        s = sum((c[w] / (c[w] + 1.2)) * math.log(1 + n_docs / (1 + df[w])) for w in q if c[w])
        if s > 0:
            scored.append((s, a))
    scored.sort(key=lambda x: (-x[0], x[1]["article_id"]))
    max_results = max(1, min(max_results, 20))
    return {"query": query, "results": [{"article_id": a["article_id"], "title": a["title"], "category": a["category"], "updated": a["updated"],
                                          "snippet": a["body"][:280], "score": round(s, 3)} for s, a in scored[:max_results]]}


@tool("support")
async def kb_get_article(article_id: str) -> dict:
    """Full text of one knowledge-base article."""
    a = data().kb_by_id.get(article_id.strip().upper())
    if not a:
        raise ToolError(f"Article '{article_id}' not found. Use kb_search to find article ids.")
    return a


# --- Travel -----------------------------------------------------------------------------------
def _city_code(s: str) -> str:
    s = s.strip()
    if s.upper() in CITY_BY_CODE:
        return s.upper()
    for c in CITIES:
        if c[1].lower() == s.lower():
            return c[0]
    raise ToolError(f"Unknown city or airport '{s}'. Supported: " + ", ".join(f"{c[1]} ({c[0]})" for c in CITIES))


@tool("travel")
async def travel_search_flights(origin: str, destination: str, date: str, cabin: str = "economy", max_stops: int = 1, max_price_usd: float = 0.0) -> dict:
    """Search flights between two of the 40 supported cities on a date."""
    o, dst = _city_code(origin), _city_code(destination)
    if o == dst:
        raise ToolError("origin and destination must differ")
    day = _date(date, "date")
    if day < AS_OF:
        raise ToolError(f"Cannot search flights in the past (today is {AS_OF.isoformat()}).")
    if day > AS_OF + timedelta(days=330):
        raise ToolError("Flights are only bookable up to 330 days ahead.")
    cabin = cabin.lower().strip().replace(" ", "_")
    if cabin not in ("economy", "premium_economy", "business", "first"):
        raise ToolError("cabin must be one of: economy, premium_economy, business, first")
    res = [f for f in flights_for(o, dst, day, cabin) if f["stops"] <= max_stops and (not max_price_usd or f["price_usd"] <= max_price_usd)]
    return {"origin": f"{CITY_BY_CODE[o][1]} ({o})", "destination": f"{CITY_BY_CODE[dst][1]} ({dst})", "date": day.isoformat(), "cabin": cabin,
            "flights": res, "note": "Times are local to the departure city. seats_left=0 means sold out." if res else "No flights match; relax max_stops or max_price_usd."}


@tool("travel")
async def travel_search_hotels(city: str, check_in: str, check_out: str, guests: int = 1, max_price_per_night_usd: float = 0.0) -> dict:
    """Search hotels in a supported city for a date range."""
    code = _city_code(city)
    a, b = _date(check_in, "check_in"), _date(check_out, "check_out")
    if b <= a:
        raise ToolError("check_out must be after check_in")
    if a < AS_OF:
        raise ToolError(f"check_in is in the past (today is {AS_OF.isoformat()}).")
    if (b - a).days > 30:
        raise ToolError("Stays longer than 30 nights are not supported.")
    if not 1 <= guests <= 8:
        raise ToolError("guests must be between 1 and 8")
    nights = [a + timedelta(days=i) for i in range((b - a).days)]
    res = []
    for h in hotels_in(code):
        rates = [hotel_nightly(h, n) * (1 + 0.15 * max(0, guests - 2)) for n in nights]
        avg = round(sum(rates) / len(rates), 2)
        sold = any(hotel_sold_out(h, n) for n in nights)
        if max_price_per_night_usd and avg > max_price_per_night_usd:
            continue
        res.append({**{k: h[k] for k in ("hotel_id", "name", "stars", "guest_rating", "neighborhood", "amenities", "cancellation")},
                    "avg_nightly_usd": avg, "total_usd": round(sum(rates), 2), "available": not sold})
    res.sort(key=lambda x: x["avg_nightly_usd"])
    return {"city": CITY_BY_CODE[code][1], "check_in": a.isoformat(), "check_out": b.isoformat(), "nights": len(nights), "guests": guests, "hotels": res}


@tool("travel", write=True)
async def travel_book(offer_id: str, traveler_name: str, check_in: str = "", check_out: str = "", traveler_email: str = "") -> dict:
    """Book a flight (offer_id FL-...) or hotel (hotel_id HT-..., needs dates)."""
    if (prev := await _idem()) is not None:
        return prev
    oid = offer_id.strip().upper()
    if not traveler_name.strip():
        raise ToolError("traveler_name is required")
    m = re.fullmatch(r"FL-([A-Z]{3})-([A-Z]{3})-(\d{8})-(\d+)([YWJF])", oid)
    if m:
        o, dst, ds, _, cc = m.groups()
        try:
            day = datetime.strptime(ds, "%Y%m%d").date()
        except ValueError:
            raise ToolError(f"Offer '{offer_id}' is invalid.") from None
        if o not in CITY_BY_CODE or dst not in CITY_BY_CODE or day < AS_OF:
            raise ToolError(f"Offer '{offer_id}' is invalid or expired.")
        cabin = {v: k for k, v in CABIN_CODES.items()}[cc]
        flight = next((f for f in flights_for(o, dst, day, cabin) if f["offer_id"] == oid), None)
        if not flight:
            raise ToolError(f"Offer '{offer_id}' not found. Search flights again.")
        if flight["seats_left"] == 0:
            raise ToolError(f"Flight {flight['flight_number']} is sold out. Choose another offer.")
        result = {"confirmation_id": _conf("BK"), "type": "flight", "status": "confirmed", "traveler": traveler_name.strip(),
                  "flight_number": flight["flight_number"], "airline": flight["airline"], "route": f"{o}-{dst}",
                  "departure_local": flight["departure_local"], "cabin": cabin, "price_usd": flight["price_usd"], "refundable": flight["refundable"]}
    else:
        m = re.fullmatch(r"HT-([A-Z]{3})-(\d{2})", oid)
        if not m:
            raise ToolError("offer_id must be a flight offer id from travel_search_flights (like FL-JFK-LHR-20261201-2Y) or a hotel id (like HT-LHR-03).")
        if m.group(1) not in CITY_BY_CODE:
            raise ToolError(f"Hotel '{offer_id}' not found.")
        h = next((x for x in hotels_in(m.group(1)) if x["hotel_id"] == oid), None)
        if not h:
            raise ToolError(f"Hotel '{offer_id}' not found.")
        if not check_in or not check_out:
            raise ToolError("Hotel bookings need check_in and check_out dates.")
        a, b = _date(check_in, "check_in"), _date(check_out, "check_out")
        if b <= a or a < AS_OF:
            raise ToolError("Invalid dates: check_out must be after check_in, and check_in cannot be in the past.")
        nights = [a + timedelta(days=i) for i in range((b - a).days)]
        if any(hotel_sold_out(h, n) for n in nights):
            raise ToolError(f"{h['name']} is sold out for at least one night in that range.")
        total = round(sum(hotel_nightly(h, n) for n in nights), 2)
        result = {"confirmation_id": _conf("BK"), "type": "hotel", "status": "confirmed", "traveler": traveler_name.strip(),
                  "hotel_id": oid, "hotel": h["name"], "city": h["city"], "check_in": a.isoformat(), "check_out": b.isoformat(),
                  "nights": len(nights), "total_usd": total, "cancellation": h["cancellation"]}
    if traveler_email:
        result["traveler_email"] = traveler_email.strip()
    await store.journal("booking", {"entity": oid, "args": {"traveler_name": traveler_name}, "result": result})
    return result


# --- Finance ----------------------------------------------------------------------------------
def _ticker(symbol: str) -> dict:
    s = data().prices.get(symbol.strip().upper())
    if not s:
        raise ToolError(f"Unknown symbol '{symbol}'. This market data feed covers {len(data().prices)} symbols; see the catalog.")
    return s


@tool("finance")
async def finance_quote(symbol: str) -> dict:
    """Latest quote for a ticker (synthetic market data, as of 2026-11-13)."""
    s = _ticker(symbol)
    rows = s["rows"]
    d, o, hi, lo, c, v = rows[-1]
    prev = rows[-2][4]
    last52 = [r for r in rows if r[0] >= (AS_OF - timedelta(days=365)).isoformat()]
    return {"symbol": s["symbol"], "name": s["name"], "sector": s["sector"], "exchange": s["exchange"], "currency": "USD",
            "as_of": d, "price": c, "open": o, "day_high": hi, "day_low": lo, "previous_close": prev,
            "change": round(c - prev, 2), "change_pct": round((c / prev - 1) * 100, 2), "volume": v,
            "week52_high": max(r[2] for r in last52), "week52_low": min(r[3] for r in last52),
            "market_cap_usd": round(c * s["shares_outstanding"]), "pe_ratio": round(c / s["eps_ttm"], 2) if s["eps_ttm"] else None,
            "dividend_yield_pct": s["dividend_yield_pct"], "data_note": "Synthetic prices for a research simulation; not real market data."}


@tool("finance")
async def finance_history(symbol: str, start_date: str = "", end_date: str = "", interval: str = "daily") -> dict:
    """Historical OHLCV bars (daily, weekly, or monthly)."""
    s = _ticker(symbol)
    end = _date(end_date, "end_date") if end_date else AS_OF
    start = _date(start_date, "start_date") if start_date else end - timedelta(days=30)
    if start > end:
        raise ToolError("start_date must be on or before end_date")
    rows = [r for r in s["rows"] if start.isoformat() <= r[0] <= end.isoformat()]
    interval = interval.lower().strip()
    if interval not in ("daily", "weekly", "monthly"):
        raise ToolError("interval must be daily, weekly, or monthly")
    if interval != "daily":
        buckets: dict[str, list] = {}
        for r in rows:
            dd = date.fromisoformat(r[0])
            k = (dd - timedelta(days=dd.weekday())).isoformat() if interval == "weekly" else dd.strftime("%Y-%m-01")
            buckets.setdefault(k, []).append(r)
        rows = [(k, b[0][1], max(x[2] for x in b), min(x[3] for x in b), b[-1][4], sum(x[5] for x in b)) for k, b in sorted(buckets.items())]
    cap = 400
    bars = [{"date": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in rows]
    out = {"symbol": s["symbol"], "interval": interval, "start": start.isoformat(), "end": end.isoformat(), "count": len(bars),
           "bars": bars[-cap:], "truncated": len(bars) > cap,
           "data_note": "Synthetic prices. Data starts 2023-01-02 and ends 2026-11-13."}
    if not bars:
        out["note"] = "No trading data in this range (weekend, holiday, trading halt, or outside the data range)."
    return out


# --- HR ---------------------------------------------------------------------------------------
@tool("hr")
async def hr_get_employee(employee_id: str) -> dict:
    """Employee record including manager, office, and PTO balance."""
    d = data()
    e = d.employee_by_id.get(employee_id.strip().upper())
    if not e:
        raise ToolError(f"Employee '{employee_id}' not found. Use hr_search_employees to look up ids.")
    m = d.employee_by_id.get(e["manager_id"]) if e["manager_id"] else None
    return {**e, "manager_name": m["name"] if m else None,
            "direct_reports": [x["employee_id"] for x in d.employees if x["manager_id"] == e["employee_id"]]}


@tool("hr")
async def hr_search_employees(query: str = "", department: str = "", office: str = "", limit: int = 20) -> dict:
    """Find employees by name, title, email, department, or office."""
    d = data()
    res = []
    for e in d.employees:
        if query and not _match(query, e["name"], e["title"], e["email"], e["employee_id"]):
            continue
        if department and e["department"].lower() != department.lower().strip():
            continue
        if office and (e["office"] or "").lower() != office.lower().strip():
            continue
        res.append({k: e[k] for k in ("employee_id", "name", "title", "department", "office", "email", "status")})
    limit = max(1, min(limit, 80))
    return {"total_matches": len(res), "employees": res[:limit], "truncated": len(res) > limit,
            "departments": sorted({e["department"] for e in d.employees})}


@tool("hr")
async def hr_policy_lookup(query: str) -> dict:
    """Find company policies by name or keyword; returns full policy text."""
    d = data()
    q = query.strip().lower()
    if q in d.policies:
        return {"policies": [{"policy_id": q, **d.policies[q]}]}
    words = [w for w in tokens(q) if len(w) > 2]
    scored = []
    for pid, p in d.policies.items():
        hay = tokens(pid + " " + p["title"]) * 3 + tokens(p["text"])
        s = sum(hay.count(w) for w in words)
        if s:
            scored.append((s, pid))
    scored.sort(reverse=True)
    if not scored:
        return {"policies": [], "note": "No matching policy. Available policy ids: " + ", ".join(d.policies)}
    return {"policies": [{"policy_id": pid, **d.policies[pid]} for _, pid in scored[:3]]}


# --- Calendar and email ----------------------------------------------------------------------
def _parse_dt(s: str, field: str) -> datetime:
    try:
        return datetime.fromisoformat(s.strip().replace("Z", "").replace(" ", "T"))
    except ValueError:
        raise ToolError(f"{field} must be ISO date-time like 2026-11-18T14:00 (got '{s}')") from None


def _attendees(ids: list[str]) -> list[str]:
    d = data()
    out = []
    for raw in ids:
        for x in str(raw).split(","):
            x = x.strip()
            if not x:
                continue
            e = d.employee_by_id.get(x.upper()) or next((e for e in d.employees if e["email"].lower() == x.lower() or e["name"].lower() == x.lower()), None)
            if not e:
                raise ToolError(f"Unknown attendee '{x}'. Use employee ids (EMP-xxx), work emails, or exact names.")
            out.append(e["employee_id"])
    if not out:
        raise ToolError("At least one attendee is required.")
    return sorted(set(out))


async def _busy(eid: str, day: date) -> list[dict]:
    blocks = busy_blocks(eid, day)
    created = await store.overlay_get("event", eid)
    blocks += [{"start": c["start"], "end": c["end"], "title": c["title"]} for c in created if c["start"][:10] == day.isoformat()]
    return sorted(blocks, key=lambda b: b["start"])


@tool("productivity")
async def calendar_get_events(employee_id: str, date: str) -> dict:
    """An employee's meetings on a date (times in America/Chicago)."""
    eid = _attendees([employee_id])[0]
    day = _date(date, "date")
    return {"employee_id": eid, "date": day.isoformat(), "timezone": TIMEZONE, "events": await _busy(eid, day)}


@tool("productivity")
async def calendar_free_slots(attendees: list[str], date: str, duration_minutes: int = 30) -> dict:
    """Common free slots for attendees on a date within 09:00-17:00 America/Chicago."""
    ids = _attendees(attendees)
    day = _date(date, "date")
    if not 15 <= duration_minutes <= 480:
        raise ToolError("duration_minutes must be between 15 and 480")
    d = data()
    inactive = [i for i in ids if d.employee_by_id[i]["status"] != "active"]
    if day.weekday() >= 5:
        return {"date": day.isoformat(), "attendees": ids, "free_slots": [], "note": "Weekend: no working hours."}
    busy = []
    for i in ids:
        for b in await _busy(i, day):
            s, e = datetime.fromisoformat(b["start"]), datetime.fromisoformat(b["end"])
            busy.append((s.hour * 60 + s.minute, e.hour * 60 + e.minute))
    slots, t = [], 9 * 60
    while t + duration_minutes <= 17 * 60:
        if all(t + duration_minutes <= s or t >= e for s, e in busy):
            slots.append({"start": f"{day.isoformat()}T{t // 60:02d}:{t % 60:02d}", "end": f"{day.isoformat()}T{(t + duration_minutes) // 60:02d}:{(t + duration_minutes) % 60:02d}"})
        t += 30
    out = {"date": day.isoformat(), "timezone": TIMEZONE, "attendees": ids, "duration_minutes": duration_minutes, "free_slots": slots}
    if inactive:
        out["warning"] = f"Not active (on leave or terminated): {', '.join(inactive)}"
    return out


@tool("productivity", write=True)
async def calendar_create_event(title: str, attendees: list[str], start: str, duration_minutes: int = 30, description: str = "") -> dict:
    """Create a calendar event; reports conflicts with existing meetings."""
    if (prev := await _idem()) is not None:
        return prev
    ids = _attendees(attendees)
    st = _parse_dt(start, "start")
    if not 5 <= duration_minutes <= 600:
        raise ToolError("duration_minutes must be between 5 and 600")
    if st.date() < AS_OF:
        raise ToolError(f"Cannot create events in the past (today is {AS_OF.isoformat()}).")
    en = st + timedelta(minutes=duration_minutes)
    conflicts = []
    for i in ids:
        for b in await _busy(i, st.date()):
            if datetime.fromisoformat(b["start"]) < en and st < datetime.fromisoformat(b["end"]):
                conflicts.append({"employee_id": i, **b})
    result = {"confirmation_id": _conf("EVT"), "title": title, "attendees": ids, "start": st.isoformat(timespec="minutes"),
              "end": en.isoformat(timespec="minutes"), "timezone": TIMEZONE, "status": "created", "conflicts": conflicts}
    await store.journal("calendar_event", {"entity": result["confirmation_id"], "args": {"title": title, "attendees": ids, "description": description}, "result": result})
    for i in ids:
        await store.overlay_put("event", i, {"start": result["start"], "end": result["end"], "title": title})
    return result


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@tool("productivity", write=True)
async def email_send(to: str, subject: str, body: str, cc: str = "") -> dict:
    """Send an email (goes to an outbox; nothing leaves the system)."""
    if (prev := await _idem()) is not None:
        return prev
    rcpt = [x.strip() for x in to.split(",") if x.strip()]
    ccs = [x.strip() for x in cc.split(",") if x.strip()]
    bad = [x for x in rcpt + ccs if not _EMAIL.match(x)]
    if not rcpt:
        raise ToolError("to is required")
    if bad:
        raise ToolError(f"Invalid email address(es): {', '.join(bad)}")
    if not subject.strip():
        raise ToolError("subject is required")
    if len(body) > 20000:
        raise ToolError("body is too long (max 20000 chars)")
    result = {"message_id": _conf("MSG"), "status": "queued_to_outbox", "to": rcpt, "cc": ccs, "subject": subject,
              "body_chars": len(body), "external": [x for x in rcpt + ccs if not x.lower().endswith("@lumenco.example")]}
    await store.journal("email", {"entity": result["message_id"], "args": {"to": rcpt, "cc": ccs, "subject": subject, "body": body[:4000]}, "result": result})
    return result
