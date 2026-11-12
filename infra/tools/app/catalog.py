"""Machine-readable catalog of entity ids for task writers (served at GET /v1/catalog)."""

from __future__ import annotations

from mockdata import AS_OF, CITIES, COMPANY, TIMEZONE, WAREHOUSES, data, hotels_in
from sqltools import DBS


def build_catalog() -> dict:
    d = data()
    act_orders = [o for o in d.orders if o["status"] == "delivered"]
    multi = [c for c in d.customers if len(d.orders_by_customer.get(c["customer_id"], [])) >= 5]
    return {
        "as_of": AS_OF.isoformat(),
        "timezone": TIMEZONE,
        "company": COMPANY,
        "counts": {"customers": len(d.customers), "products": len(d.products), "orders": len(d.orders), "tickets": len(d.tickets),
                   "employees": len(d.employees), "kb_articles": len(d.kb), "policies": len(d.policies), "tickers": len(d.prices),
                   "cities": len(CITIES)},
        "crm": {
            "customer_id_range": [d.customers[0]["customer_id"], d.customers[-1]["customer_id"]],
            "sample_customers": [{k: c[k] for k in ("customer_id", "name", "email", "city", "tier", "account_type")} for c in d.customers[:400:20]],
            "customers_with_5plus_orders": [c["customer_id"] for c in multi[:25]],
            "business_accounts": sorted({c["company"] for c in d.customers if c["company"]}),
            "messy": d.messy["customers"],
        },
        "commerce": {
            "order_id_range": [d.orders[0]["order_id"], d.orders[-1]["order_id"]],
            "sample_orders": [{k: o[k] for k in ("order_id", "customer_id", "order_date", "status", "total_usd")} for o in d.orders[::150]],
            "recent_delivered_refundable": [o["order_id"] for o in act_orders if o["delivery_date"] and o["delivery_date"] >= "2026-10-20"][:20],
            "order_statuses": sorted({o["status"] for o in d.orders}),
            "sample_products": [{k: p[k] for k in ("sku", "name", "category", "price_usd")} for p in d.products[::10]],
            "product_categories": sorted({p["category"] for p in d.products}),
            "warehouses": WAREHOUSES,
            "messy_orders": d.messy["orders"],
            "messy_products": d.messy["products"],
        },
        "support": {
            "ticket_id_range": [d.tickets[0]["ticket_id"], d.tickets[-1]["ticket_id"]],
            "sample_open_tickets": [t["ticket_id"] for t in d.tickets if t["status"] == "open"][:25],
            "ticket_categories": sorted({t["category"] for t in d.tickets}),
            "messy_tickets": d.messy["tickets"],
            "kb_article_id_range": [d.kb[0]["article_id"], d.kb[-1]["article_id"]],
            "kb_articles": [{"article_id": a["article_id"], "title": a["title"], "category": a["category"]} for a in d.kb],
            "kb_injection_articles": d.messy["kb"]["injection"],
            "kb_contradictions": d.messy["kb"]["contradiction"],
        },
        "travel": {
            "cities": [{"code": c[0], "city": c[1], "country": c[2]} for c in CITIES],
            "sample_hotels": {c[0]: [h["hotel_id"] for h in hotels_in(c[0])][:3] for c in CITIES[:6]},
            "flight_offer_id_format": "FL-<ORIGIN>-<DEST>-<YYYYMMDD>-<n><cabin Y|W|J|F>",
            "bookable_window": [AS_OF.isoformat(), "AS_OF + 330 days"],
        },
        "finance": {
            "tickers": [{"symbol": s, "name": v["name"], "sector": v["sector"]} for s, v in d.prices.items()],
            "data_range": [d.prices["AAPL"]["rows"][0][0], d.prices["AAPL"]["rows"][-1][0]],
            "messy": d.messy["finance"],
        },
        "hr": {
            "employees": [{k: e[k] for k in ("employee_id", "name", "title", "department", "office", "status")} for e in d.employees],
            "policy_ids": list(d.policies),
            "messy": d.messy["employees"] | {"policy_conflict": ["travel-2024", "travel-2026"]},
        },
        "sql": {"databases": list(DBS)},
    }
