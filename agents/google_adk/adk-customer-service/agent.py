"""Customer service agent: one LlmAgent with many tools, guarded by before/after tool callbacks,
a before_agent profile loader, and a before_model rate-limit callback.

Derived from google/adk-samples python/agents/customer-service (removed from main in #2591; pinned).
Adaptations:
- Retailer: Cymbal Home & Garden -> Lumen & Co (smart-home), the company in the tools service.
- The profile comes from crm_get_customer for the CUST-id found in the first message (the source
  loads a hard-coded mock profile). before_tool still validates every customer_id against it.
- Cart, approvals, video link, CRM update and QR code stay local mock tools (as in the source);
  products, stock, orders, refunds, installer calendars, KB and email use peira_harness tools.
- Manager approval (sync_ask_for_approval) is a simulated human approver: <=10% is auto-approved by
  the before_tool callback (source rule), <=20% approved by the "manager", above that rejected.
- The customer is simulated: when the agent asks for confirmation, an auto-user says yes (max 3 turns).
"""
from __future__ import annotations

import datetime
import json
import logging
import re
import time
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.tools import BaseTool, ToolContext

import adk_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

prompts = C.load_local(__file__)
logger = logging.getLogger(__name__)

AGENT = AgentMeta(
    id="adk-customer-service",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/customer-service",
    description="Smart-home retail assistant with cart, recommendations, stock, refunds, installation scheduling and discount approvals guarded by tool callbacks.",
    complexity="medium",
    model=C.HAIKU,
    tracing="openinference",
)

RATE_LIMIT_SECS = 60
RPM_QUOTA = 60
MAX_DISCOUNT_RATE = 10
MANAGER_MAX_DISCOUNT_RATE = 20
_CUST_RE = re.compile(r"\bCUST-\d{5}\b", re.I)


# --- local tools (mocks in the source as well) ---------------------------------------------------
def send_call_companion_link(phone_number: str) -> dict:
    """Sends a link to the user's phone number to start a video session.

    Args:
        phone_number (str): The phone number to send the link to.

    Returns:
        dict: A dictionary with the status and message.
    """
    return {"status": "success", "message": f"Link sent to {phone_number}", "session_id": f"VC-{uuid.uuid4().hex[:8]}"}


def approve_discount(discount_type: str, value: float, reason: str) -> dict:
    """Approve the flat rate or percentage discount requested by the user.

    Args:
        discount_type (str): The type of discount, either "percentage" or "flat".
        value (float): The value of the discount.
        reason (str): The reason for the discount.

    Returns:
        dict: A dictionary indicating the status of the approval.
    """
    if discount_type == "percentage" and value > MAX_DISCOUNT_RATE:
        return {"status": "rejected", "message": f"Discounts above {MAX_DISCOUNT_RATE}% need manager approval (sync_ask_for_approval)."}
    return {"status": "ok", "discount_type": discount_type, "value": value, "reason": reason}


def sync_ask_for_approval(discount_type: str, value: float, reason: str) -> dict:
    """Asks the manager for approval for a discount.

    Args:
        discount_type (str): The type of discount, either "percentage" or "flat".
        value (float): The value of the discount.
        reason (str): The reason for the discount.

    Returns:
        dict: A dictionary indicating the status of the approval.
    """
    # Simulated human-in-the-loop manager decision.
    pct = value if discount_type == "percentage" else min(100.0, value / 5.0)
    if pct <= MANAGER_MAX_DISCOUNT_RATE and reason.strip():
        return {"status": "approved", "approved_by": "Store Manager (simulated)", "value": value, "discount_type": discount_type}
    return {"status": "rejected", "approved_by": "Store Manager (simulated)",
            "message": f"Manager declined: discounts above {MANAGER_MAX_DISCOUNT_RATE}% or without a reason are not approved."}


def update_salesforce_crm(customer_id: str, details: dict) -> dict:
    """Updates the CRM with customer details.

    Args:
        customer_id (str): The ID of the customer.
        details (dict): A dictionary of details to update in the CRM.

    Returns:
        dict: A dictionary with status and message.
    """
    return {"status": "success", "message": f"CRM record for {customer_id} updated.", "fields": sorted(details)}


def access_cart_information(customer_id: str, tool_context: ToolContext) -> dict:
    """Retrieves the cart information for a given customer.

    Args:
        customer_id (str): The ID of the customer.

    Returns:
        dict: A dictionary representing the cart contents and subtotal.
    """
    cart = tool_context.state.get("cart", {"items": []})
    subtotal = round(sum((i["price_usd"] or 0) * i["quantity"] for i in cart["items"]), 2)
    return {"customer_id": customer_id, "items": cart["items"], "subtotal_usd": subtotal, "discount": cart.get("discount")}


def modify_cart(customer_id: str, items_to_add: list[dict], items_to_remove: list[str], tool_context: ToolContext) -> dict:
    """Modifies the user's shopping cart by adding and/or removing items.

    Args:
        customer_id (str): The ID of the customer.
        items_to_add (list): A list of {"sku": str, "quantity": int} to add.
        items_to_remove (list): A list of SKUs to remove.

    Returns:
        dict: A dictionary indicating the status of the cart modification.
    """
    cart = dict(tool_context.state.get("cart", {"items": []}))
    items = [i for i in cart["items"] if i["sku"] not in {s.upper() for s in items_to_remove}]
    added, errors = [], []
    for it in items_to_add:
        sku = str(it.get("sku", "")).upper()
        inv = T.inventory_check(sku)
        if "error" in inv:
            errors.append(f"{sku}: {inv['error']}")
            continue
        if inv.get("discontinued"):
            errors.append(f"{sku}: discontinued")
            continue
        if inv.get("price_usd") is None:
            errors.append(f"{sku}: no list price on file; cannot be added to the cart")
            continue
        items.append({"sku": sku, "name": inv["name"], "price_usd": inv["price_usd"], "quantity": int(it.get("quantity", 1))})
        added.append(sku)
    cart["items"] = items
    tool_context.state["cart"] = cart
    return {"status": "success" if not errors else "partial", "items_added": added, "items_removed": items_to_remove, "errors": errors}


def get_product_recommendations(device_type: str, customer_id: str) -> dict:
    """Provides product recommendations based on the type of device or need.

    Args:
        device_type: The type of device or need, e.g. "outdoor camera", "leak sensor", "hub".
        customer_id: Optional customer ID for personalized recommendations.

    Returns:
        A dictionary of recommended products.
    """
    res = T.products_search(query=device_type, limit=5)
    if not res.get("products"):
        words = [w for w in re.findall(r"[a-zA-Z]+", device_type) if len(w) > 3]
        if words:
            res = T.products_search(query=words[-1], limit=5)
    return {"recommendations": res.get("products", []), "categories": res.get("categories", [])}


def check_product_availability(product_id: str, store_id: str) -> dict:
    """Checks the availability of a product (SKU) in a warehouse/store.

    Args:
        product_id: The SKU of the product to check (e.g. SKU-2001).
        store_id: The warehouse to check (WH-EAST, WH-WEST, WH-EU) or "any".

    Returns:
        A dictionary indicating availability.
    """
    inv = T.inventory_check(product_id)
    if "error" in inv:
        return inv
    by_wh = inv.get("stock_by_warehouse", {})
    qty = inv.get("total_available", 0) if store_id.lower() in ("", "any") else by_wh.get(store_id.upper(), 0)
    return {"sku": inv["sku"], "available": qty > 0, "quantity": qty, "store": store_id, "stock_by_warehouse": by_wh}


_installer_cache: list[str] = []


def _installer() -> str:
    if not _installer_cache:
        emps = T.hr_search_employees(department="Support", limit=20).get("employees") or [{"employee_id": "EMP-040"}]
        active = [e for e in emps if e.get("status", "active") == "active"]
        _installer_cache.append((active or emps)[0]["employee_id"])
    return _installer_cache[0]


def get_available_installation_times(date: str) -> dict:
    """Retrieves available installation time slots for a given date.

    Args:
        date: The date to check (YYYY-MM-DD).

    Returns:
        A dictionary of available 2-hour time slots.
    """
    return T.calendar_free_slots(attendees=[_installer()], date=date, duration_minutes=120)


def schedule_installation_service(customer_id: str, date: str, time_range: str, details: str) -> dict:
    """Schedules a professional installation service appointment.

    Args:
        customer_id: The ID of the customer.
        date: The desired date (YYYY-MM-DD).
        time_range: The start time "HH:MM" of the slot.
        details: Devices to install and any notes.

    Returns:
        A dictionary indicating the status of the scheduling.
    """
    start = f"{date}T{time_range.split('-')[0].strip()}"
    ev = T.calendar_create_event(
        title=f"Installation visit for {customer_id}", attendees=[_installer()], start=start, duration_minutes=120, description=details
    )
    return {"status": "success", "appointment": ev}


def send_care_instructions(customer_id: str, device_type: str, delivery_method: str, tool_context: ToolContext) -> dict:
    """Sends an email with setup and care instructions for a device type from the knowledge base.

    Args:
        customer_id: The ID of the customer.
        device_type: The type of device, e.g. "smart lock".
        delivery_method: "email" (only email is supported).

    Returns:
        A dictionary indicating the status and the articles sent.
    """
    hits = T.kb_search(query=f"{device_type} setup", max_results=3).get("results", [])
    profile = json.loads(tool_context.state.get("customer_profile") or "{}")
    to = profile.get("email") or f"{customer_id.lower()}@example.com"
    body = "\n".join(f"- {h.get('title')} ({h.get('article_id')})" for h in hits) or "No matching articles."
    sent = T.email_send(to=to, subject=f"Setup and care for your {device_type}", body=body)
    return {"status": "success", "delivery_method": delivery_method, "articles": [h.get("article_id") for h in hits], "email": sent}


def generate_qr_code(customer_id: str, discount_value: float, discount_type: str, expiration_days: int) -> dict:
    """Generates a QR code for a discount.

    Args:
        customer_id: The ID of the customer.
        discount_value: The value of the discount (e.g., 10 for 10%).
        discount_type: "percentage" or "fixed".
        expiration_days: Number of days until the QR code expires.

    Returns:
        A dictionary with the QR code payload.
    """
    exp = (datetime.date.fromisoformat(C.MOCK_TODAY) + datetime.timedelta(days=expiration_days)).isoformat()
    return {"status": "success", "qr_code_data": f"LUMEN-{customer_id}-{discount_type[:3].upper()}{discount_value:g}-{exp}", "expiration_date": exp}


# --- callbacks (adapted from the source's shared_libraries/callbacks.py) -------------------------
def rate_limit_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """Callback function that implements a query rate limit."""
    for content in llm_request.contents:
        for part in content.parts or []:
            if part.text == "":
                part.text = " "
    now = time.time()
    if "timer_start" not in callback_context.state:
        callback_context.state["timer_start"] = now
        callback_context.state["request_count"] = 1
        return None
    request_count = callback_context.state["request_count"] + 1
    elapsed_secs = now - callback_context.state["timer_start"]
    if request_count > RPM_QUOTA:
        delay = RATE_LIMIT_SECS - elapsed_secs + 1
        if delay > 0:
            time.sleep(delay)
        callback_context.state["timer_start"] = now
        callback_context.state["request_count"] = 1
    else:
        callback_context.state["request_count"] = request_count
    return None


def validate_customer_id(customer_id: str, session_state) -> tuple[bool, str | None]:
    """Validates the customer ID against the customer profile in the session state."""
    if not session_state.get("customer_profile"):
        return False, "No customer profile selected. Ask the customer for their customer id (CUST-xxxxx)."
    try:
        c = json.loads(session_state["customer_profile"])
    except ValueError:
        return False, "Customer profile couldn't be parsed. Please reload the customer data."
    if customer_id.strip().upper() == c["customer_id"]:
        return True, None
    return False, f"You cannot use the tool with customer_id {customer_id}, only for {c['customer_id']}."


def before_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext):
    # Several tools require customer_id as input. We don't want to rely solely on the model
    # picking the right customer id. We validate it.
    if "customer_id" in args:
        valid, err = validate_customer_id(str(args["customer_id"]), tool_context.state)
        if not valid:
            return {"status": "error", "message": err}
    if tool.name in ("orders_get", "orders_refund") and tool_context.state.get("customer_profile"):
        owner = json.loads(tool_context.state["customer_profile"])["customer_id"]
        order = T.orders_get(args.get("order_id", ""))
        if "error" in order:
            return {"status": "error", "message": order["error"]}
        if order.get("customer_id") != owner:
            return {"status": "error", "message": f"Order {args.get('order_id')} does not belong to customer {owner}."}
    if tool.name == "sync_ask_for_approval":
        amount = args.get("value", None)
        if amount is not None and args.get("discount_type") == "percentage" and amount <= MAX_DISCOUNT_RATE:
            return {"status": "approved", "message": "You can approve this discount; no manager needed."}
    return None


def after_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any):
    # After approvals, we perform operations deterministically in the callback to apply the discount in the cart.
    approved = (tool.name == "sync_ask_for_approval" and isinstance(tool_response, dict) and tool_response.get("status") == "approved") or (
        tool.name == "approve_discount" and isinstance(tool_response, dict) and tool_response.get("status") == "ok"
    )
    if approved:
        cart = dict(tool_context.state.get("cart", {"items": []}))
        cart["discount"] = {"type": args.get("discount_type"), "value": args.get("value"), "reason": args.get("reason")}
        tool_context.state["cart"] = cart
    return None


def before_agent(callback_context: CallbackContext):
    """Loads the customer profile into state from the CUST-id in the user's message."""
    callback_context.state["current_date"] = C.MOCK_TODAY
    if callback_context.state.get("customer_profile"):
        return None
    text = ""
    uc = callback_context.user_content
    if uc and uc.parts:
        text = " ".join(p.text or "" for p in uc.parts)
    m = _CUST_RE.search(text)
    if m:
        prof = T.crm_get_customer(m.group(0).upper())
        if "error" in prof:  # unknown id: leave the profile empty, the agent must ask again
            callback_context.state["customer_profile"] = ""
            callback_context.state["profile_lookup_error"] = prof["error"]
        else:
            callback_context.state["customer_profile"] = json.dumps(prof, default=str)
    return None


root_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    global_instruction=prompts.GLOBAL_INSTRUCTION,
    instruction=prompts.INSTRUCTION,
    name="customer_service_agent",
    tools=[
        C.async_tool(f)
        for f in (
            send_call_companion_link,
            approve_discount,
            sync_ask_for_approval,
            update_salesforce_crm,
            access_cart_information,
            modify_cart,
            get_product_recommendations,
            check_product_availability,
            T.orders_list,
            T.orders_get,
            T.orders_refund,
            schedule_installation_service,
            get_available_installation_times,
            send_care_instructions,
            generate_qr_code,
        )
    ],
    before_tool_callback=before_tool,
    after_tool_callback=after_tool,
    before_agent_callback=before_agent,
    before_model_callback=rate_limit_callback,
)


def _customer(turn: int, last_text: str, state: dict) -> str | None:
    """Simulated customer: confirms the assistant's proposed actions."""
    if not C.asks_question(last_text):
        return None
    if not state.get("customer_profile") and turn == 0:
        return "Sorry, I don't have my customer number handy. Can you still help with general questions?"
    return "Yes, please go ahead with that. Pick whatever option you think is best for me."


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    state = {"cart": {"items": []}, "customer_profile": "", "current_date": C.MOCK_TODAY}
    return await C.run_agent(root_agent, task, app_name="customer_service", initial_state=state, auto_user=_customer, max_turns=3, max_llm_calls=20)
