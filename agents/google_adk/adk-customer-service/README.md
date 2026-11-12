# adk-customer-service

Source: https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/customer-service

Smart-home retail assistant with cart, recommendations, stock, refunds, installation scheduling and discount approvals guarded by tool callbacks.

## Shape

single LlmAgent with 15 tools; before_agent (profile loader), before_tool (customer-id validation, ownership check, auto-approval), after_tool (apply discount), before_model (rate limit) callbacks; simulated customer turns

## Adaptations

- Retailer changed from Cymbal Home & Garden to Lumen & Co (the tools-service company). Prompt structure and tool roles kept.
- The profile comes from `crm_get_customer` for the CUST id in the first message.
- Cart, discount approval, video link, CRM update and QR code stay local mocks (as in the source). Manager approval is a simulated human (<=20% approved).
- Products, stock, orders, refunds, installer calendar, KB and email use the shared tools.

## Model and tracing

- Complexity: medium. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

products_search, inventory_check, orders_list, orders_get, orders_refund, crm_get_customer, hr_search_employees, calendar_free_slots, calendar_create_event, kb_search, email_send + local cart/approval tools

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 7.3 LLM generations per run.
Mean cost $0.0068 per run (Langfuse trace cost).
