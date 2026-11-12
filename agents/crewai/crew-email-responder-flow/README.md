# crew-email-responder-flow

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/flows/email_auto_responder_flow

Based on the action-required emails identified, draft responses for each. Ensure each response 

## What was adapted

Email auto-responder flow, adapted from crewAI-examples/flows/email_auto_responder_flow.

Shape kept from the source: a Flow whose @start step fetches new emails, then an EmailFilterCrew with three agents
(Senior Email Analyst -> Email Action Specialist -> Email Response Writer) filters them, pulls each thread,
and drafts replies. The source's filter/action/draft task prompts (from its CrewAI-LangGraph sibling) are kept.
Adapted: Gmail is replaced by the support-ticket inbox (a ticket is an email thread; the customer's address comes
from the CRM) and drafts go to the email outbox; Tavily is replaced by the knowledge base. The 180-second polling
loop is replaced by one pass. A @router decides between "empty inbox" and "draft replies" paths, and the inbox is
selected by a single LiteAgent (Agent.kickoff) with a pydantic output.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: medium.
- Shared tools: crm_get_customer, email_send, kb_search, orders_get, tickets_get, tickets_search, tickets_update.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 7.7 (per task: 4, 10, 9).
Mean cost per run: $0.0046 (Langfuse trace cost).
Task t002 matches no gold-tier sender and routes to the empty-inbox branch.
