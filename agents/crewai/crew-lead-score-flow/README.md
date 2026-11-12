# crew-lead-score-flow

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/flows/lead-score-flow

Compose and send a personalized follow-up email to this lead.\n\nLEAD\n----\nLead ID: {candidate_id}\n

## What was adapted

Lead score flow, adapted from crewAI-examples/flows/lead-score-flow.

Shape kept from the source: a Flow with typed state; @start loads the leads; score_leads runs one LeadScoreCrew
per lead concurrently (asyncio.gather of kickoff_async) with a pydantic CandidateScore output; a @router
"human_in_the_loop" step ranks the leads and either loops back to scoring with feedback
("scored_leads_feedback", via or_) or goes on to "generate_emails", where a LeadResponseCrew per lead writes a
follow-up email (proceed for the top leads, polite decline for the rest).
Adapted: candidates from a CSV become B2B leads from the CRM (an upsell to the Business plan or a sales offer
named in the request); a LiteAgent sources up to 3 leads; the human reviewer is a deterministic auto-approver
that asks for one re-score when every score is weak; emails go to the outbox with email_send instead of files.
The scoring agent runs on Sonnet; sourcing and email writing run on Haiku.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-sonnet-5-5 for the lead scorer, claude-haiku-5-5 for lead sourcing and emails.
- Complexity: complex.
- Shared tools: crm_get_customer, crm_search_customers, email_send, orders_list, tickets_search.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 14.0 (per task: 5, 15, 22).
Mean cost per run: $0.0573 (Langfuse trace cost).
Task t003 finds no matching leads and routes to the no-leads branch. Sonnet scoring is most of the cost.
