# crew-meeting-prep

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/crews/prep-for-a-meeting

Compile all the research findings, industry analysis, and strategic talking points into a concise, 

## What was adapted

Meeting preparation crew, adapted from crewAI-examples/crews/prep-for-a-meeting.

Shape kept from the source: research specialist and industry analyst run in parallel (async_execution=True),
then the meeting strategy advisor and the briefing coordinator consume both via task context.
Adapted: Exa search is replaced by web search plus the internal CRM, orders, tickets, HR directory and calendar,
so participants can be Lumen & Co customers, business accounts, or colleagues.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: simple.
- Shared tools: calendar_get_events, crm_get_customer, crm_search_customers, hr_search_employees, kb_search, orders_list, tickets_search, web_search, wikipedia_search.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 9.0 (per task: 8, 10, 9).
Mean cost per run: $0.0140 (Langfuse trace cost).
