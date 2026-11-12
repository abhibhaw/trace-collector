# lg-travel-support

Primary travel assistant delegating to flight/hotel/itinerary/trip-idea sub-assistants, with interrupt-before-sensitive-tools and an auto-approver.

- Source: https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/customer-support/customer-support.ipynb
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: complex.

## Derived from

Travel customer-support bot, Part 4 "Specialized Workflows" of the LangGraph customer-support tutorial.

## What was adapted

Kept from the source: State with a `dialog_state` stack (update_dialog_stack reducer) and
`user_info`; a fetch_user_info node; the primary assistant that routes with To*Assistant
pydantic tools; per-skill entry nodes (create_entry_node) that complete the routing tool call;
specialized assistants with CompleteOrEscalate; split safe / sensitive ToolNodes with tool-error
fallbacks; the shared leave_skill (pop_dialog_state) node; route_to_workflow; the Assistant
wrapper that re-prompts on empty output; and an InMemorySaver checkpointer compiled with
`interrupt_before` on every sensitive-tools node.
Human-in-the-loop: the tutorial pauses for a human to approve each sensitive action. Here an
`auto_approver` (an LLM playing the traveler, plus a seeded "change of mind" rate) decides. On
approval the graph resumes with invoke(None); on denial it resumes with the tutorial's
"API call denied by user" ToolMessage.
Multi-turn: when an assistant asks the traveler to confirm or choose, a `simulated_traveler` (LLM) answers,
up to 2 turns, like the tutorial's chat loop. Each reply re-enters through route_to_workflow.
Changed: Swiss Airlines with flights / hotels / car rental / excursions over a local SQLite file
becomes the Lumen & Co corporate travel desk over the tools service: flights, hotels, itinerary
(calendar + email) and trip-ideas assistants, company travel policy lookup, employee records.

## Tools

travel_search_flights, travel_search_hotels, travel_book, hr_policy_lookup, hr_get_employee, calendar_get_events, calendar_create_event, email_send, web_search, wikipedia_search

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [7, 6, 8] (mean 7.0). Cost per run from Langfuse: mean $0.0033 (total $0.0100 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.
