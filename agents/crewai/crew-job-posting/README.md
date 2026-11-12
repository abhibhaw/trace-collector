# crew-job-posting

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting

Conduct a short analysis of the industry related to the role and company (smart home / IoT 

## What was adapted

Job posting crew, adapted from crewAI-examples/crews/job-posting.

Shape kept from the source: research analyst, job description writer, and review specialist; five sequential
tasks (company culture, role requirements as structured JSON, draft, review/edit, industry analysis).
Adapted: Serper/WebsiteSearch/FileRead tools are replaced with web search, URL fetch, and the HR API
(employee directory for team context, policy lookup for real benefits). The hiring company is Lumen & Co.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: simple.
- Shared tools: fetch_url, hr_policy_lookup, hr_search_employees, web_search.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 11.0 (per task: 9, 11, 13).
Mean cost per run: $0.0130 (Langfuse trace cost).
