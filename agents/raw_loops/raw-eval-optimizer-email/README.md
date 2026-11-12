# raw-eval-optimizer-email

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/evaluator_optimizer.ipynb

The cookbook's evaluator-optimizer `loop()`: generate (<thoughts>/<response>) then evaluate (<evaluation>PASS|NEEDS_IMPROVEMENT|FAIL + <feedback>), with previous attempts and feedback fed back.

## What was adapted

- Task changed from code to customer emails; a fact-gathering tool loop runs first.
- `while True` replaced by a cap of 3 rounds. email_send runs only after PASS and only if the requester asked to send.
- The evaluator gives PASS when all four criteria hold; optional polish goes into feedback. The cookbook's "no further suggestions" bar made Haiku never PASS.

## Tools

crm_search_customers, crm_get_customer, orders_list, orders_get, tickets_search, tickets_get, kb_search, kb_get_article, email_send

## Trace shape

agent > gather_facts (agent) > research-turn-N; round-N > generate (chain) + evaluate (evaluator); email_send tool

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 7.7. Mean cost per run: USD 0.0035 (Langfuse calculated cost).
