# hay-ticket-resolver

Haystack pipeline: ticket viewer -> prompt builder -> Agent with KB/order/CRM tools that exits by writing a ticket reply.

- Framework: haystack
- Source: https://github.com/deepset-ai/haystack-cookbook/blob/main/notebooks/github_issue_resolver_agent.ipynb
- Model: claude-haiku-5-5
- Complexity: medium
- Tracing: openinference (see `../dsh_common.py`)

## What was adapted

Derived from deepset-ai/haystack-cookbook notebooks/github_issue_resolver_agent.ipynb.
Kept: the pipeline shape (an issue-viewer component that turns the issue into Documents -> ChatPromptBuilder
with the <issue-comment> template -> Agent), the Agent configuration (AnthropicChatGenerator, a system prompt,
tools, `exit_conditions=["write_..._comment"]`, `state_schema={"documents": list[Document]}`), and the mocked
comment-writing tool that doubles as the exit condition.
Changed: GitHub issue -> Lumen & Co support ticket (tools-service tickets/CRM/orders/KB);
GitHubRepoViewerTool -> KB search/article, order, customer and inventory tools that also append Documents
to agent state; the exit tool posts an internal note via tickets_update.

## Tools

tickets_get, tickets_update, kb_search, kb_get_article, orders_get, orders_list, crm_get_customer, inventory_check (tools service)

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 2.7 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0015 (Langfuse per-trace cost); smoke total $0.0046.
