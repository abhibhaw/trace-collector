# da-content-builder

Source: https://github.com/langchain-ai/deepagents/tree/main/examples/content-builder-agent

## What was adapted

Kept: AGENTS.md brand voice, blog-post and social-media skills verbatim, researcher sub-agent prompt, generate_cover and generate_social_image tool interfaces. FilesystemBackend -> StateBackend. Gemini image generation -> a local mock that writes a placeholder file with the prompt (Command state update). Tavily -> web_search + fetch_url. Caps: main 12, researcher 5.

## Tools and features

web_search, fetch_url, generate_cover/generate_social_image (local mock), skills, memory, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 7.7. Mean cost per run: $0.0055 (Langfuse cost from scripts/inspect_trace.py).
