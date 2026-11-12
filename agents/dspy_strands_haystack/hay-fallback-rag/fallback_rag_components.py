"""Custom Haystack components for hay-fallback-rag.

They live in a normal importable module because the OpenInference Haystack instrumentor re-imports a
component's module by name to wrap its `run`, and the runner loads agent.py without registering it in sys.modules.
"""
from __future__ import annotations

from haystack import Document, component
from peira_harness import tools


@component
class WebSearch:
    """Keyless web search returning Haystack Documents (stands in for DuckduckgoApiWebSearch)."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str):
        res = tools.web_search(query=query, max_results=self.top_k)
        items = res.get("results", []) if isinstance(res, dict) else res
        docs, links = [], []
        for r in items:
            content = f"{r.get('title', '')}\n{r.get('snippet') or r.get('content') or ''}"
            docs.append(Document(content=content, meta={"link": r.get("url"), "title": r.get("title")}))
            links.append(r.get("url"))
        return {"documents": docs, "links": links}
