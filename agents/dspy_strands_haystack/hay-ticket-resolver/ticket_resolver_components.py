"""Custom Haystack components for hay-ticket-resolver.

They live in a normal importable module because the OpenInference Haystack instrumentor re-imports a
component's module by name to wrap its `run`, and the runner loads agent.py without registering it in sys.modules.
"""
from __future__ import annotations

import json
import re

from haystack import Document, component
from peira_harness import tools


def _j(x) -> str:
    return json.dumps(x, default=str)[:5000]


@component
class TicketViewer:
    """Turns a ticket id (or free text) into Documents: the ticket first, then context (stands in for GitHubIssueViewer)."""

    @component.output_types(documents=list[Document])
    def run(self, request: str):
        m = re.search(r"TCK-\d{5}", request.upper())
        docs = [Document(content=request, meta={"title": "Request from the support queue", "type": "request"})]
        if m:
            t = tools.tickets_get(ticket_id=m.group(0))
            if "error" in t:
                docs.append(Document(content=f"Ticket lookup failed: {t['error']}", meta={"type": "error"}))
            else:
                docs = [Document(content=f"{t.get('subject')}\n\n{t.get('body')}", meta={"title": f"{t.get('ticket_id')}: {t.get('subject')}", "type": "ticket"}),
                        Document(content=_j({k: v for k, v in t.items() if k not in ('body',)}), meta={"type": "ticket_fields"}),
                        Document(content=request, meta={"type": "request"})]
        return {"documents": docs}
