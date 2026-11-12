"""Router query engine as a Workflow, adapted from LlamaIndex's router_query_engine workflow notebook.

Shape kept from the source (docs/examples/workflow/router_query_engine.ipynb):
- RouterQueryEngineWorkflow with three steps: `selector` (LLM selector over query-engine
  tool metadata, single or multi), `generate_responses` (runs the selected engines,
  in parallel when several are chosen), and `synthesize_responses` (TreeSummarize over the
  partial answers when more than one engine answered).
Adaptations: the notebook's vector and summary indexes over one essay need OpenAI
embeddings. They are replaced by four RetrieverQueryEngines over custom retrievers that call
the tools service (product KB, HR policies, market data, Wikipedia). No embeddings are used.
"""

import asyncio
import json
import re
from typing import List

from llama_index.core import Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import SelectorResult, SingleSelection
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.selectors.utils import get_selector_from_llm
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

import pl_common as pc
from peira_harness import AgentMeta, tools

AGENT = AgentMeta(
    id="li-router-qe",
    framework="llamaindex",
    source_url="https://github.com/run-llama/llama_index/blob/main/docs/examples/workflow/router_query_engine.ipynb",
    description="Router workflow: LLM multi-selector over four query-engine tools, parallel sub-queries, TreeSummarize synthesis.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="openinference",
)


# --------------------------------------------------------------------------- retrievers


class ToolRetriever(BaseRetriever):
    """Base: tool calls are blocking HTTP, so async retrieval runs them in a worker thread."""

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await asyncio.to_thread(self._retrieve, query_bundle)


def _nodes(items: list[tuple[str, dict]]) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=t, metadata=m), score=1.0 / (i + 1)) for i, (t, m) in enumerate(items)]


class KBRetriever(ToolRetriever):
    """Product knowledge base: search, then read the top articles in full."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        res = tools.kb_search(query_bundle.query_str, max_results=4)
        items = []
        for hit in (res.get("results") or [])[:3]:
            art = tools.kb_get_article(hit["article_id"])
            body = art.get("body", hit.get("snippet", "")) if isinstance(art, dict) else hit.get("snippet", "")
            items.append((f"[{hit['article_id']}] {hit['title']} (updated {hit.get('updated')})\n{body}",
                          {"article_id": hit["article_id"]}))
        return _nodes(items or [(f"No KB article matched: {res.get('error', 'no results')}", {})])


class PolicyRetriever(ToolRetriever):
    """Internal HR and company policies."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        res = tools.hr_policy_lookup(query_bundle.query_str)
        return _nodes([(pc.to_text(res, 7000), {"source": "hr_policy_lookup"})])


class MarketRetriever(ToolRetriever):
    """Quotes for ticker symbols mentioned in the question (synthetic market data)."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        syms = list(dict.fromkeys(re.findall(r"\b[A-Z]{2,5}\b", query_bundle.query_str)))[:3]
        if not syms:
            return _nodes([("No ticker symbol found in the question. Symbols are 2-5 uppercase letters, e.g. LMNC.", {})])
        return _nodes([(f"{s}: {pc.to_text(tools.finance_quote(s), 2500)}", {"symbol": s}) for s in syms])


class WikiRetriever(ToolRetriever):
    """General knowledge from Wikipedia."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        res = tools.wikipedia_search(query_bundle.query_str, max_results=3)
        hits = res.get("results", []) if isinstance(res, dict) else []
        items = []
        for h in hits[:2]:
            title = h.get("title", "")
            page = tools.wikipedia_page(title, max_chars=3000)
            items.append((f"{title}\n{page.get('content') or page.get('extract') or pc.to_text(page, 3000)}", {"title": title}))
        return _nodes(items or [(f"No Wikipedia result: {pc.to_text(res, 500)}", {})])


def build_query_engine_tools() -> list[QueryEngineTool]:
    spec = [
        ("product_kb", KBRetriever(), "Lumen & Co product knowledge base: device setup, Wi-Fi, firmware, LED codes, "
         "app features, Lumen Cloud plans, billing, orders, returns, shipping, and warranty."),
        ("company_policies", PolicyRetriever(), "Lumen & Co internal HR and company policies: PTO, sick and parental "
         "leave, remote work, expenses, travel, equipment, security, data retention, customer refunds, holidays."),
        ("market_data", MarketRetriever(), "Latest stock quotes for ticker symbols (synthetic market data), "
         "for example LMNC (Lumen & Co). Use only when the question names a ticker."),
        ("general_knowledge", WikiRetriever(), "General encyclopedic knowledge from Wikipedia: history, science, "
         "technology standards, places, people."),
    ]
    out = []
    for name, retriever, desc in spec:
        qe = RetrieverQueryEngine.from_args(retriever, llm=Settings.llm, response_mode="compact")
        out.append(QueryEngineTool.from_defaults(query_engine=qe, name=name, description=desc))
    return out


# --------------------------------------------------------------------------- workflow


class QueryEngineSelectionEvent(Event):
    """Result of selecting the query engine tools."""

    selected_query_engines: SelectorResult


class SynthesizeEvent(Event):
    """Event for synthesizing the response from different query engines."""

    result: List[RESPONSE_TYPE]
    selected_query_engines: SelectorResult


class RouterQueryEngineWorkflow(Workflow):
    @step
    async def selector(self, ctx: Context, ev: StartEvent) -> QueryEngineSelectionEvent:
        """Selects a single/ multiple query engines based on the query."""
        await ctx.store.set("query", ev.get("query"))
        await ctx.store.set("query_engine_tools", ev.get("query_engine_tools"))
        await ctx.store.set("summarizer", ev.get("summarizer"))
        selector = get_selector_from_llm(Settings.llm, is_multi=ev.get("select_multi"))
        metadata = [qe.metadata for qe in ev.get("query_engine_tools")]
        selected = await selector.aselect(metadata, ev.get("query"))
        return QueryEngineSelectionEvent(selected_query_engines=selected)

    @step
    async def generate_responses(self, ctx: Context, ev: QueryEngineSelectionEvent) -> SynthesizeEvent:
        """Generate the responses from the selected query engines."""
        query = await ctx.store.get("query", default=None)
        query_engines = [t.query_engine for t in await ctx.store.get("query_engine_tools")]
        selections = [s for s in ev.selected_query_engines.selections if 0 <= s.index < len(query_engines)][:3]
        if not selections:
            # The selector chose nothing: fall back to general knowledge rather than failing the run.
            fallback = SingleSelection(index=len(query_engines) - 1, reason="no engine selected; fallback")
            selections = [fallback]
            ev.selected_query_engines.selections = selections
        responses = await asyncio.gather(*(query_engines[s.index].aquery(query) for s in selections))
        return SynthesizeEvent(result=list(responses), selected_query_engines=ev.selected_query_engines)

    async def acombine_responses(self, summarizer: TreeSummarize, responses: List[RESPONSE_TYPE],
                                 query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Async combine multiple response from sub-engines."""
        source_nodes = []
        for r in responses:
            source_nodes.extend(r.source_nodes)
        summary = await summarizer.aget_response(query_bundle.query_str, [str(r) for r in responses])
        return Response(response=str(summary), source_nodes=source_nodes)

    @step
    async def synthesize_responses(self, ctx: Context, ev: SynthesizeEvent) -> StopEvent:
        """Synthesizes the responses from the generated responses."""
        response_generated = ev.result
        query = await ctx.store.get("query", default=None)
        summarizer = await ctx.store.get("summarizer")
        if len(response_generated) > 1:
            response = await self.acombine_responses(summarizer, response_generated, QueryBundle(query_str=query))
        else:
            response = response_generated[0]
        response.metadata = response.metadata or {}
        response.metadata["selector_result"] = ev.selected_query_engines
        return StopEvent(result=response)


_QE_TOOLS: list[QueryEngineTool] = []


def setup() -> None:
    pc.li_setup()
    _QE_TOOLS.extend(build_query_engine_tools())


async def arun(task: dict) -> str:
    pc.begin_task(AGENT.id, task)
    wf = RouterQueryEngineWorkflow(timeout=300, verbose=False)
    response = await wf.run(
        query=task["input"],
        query_engine_tools=_QE_TOOLS,
        summarizer=TreeSummarize(llm=Settings.llm),
        select_multi=True,
    )
    sel = response.metadata.get("selector_result")
    chosen = [(_QE_TOOLS[s.index].metadata.name, s.reason) for s in sel.selections] if sel else []
    return json.dumps({"answer": str(response), "routed_to": chosen}, default=str)
