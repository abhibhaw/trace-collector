"""Haystack agentic RAG with conditional fallback to web search.

Derived from deepset-ai/haystack-cookbook notebooks/llama32_agentic_rag.ipynb (and the tutorial it cites,
"Building Fallbacks to Websearch with Conditional Routing").
Kept: the Seven Wonders knowledge base (bilgeyucel/seven-wonders, 151 docs, bundled as seven_wonders.json),
the two prompt templates ('no_answer' / "FROM THE KNOWLEDGE BASE:" / "FROM THE WEB:"), the ConditionalRouter
routes, the BranchJoiner loop that sends the web-search prompt back into the same LLM, and the exact
component graph and connections.
Changed: Llama 3.2 (transformers) -> Claude via the proxy; sentence-transformers embedding retrieval ->
InMemoryBM25Retriever (no torch in the pod); DuckduckgoApiWebSearch -> a small component over the
tools-service web_search.
"""

import json
from pathlib import Path

import dsh_common
from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import BranchJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator  # noqa: F401 - registered before instrumenting
from peira_harness import AgentMeta, tools
from fallback_rag_components import WebSearch  # custom components live in an importable module (see README)

AGENT = AgentMeta(
    id="hay-fallback-rag",
    framework="haystack",
    source_url="https://github.com/deepset-ai/haystack-cookbook/blob/main/notebooks/llama32_agentic_rag.ipynb",
    description="Haystack pipeline: BM25 RAG over a Seven Wonders KB; a ConditionalRouter falls back to web search when the LLM says no_answer.",
    complexity="simple",
    model="claude-haiku-5-5",
    tracing="openinference",
)


prompt_template = [
    ChatMessage.from_user(
        """Answer the following query given the documents.
If the answer is not contained within the documents reply with 'no_answer'.
If the answer is contained within the documents, start the answer with "FROM THE KNOWLEDGE BASE: ".

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Query: {{query}}"""
    )
]

prompt_template_after_websearch = [
    ChatMessage.from_user(
        """Answer the following query given the documents retrieved from the web.
Start the answer with "FROM THE WEB: ".

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Query: {{query}}"""
    )
]

routes = [
    {
        "condition": "{{'no_answer' in replies[0].text}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0].text}}",
        "output": "{{replies[0].text}}",
        "output_name": "answer",
        "output_type": str,
    },
]

_document_store: InMemoryDocumentStore | None = None


def _store() -> InMemoryDocumentStore:
    global _document_store
    if _document_store is None:
        rows = json.loads((Path(__file__).parent / "seven_wonders.json").read_text())
        store = InMemoryDocumentStore()
        store.write_documents([Document(content=r["content"], meta={"url": r["url"]}) for r in rows])
        _document_store = store
    return _document_store


def build_pipeline() -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(_store(), top_k=5))
    pipe.add_component("prompt_builder", ChatPromptBuilder(template=prompt_template, required_variables="*"))
    pipe.add_component("prompt_joiner", BranchJoiner(list[ChatMessage]))
    pipe.add_component("llm", dsh_common.haystack_generator(AGENT.model))
    pipe.add_component("router", ConditionalRouter(routes))
    pipe.add_component("websearch", WebSearch(top_k=5))
    pipe.add_component(
        "prompt_builder_after_websearch",
        ChatPromptBuilder(template=prompt_template_after_websearch, required_variables="*"),
    )

    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "prompt_joiner.value")
    pipe.connect("prompt_joiner.value", "llm.messages")
    pipe.connect("llm.replies", "router.replies")
    pipe.connect("router.go_to_websearch", "websearch.query")
    pipe.connect("router.go_to_websearch", "prompt_builder_after_websearch.query")
    pipe.connect("websearch.documents", "prompt_builder_after_websearch.documents")
    pipe.connect("prompt_builder_after_websearch.prompt", "prompt_joiner.value")
    return pipe


def setup() -> None:
    _store()
    dsh_common.setup_haystack()


def run(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    query = task["input"]
    result = build_pipeline().run({"retriever": {"query": query}, "prompt_builder": {"query": query}, "router": {"query": query}})
    if "router" in result and "answer" in result["router"]:
        return result["router"]["answer"]
    raise RuntimeError(f"pipeline ended without an answer: {list(result)}")
