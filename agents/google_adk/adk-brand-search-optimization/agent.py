"""Brand search optimization: root coordinator transferring through keyword finding -> search
results -> comparison (generator + critic) sub-agent tree.

Derived from google/adk-samples contrib/python/brand-search-optimization.
Adaptations: BigQuery catalog -> tools-service product catalog (products_search / inventory_check
for brand Lumen & Co); Gemini Computer Use browsing of Google Shopping -> web_search + fetch_url.
The agent tree and the transfer_to_agent hand-offs are kept as in the source.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

import adk_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

prompts = C.load_local(__file__)

AGENT = AgentMeta(
    id="adk-brand-search-optimization",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/contrib/python/brand-search-optimization",
    description="Coordinator hands off via transfer_to_agent from catalog keyword mining to competitor search to a generator/critic title-comparison team.",
    complexity="medium",
    model=C.HAIKU,
    tracing="openinference",
)

M = C.claude(C.HAIKU)


def get_product_details_for_brand(brand: str, product_line: str = "", limit: int = 5) -> dict:
    """Retrieves product details (title, category, price, stock) for a brand from the product catalog.

    Args:
        brand: The brand name to query in the product catalog (the catalog holds Lumen & Co products).
        product_line: Optional product line, category or keyword (e.g. "cameras", "Smart Lock").
        limit: Maximum number of product records to retrieve (default: 5).

    Returns:
        A dict with the brand and structured product records.
    """
    if not brand or "lumen" not in brand.lower():
        return {"brand": brand, "products": [], "total_count": 0,
                "note": "This catalog only contains the brand 'Lumen & Co'."}
    cats = ("hubs", "cameras", "sensors", "lighting", "locks", "climate", "plugs", "accessories")
    pl = product_line.strip().lower()
    if pl in cats:
        res = T.products_search(category=pl, limit=max(1, min(limit, 10)))
    else:
        res = T.products_search(query=product_line, limit=max(1, min(limit, 10)))
    if "error" in res:
        return res
    return {"brand": "Lumen & Co", "product_line": product_line, "products": res.get("products", []),
            "total_count": res.get("total_matches", 0), "categories": res.get("categories", [])}


keyword_finding_agent = LlmAgent(
    model=M,
    name="keyword_finding_agent",
    description="Extracts and ranks high-intent shopper search keywords from brand catalog data.",
    instruction=prompts.KEYWORD_FINDING_AGENT_PROMPT,
    tools=[C.async_tool(get_product_details_for_brand), C.AT.inventory_check],
    output_key="extracted_keywords",
)

search_results_agent = LlmAgent(
    model=M,
    name="search_results_agent",
    description="Searches the web like a shopper to observe top competitor product titles for a keyword.",
    instruction=prompts.SEARCH_RESULT_AGENT_PROMPT,
    tools=[C.AT.web_search, C.AT.fetch_url],
    output_key="observed_search_results",
)

comparison_generator_agent = LlmAgent(
    model=M,
    name="comparison_generator_agent",
    description="Generates detailed title comparison and search optimization proposals.",
    instruction=prompts.COMPARISON_AGENT_PROMPT,
    output_key="comparison_proposals",
)

comparison_critic_agent = LlmAgent(
    model=M,
    name="comparison_critic_agent",
    description="Critiques title optimization proposals for quality, brand accuracy, and search intent alignment.",
    instruction=prompts.COMPARISON_CRITIC_AGENT_PROMPT,
    output_key="critique_feedback",
)

comparison_root_agent = LlmAgent(
    model=M,
    name="comparison_root_agent",
    description="Coordinates comparison generation and critique to produce the final optimization report.",
    instruction=prompts.COMPARISON_ROOT_AGENT_PROMPT,
    sub_agents=[comparison_generator_agent, comparison_critic_agent],
    output_key="final_optimization_report",
)

root_agent = LlmAgent(
    model=M,
    name="brand_search_optimization",
    description="Helps e-commerce brands optimize product titles for retail search.",
    instruction=prompts.ROOT_PROMPT,
    sub_agents=[keyword_finding_agent, search_results_agent, comparison_root_agent],
)


def _brand_manager(turn: int, last_text: str, state: dict) -> str | None:
    """Simulated brand manager: answers the brand question, then asks to continue the workflow."""
    if state.get("final_optimization_report"):
        return None
    if not state.get("extracted_keywords"):
        return "The brand is Lumen & Co. Use the product line from my request, or cameras if I didn't name one."
    return "Please continue the workflow through to the final title optimization report."


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(root_agent, task, app_name="brand_search_optimization", auto_user=_brand_manager, max_turns=3, max_llm_calls=20)
