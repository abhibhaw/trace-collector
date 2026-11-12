"""Strands "agents as tools": an orchestrator routes to specialist agents wrapped as @tool functions.

Derived from strands-agents/samples python/01-learn/10-agents-as-tools/agent-as-tools.ipynb.
Kept: the orchestrator system prompt and routing rules, the three specialists (research_assistant,
product_recommendation_assistant, trip_planning_assistant) with their prompts and docstrings, a file-writing
tool, and the pattern of building a fresh specialist Agent inside each tool call.
Changed: Bedrock -> Anthropic model via the proxy; specialists get real tools (web search/fetch/Wikipedia,
the Lumen & Co product catalog and inventory, flight and hotel search); `strands_tools.file_write` is
replaced by an in-memory `file_write`; tools are async so OTel context stays on the task's trace.
"""

import json

import dsh_common
from peira_harness import AgentMeta, tools
from strands import Agent, tool

AGENT = AgentMeta(
    id="str-agents-as-tools",
    framework="strands",
    source_url="https://github.com/strands-agents/samples/blob/main/python/01-learn/10-agents-as-tools/agent-as-tools.ipynb",
    description="Strands orchestrator that delegates to research, product-recommendation and trip-planning sub-agents exposed as tools.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="native-otel",
)

RESEARCH_ASSISTANT_PROMPT = """You are a specialized research assistant. Focus only on providing
factual, well-sourced information in response to research questions.
Always cite your sources when possible."""

MAIN_SYSTEM_PROMPT = """
You are an assistant that routes queries to specialized agents:
- For research questions and factual information → Use the research_assistant tool
- For product recommendations and shopping advice → Use the product_recommendation_assistant tool
- For travel planning and itineraries → Use the trip_planning_assistant tool
- For simple questions not requiring specialized knowledge → Answer directly

Always select the most appropriate tool based on the user's query.
If the user asks to save or write the final response, use the file_write tool.
Today's date is 2026-11-13. Product questions are about Lumen & Co smart-home devices.
"""


def _j(x) -> str:
    return json.dumps(x, default=str)[:6000]


# ------------------------------------------------------------- specialist tools (shared tools service)
@tool
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and snippets."""
    return _j(tools.web_search(query=query, max_results=5))


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page and return its readable text."""
    return _j(tools.fetch_url(url=url, max_chars=5000))


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return matching page titles and summaries."""
    return _j(tools.wikipedia_search(query=query, max_results=5))


@tool
def products_search(query: str = "", category: str = "", max_price_usd: float = 0.0) -> str:
    """Search the Lumen & Co product catalog by keyword, category, and maximum price (0 = no limit)."""
    return _j(tools.products_search(query=query, category=category, max_price_usd=max_price_usd, limit=10))


@tool
def inventory_check(sku: str) -> str:
    """Check warehouse stock for a product SKU (like SKU-2001)."""
    return _j(tools.inventory_check(sku=sku))


@tool
def search_flights(origin: str, destination: str, date: str, cabin: str = "economy") -> str:
    """Search flights between two supported cities on a date (YYYY-MM-DD)."""
    return _j(tools.travel_search_flights(origin=origin, destination=destination, date=date, cabin=cabin))


@tool
def search_hotels(city: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Search hotels in a supported city for a date range (YYYY-MM-DD)."""
    return _j(tools.travel_search_hotels(city=city, check_in=check_in, check_out=check_out, guests=guests))


# ------------------------------------------------------------- agents wrapped as tools (from the notebook)
@tool
async def research_assistant(query: str) -> str:
    """
    Process and respond to research-related queries.

    Args:
        query: A research question requiring factual information

    Returns:
        A detailed research answer with citations
    """
    try:
        research_agent = Agent(
            name="research_assistant",
            model=dsh_common.strands_model(AGENT.model),
            system_prompt=RESEARCH_ASSISTANT_PROMPT,
            tools=[web_search, fetch_url, wikipedia_search],
            callback_handler=None,
        )
        response = await research_agent.invoke_async(query)
        return str(response)
    except Exception as e:  # noqa: BLE001 - mirrors the notebook's error-to-text handling
        return f"Error in research assistant: {str(e)}"


@tool
async def product_recommendation_assistant(query: str) -> str:
    """
    Handle product recommendation queries by suggesting appropriate products.

    Args:
        query: A product inquiry with user preferences

    Returns:
        Personalized product recommendations with reasoning
    """
    try:
        product_agent = Agent(
            name="product_recommendation_assistant",
            model=dsh_common.strands_model(AGENT.model),
            system_prompt="""You are a specialized product recommendation assistant for Lumen & Co smart-home devices.
            Provide personalized product suggestions based on user preferences. Use the catalog and inventory tools;
            never recommend discontinued or out-of-stock items without saying so. Always cite SKUs.""",
            tools=[products_search, inventory_check],
            callback_handler=None,
        )
        response = await product_agent.invoke_async(query)
        return str(response)
    except Exception as e:  # noqa: BLE001
        return f"Error in product recommendation: {str(e)}"


@tool
async def trip_planning_assistant(query: str) -> str:
    """
    Create travel itineraries and provide travel advice.

    Args:
        query: A travel planning request with destination and preferences

    Returns:
        A detailed travel itinerary or travel advice
    """
    try:
        travel_agent = Agent(
            name="trip_planning_assistant",
            model=dsh_common.strands_model(AGENT.model),
            system_prompt="""You are a specialized travel planning assistant.
            Create detailed travel itineraries based on user preferences. Use the flight and hotel search tools
            for concrete options and prices. Today's date is 2026-11-13.""",
            tools=[search_flights, search_hotels, wikipedia_search],
            callback_handler=None,
        )
        response = await travel_agent.invoke_async(query)
        return str(response)
    except Exception as e:  # noqa: BLE001
        return f"Error in trip planning: {str(e)}"


def make_file_write(files: dict[str, str]):
    @tool
    def file_write(path: str, content: str) -> str:
        """Write content to a file at the given path (in the agent's sandboxed workspace)."""
        files[path] = content
        return f"Wrote {len(content)} characters to {path}"

    return file_write


def setup() -> None:
    dsh_common.setup_strands()


async def arun(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    files: dict[str, str] = {}
    orchestrator = Agent(
        name="orchestrator",
        model=dsh_common.strands_model(AGENT.model),
        system_prompt=MAIN_SYSTEM_PROMPT,
        tools=[research_assistant, product_recommendation_assistant, trip_planning_assistant, make_file_write(files)],
        callback_handler=None,
    )
    response = await orchestrator.invoke_async(task["input"])
    out = str(response)
    if files:
        out += "\n\n[files written: " + ", ".join(files) + "]"
    return out
