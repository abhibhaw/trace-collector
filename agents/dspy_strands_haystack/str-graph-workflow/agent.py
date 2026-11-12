r"""Strands Graph multi-agent workflow: conditional routing, then a parallel fan-out and a risk join.

Derived from strands-agents/samples python/01-learn/12-graph/graph.ipynb, which shows three topologies:
basic chain, parallel processing (finance_expert -> tech_expert + market_expert -> risk_analyst), and
branching with conditions (classifier -> technical_report | business_report via is_technical/is_business).
This agent combines the two latter graphs into one GraphBuilder graph, keeping the notebook's agent names,
system prompts and condition functions:

    classifier --is_business--> finance_expert --> tech_expert   --\
                                              \--> market_expert --> risk_analyst
    classifier --is_technical-> technical_report ---------------------/

Changed: Bedrock -> Anthropic via the proxy; experts get tools (web search, Lumen & Co catalog,
finance quotes, sandboxed Python for ROI math); execution is bounded (max_node_executions, timeouts).
"""

import json

import dsh_common
from peira_harness import AgentMeta, tools
from strands import Agent, tool
from strands.multiagent import GraphBuilder

AGENT = AgentMeta(
    id="str-graph-workflow",
    framework="strands",
    source_url="https://github.com/strands-agents/samples/blob/main/python/01-learn/12-graph/graph.ipynb",
    description="Strands GraphBuilder workflow: a classifier routes to a technical branch or a finance->(tech, market) fan-out joined by a risk analyst.",
    complexity="complex",
    model="claude-haiku-5-5",
    tracing="native-otel",
)


def _j(x) -> str:
    return json.dumps(x, default=str)[:5000]


@tool
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and snippets."""
    return _j(tools.web_search(query=query, max_results=5))


@tool
def python_calc(code: str) -> str:
    """Run a short Python snippet in a sandbox and return stdout. Use print() for results (ROI, NPV, payback)."""
    return _j(tools.python_exec(code=code, timeout_s=20))


@tool
def finance_quote(symbol: str) -> str:
    """Get the latest (synthetic) stock quote for a ticker symbol."""
    return _j(tools.finance_quote(symbol=symbol))


@tool
def products_search(query: str = "", category: str = "") -> str:
    """Search the Lumen & Co product catalog (smart-home devices) by keyword or category."""
    return _j(tools.products_search(query=query, category=category, limit=10))


def is_technical(state):
    classifier_result = state.results.get("classifier")
    if not classifier_result:
        return False
    result_text = str(classifier_result.result)
    return "technical" in result_text.lower()


def is_business(state):
    classifier_result = state.results.get("classifier")
    if not classifier_result:
        return False
    result_text = str(classifier_result.result)
    return "business" in result_text.lower()


SHORT = " Keep it under 250 words, no follow-up questions."


def build_graph():
    m = lambda: dsh_common.strands_model(AGENT.model)  # noqa: E731 - one client per node
    classifier = Agent(name="classifier", model=m(), callback_handler=None, system_prompt=(
        "You are an agent responsible for classification of the report request, return only Technical or Business classification."))
    technical_report = Agent(name="technical_expert", model=m(), callback_handler=None, tools=[web_search, products_search], system_prompt=(
        "You are a technical expert that focuses on providing short summary from technical perspective." + SHORT))
    financial_advisor = Agent(name="financial_advisor", model=m(), callback_handler=None, tools=[python_calc, finance_quote], system_prompt=(
        "You are a financial advisor focused on cost-benefit analysis, budget implications, and ROI calculations. "
        "Engage with other experts to build comprehensive financial perspectives. Use python_calc for any arithmetic." + SHORT))
    technical_architect = Agent(name="technical_architect", model=m(), callback_handler=None, tools=[web_search], system_prompt=(
        "You are a technical architect who evaluates feasibility, implementation challenges, and technical risks. "
        "Collaborate with other experts to ensure technical viability." + SHORT))
    market_researcher = Agent(name="market_researcher", model=m(), callback_handler=None, tools=[web_search, products_search], system_prompt=(
        "You are a market researcher who analyzes market conditions, user needs, and competitive landscape. "
        "Work with other experts to validate market opportunities." + SHORT))
    risk_analyst = Agent(name="risk_analyst", model=m(), callback_handler=None, system_prompt=(
        "You are a risk analyst who identifies potential risks, mitigation strategies, and compliance issues. "
        "Collaborate with other experts to ensure comprehensive risk assessment. "
        "Finish with a clear recommendation and the top 3 risks." + SHORT))

    builder = GraphBuilder()
    builder.add_node(classifier, "classifier")
    builder.add_node(technical_report, "technical_report")
    builder.add_node(financial_advisor, "finance_expert")
    builder.add_node(technical_architect, "tech_expert")
    builder.add_node(market_researcher, "market_expert")
    builder.add_node(risk_analyst, "risk_analyst")

    builder.add_edge("classifier", "technical_report", condition=is_technical)
    builder.add_edge("classifier", "finance_expert", condition=is_business)
    builder.add_edge("finance_expert", "tech_expert")
    builder.add_edge("finance_expert", "market_expert")
    builder.add_edge("tech_expert", "risk_analyst")
    builder.add_edge("market_expert", "risk_analyst")
    builder.add_edge("technical_report", "risk_analyst")

    builder.set_entry_point("classifier")
    builder.set_max_node_executions(8)
    builder.set_execution_timeout(400)
    builder.set_node_timeout(150)
    return builder.build()


def setup() -> None:
    dsh_common.setup_strands()


async def arun(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    graph = build_graph()
    result = await graph.invoke_async(task["input"])
    order = [n.node_id for n in result.execution_order]
    final = None
    for node_id in reversed(order):
        r = result.results.get(node_id)
        if r is not None and r.result is not None:
            final = str(r.result)
            break
    if final is None:
        raise RuntimeError(f"graph produced no result (status={result.status})")
    return f"{final}\n\n[path: {' -> '.join(order)}]"
