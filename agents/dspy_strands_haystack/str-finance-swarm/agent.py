"""Strands hybrid orchestrator + collaborative Swarm for stock analysis.

Derived from strands-agents/samples python/04-industry-use-cases/finance/finance-assistant-swarm-agent
(finance_assistant_swarm.py) and the Swarm tutorial (python/01-learn/11-swarm/swarm.ipynb).
Kept: the "senior research director" orchestrator with its WORKFLOW / CRITICAL RULES / REPORT STRUCTURE prompt,
the tools get_real_stock_data and analyze_company_with_collaborative_swarm, the 3-agent Swarm
(company_strategist -> financial_analyst -> market_analyst) with its prompts, max_handoffs=3, max_iterations=3,
and the ticker extraction helper.
Changed: yfinance -> tools-service finance quotes/history (synthetic prices, 50 fictional + 6 real tickers);
company news -> keyless web search; Bedrock Opus -> claude-sonnet-5-5 through the proxy; tools are async so
the swarm's spans stay inside the task trace.
"""

import json
import math
import re
from typing import Any

import dsh_common
from peira_harness import AgentMeta, tools
from strands import Agent, tool
from strands.multiagent import Swarm

AGENT = AgentMeta(
    id="str-finance-swarm",
    framework="strands",
    source_url="https://github.com/strands-agents/samples/tree/main/python/04-industry-use-cases/finance/finance-assistant-swarm-agent",
    description="Research-director agent that pulls a quote, runs a 3-agent Strands Swarm with handoffs, and writes an investment report.",
    complexity="complex",
    model="claude-sonnet-5-5",
    tracing="native-otel",
)


def _j(x) -> str:
    return json.dumps(x, default=str)[:6000]


@tool
def get_real_stock_data(ticker: str) -> dict[str, Any]:
    """Get accurate stock data outside the swarm"""
    q = tools.finance_quote(symbol=ticker)
    if "error" in q:
        return {"status": "error", "message": q["error"]}
    return {"status": "success", **q}


@tool
def get_company_info(ticker: str) -> str:
    """Get company overview: name, sector, latest price, and a short public profile if one exists."""
    q = tools.finance_quote(symbol=ticker)
    if "error" in q:
        return _j({"status": "error", "message": q["error"]})
    name = q.get("name") or q.get("company") or ticker
    profile = tools.web_search(query=f"{name} company profile business model", max_results=3)
    return _j({"status": "success", "quote": q, "web_profile": profile})


@tool
def get_financial_metrics(ticker: str) -> str:
    """Get price-based financial metrics: 1y return, annualized volatility, max drawdown, 52-week range."""
    h = tools.finance_history(symbol=ticker, start_date="2025-11-13", end_date="2026-11-13", interval="weekly")
    if "error" in h:
        return _j({"status": "error", "message": h["error"]})
    rows = h.get("bars") or []
    closes = [r.get("close") for r in rows if isinstance(r, dict) and r.get("close") is not None]
    if len(closes) < 3:
        return _j({"status": "error", "message": "not enough price history", "raw": h})
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(52)
    peak, mdd = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        mdd = min(mdd, c / peak - 1)
    return _j({
        "status": "success", "ticker": ticker.upper(), "points": len(closes),
        "return_1y_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
        "annualized_volatility_pct": round(vol * 100, 2), "max_drawdown_pct": round(mdd * 100, 2),
        "high_52w": max(closes), "low_52w": min(closes), "last_close": closes[-1],
    })


@tool
def get_stock_news(ticker: str) -> str:
    """Get recent news headlines about the company behind a ticker."""
    name = tools.finance_quote(symbol=ticker).get("name") or ticker
    return _j(tools.web_search(query=f"{name} {ticker} stock news", max_results=5))


def _extract_ticker(query: str) -> str:
    """Extract a stock ticker symbol from a query string.

    Handles cases like 'ALK', 'Analyze ALK', 'ALK (Alaska Air Group)', etc.
    """
    stripped = query.strip().upper()
    if re.fullmatch(r"[A-Z]{1,5}", stripped):
        return stripped
    match = re.search(r"\b([A-Z]{1,5})\b", query.upper())
    if match:
        return match.group(1)
    return stripped[:5] if stripped else "UNKNOWN"


@tool
async def analyze_company_with_collaborative_swarm(ticker: str) -> dict[str, Any]:
    """Run a collaborative swarm analysis for a stock ticker symbol (e.g. 'ALK', 'AAPL', 'MSFT').
    The ticker parameter must be a valid stock ticker symbol, not a full sentence."""
    try:
        ticker = _extract_ticker(ticker)

        company_strategist = Agent(
            name="company_strategist",
            system_prompt=f"Analyze {ticker} business model. Use get_company_info with ticker '{ticker}', then hand off to financial_analyst.",
            model=dsh_common.strands_model(AGENT.model),
            tools=[get_company_info],
            callback_handler=None,
        )
        financial_analyst = Agent(
            name="financial_analyst",
            system_prompt=f"Build on the company insights for {ticker}. Use get_financial_metrics with ticker '{ticker}', then hand off to market_analyst.",
            model=dsh_common.strands_model(AGENT.model),
            tools=[get_financial_metrics],
            callback_handler=None,
        )
        market_analyst = Agent(
            name="market_analyst",
            system_prompt=f"Synthesize all insights for {ticker}. Use get_stock_news with ticker '{ticker}' for final recommendation.",
            model=dsh_common.strands_model(AGENT.model),
            tools=[get_stock_news],
            callback_handler=None,
        )

        swarm = Swarm(
            [company_strategist, financial_analyst, market_analyst],
            max_handoffs=3,
            max_iterations=3,
            execution_timeout=300.0,
            node_timeout=120.0,
        )
        result = await swarm.invoke_async(f"Analyze {ticker}")

        final = None
        for node in reversed(result.node_history):
            r = result.results.get(node.node_id)
            if r is not None and r.result is not None:
                final = str(r.result)
                break
        return {
            "status": "success",
            "collaborative_analysis": final,
            "collaboration_path": [node.node_id for node in result.node_history],
        }
    except Exception as e:  # noqa: BLE001 - the source returns errors as data
        return {"status": "error", "collaborative_analysis": f"Analysis failed: {str(e)}"}


ORCHESTRATOR_PROMPT = """You are a senior research director.

        WORKFLOW:
        1. Get real stock data using get_real_stock_data with the ticker symbol
        2. Get ONE collaborative analysis using analyze_company_with_collaborative_swarm — pass ONLY the ticker symbol (e.g. "ALK"), NOT a full sentence
        3. Synthesize the findings into deep strategic insights

        CRITICAL RULES:
        - When calling analyze_company_with_collaborative_swarm, pass ONLY the stock ticker symbol (e.g. "ALK", "AAPL"), never a full sentence
        - DO NOT call analyze_company_with_collaborative_swarm multiple times
        - Focus on synthesis and strategic conclusions
        - Always prominently display the current stock price
        - Provide deep insights that demonstrate collaborative agent value
        - Prices in this environment are synthetic; say so if the user asks about real-world accuracy

        REPORT STRUCTURE:
        1. Executive Summary (current price + key thesis)
        2. Strategic Business Analysis (from collaborative insights)
        3. Financial Health Assessment (integrated metrics)
        4. Market Sentiment Analysis (news + trends)
        5. Investment Recommendation (buy/hold/sell with rationale)"""


def create_initial_messages() -> list[dict]:
    return [
        {"role": "user", "content": [{"text": "Hello, I need help analyzing company stocks."}]},
        {"role": "assistant", "content": [{"text": "I'm ready to provide comprehensive stock analysis using real-time data and collaborative multi-agent analysis. Please provide a company name or ticker."}]},
    ]


def setup() -> None:
    dsh_common.setup_strands()


async def arun(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    orchestration_agent = Agent(
        name="research_director",
        system_prompt=ORCHESTRATOR_PROMPT,
        model=dsh_common.strands_model(AGENT.model),
        tools=[get_real_stock_data, analyze_company_with_collaborative_swarm],
        messages=create_initial_messages(),
        callback_handler=None,
    )
    response = await orchestration_agent.invoke_async(task["input"])
    return str(response)
