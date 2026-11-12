"""Financial coordinator orchestrating four expert sub-agents through AgentTool.

Derived from google/adk-samples contrib/python/financial-advisor.
Adaptations: Google Search in data_analyst -> finance_quote / finance_history (synthetic market
data in the tools service) + web_search. The coordinator is interactive (asks for ticker, risk
attitude, horizon, execution preferences); a simulated user answers those questions from a short
investor profile so the four-step flow completes. Sub-agent outputs flow through output_key state.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

import adk_common as C
from peira_harness import AgentMeta

prompts = C.load_local(__file__)

AGENT = AgentMeta(
    id="adk-financial-advisor",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/contrib/python/financial-advisor",
    description="Financial coordinator that runs data, trading, execution and risk analyst sub-agents as tools, with a simulated investor answering its questions.",
    complexity="medium",
    model=C.HAIKU,
    tracing="openinference",
)

M = C.claude(C.HAIKU)

data_analyst_agent = LlmAgent(
    model=M,
    name="data_analyst_agent",
    description="Builds a market analysis report for a ticker from market data and web news.",
    instruction=prompts.DATA_ANALYST_PROMPT,
    output_key="market_data_analysis_output",
    tools=[C.AT.finance_quote, C.AT.finance_history, C.AT.web_search],
)

trading_analyst_agent = LlmAgent(
    model=M,
    name="trading_analyst_agent",
    description="Proposes trading strategies from the market analysis, risk attitude and investment period.",
    instruction=prompts.TRADING_ANALYST_PROMPT,
    output_key="proposed_trading_strategies_output",
)

execution_analyst_agent = LlmAgent(
    model=M,
    name="execution_analyst_agent",
    description="Defines a detailed execution plan for the proposed strategies.",
    instruction=prompts.EXECUTION_ANALYST_PROMPT,
    output_key="execution_plan_output",
)

risk_analyst_agent = LlmAgent(
    model=M,
    name="risk_analyst_agent",
    description="Evaluates the overall risk of the analysis, strategies and execution plan.",
    instruction=prompts.RISK_ANALYST_PROMPT,
    output_key="final_risk_assessment_output",
)

root_agent = LlmAgent(
    name="financial_coordinator",
    model=M,
    description=(
        "guide users through a structured process to receive financial "
        "advice by orchestrating a series of expert subagents. help them "
        "analyze a market ticker, develop trading strategies, define "
        "execution plans, and evaluate the overall risk."
    ),
    instruction=prompts.FINANCIAL_COORDINATOR_PROMPT
    + "\nIf the user already gave some inputs (ticker, risk attitude, period, preferences) in their message, do not ask for them again.\n",
    output_key="financial_coordinator_output",
    tools=[
        AgentTool(agent=data_analyst_agent),
        AgentTool(agent=trading_analyst_agent),
        AgentTool(agent=execution_analyst_agent),
        AgentTool(agent=risk_analyst_agent),
    ],
)


def _investor(turn: int, last_text: str, state: dict) -> str | None:
    """Simulated investor. Answers the coordinator's questions until the risk step is done."""
    if state.get("final_risk_assessment_output"):
        return None
    if not C.asks_question(last_text):
        return "Please continue with the next step."
    low = last_text.lower()
    if "ticker" in low and not state.get("market_data_analysis_output"):
        return "Use the ticker I mentioned. If I didn't give one, analyze LMNC."
    return (
        "My risk attitude is moderate and my investment period is long-term (3-5 years). "
        "No preferred broker; limit orders are fine. Please run all the remaining steps now without asking again."
    )


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(root_agent, task, app_name="financial_advisor", auto_user=_investor, max_turns=4, max_llm_calls=25)
