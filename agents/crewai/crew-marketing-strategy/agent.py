"""Marketing strategy crew, adapted from crewAI-examples/crews/marketing_strategy (marketing_posts).

Shape kept from the source: lead market analyst, chief marketing strategist, and creative content creator with the
source's pydantic models (MarketStrategy, CampaignIdea, Copy) as structured task outputs. The source ships a
commented-out `Process.hierarchical` option and an unused `chief_creative_director` agent; this adaptation turns
both on: the chief creative director (Sonnet) is the manager agent that plans, delegates every task to a coworker,
and reviews the result.
Adapted: Serper/ScrapeWebsite are replaced with web search, URL fetch, product catalog and knowledge base.
To bound cost the five source tasks are merged into three (research, strategy, campaign+copy).
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from typing import List

from crewai import Crew, Process, Task
from pydantic import BaseModel, Field

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-marketing-strategy",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/crews/marketing_strategy",
    description="Hierarchical crew where a Sonnet creative-director manager delegates research, strategy, and copy tasks to three specialists with pydantic outputs.",
    complexity="complex",
    model=cc.SONNET,
    tracing="openinference",
)


class MarketStrategy(BaseModel):
    """Market strategy model"""

    name: str = Field(..., description="Name of the market strategy")
    tactics: List[str] = Field(..., description="List of tactics to be used in the market strategy")
    channels: List[str] = Field(..., description="List of channels to be used in the market strategy")
    KPIs: List[str] = Field(..., description="List of KPIs to be used in the market strategy")


class CampaignIdea(BaseModel):
    """Campaign idea model"""

    name: str = Field(..., description="Name of the campaign idea")
    description: str = Field(..., description="Description of the campaign idea")
    audience: str = Field(..., description="Audience of the campaign idea")
    channel: str = Field(..., description="Channel of the campaign idea")


class Copy(BaseModel):
    """Copy model"""

    title: str = Field(..., description="Title of the copy")
    body: str = Field(..., description="Body of the copy")


class CampaignPackage(BaseModel):
    ideas: List[CampaignIdea] = Field(..., description="2-3 campaign ideas")
    copies: List[Copy] = Field(..., description="One marketing copy per campaign idea")


def setup() -> None:
    cc.setup_tracing()


def build_crew(run_id: str) -> Crew:
    analyst = cc.mk_agent(
        role="Lead Market Analyst",
        goal="Conduct amazing analysis of the products and competitors, providing in-depth insights to guide marketing "
        "strategies.",
        backstory="As the Lead Market Analyst at a premier digital marketing firm, you specialize in dissecting online "
        "business landscapes. Your reports are concise: at most 400 words, bullet points over prose. " + cc.WORLD,
        tools=cc.make_tools([T.web_search, T.fetch_url, T.products_search, T.kb_search], run_id),
        max_iter=3,
    )
    strategist = cc.mk_agent(
        role="Chief Marketing Strategist",
        goal="Synthesize amazing insights from product analysis to formulate incredible marketing strategies.",
        backstory="You are the Chief Marketing Strategist at a leading digital marketing agency, known for crafting "
        "bespoke strategies that drive success. " + cc.WORLD,
        tools=cc.make_tools([T.web_search], run_id),
        max_iter=3,
    )
    creator = cc.mk_agent(
        role="Creative Content Creator",
        goal="Develop compelling and innovative content for social media campaigns, with a focus on creating high-impact "
        "ad copies.",
        backstory="As a Creative Content Creator at a top-tier digital marketing agency, you excel in crafting narratives "
        "that resonate with audiences. Your expertise lies in turning marketing strategies into engaging stories and "
        "visual content that capture attention and inspire action. " + cc.WORLD,
        max_iter=2,
    )
    director = cc.mk_agent(
        role="Chief Creative Director",
        goal="Oversee the work done by your team to make sure it is the best possible and aligned with the product goals, "
        "review, approve, ask clarifying questions or delegate follow-up work if necessary.",
        backstory="You are the Chief Content Officer at a leading digital marketing agency specializing in product "
        "branding. You ensure your team crafts the best possible content for the customer. Delegate each task to the "
        "single best coworker with full context, once. Accept a usable answer as it is: never ask a coworker to "
        "rewrite or continue a report because of its length. Keep your own final answers short. " + cc.WORLD,
        model=cc.SONNET,
        allow_delegation=True,
        max_iter=3,
    )
    project = "Project brief from the customer:\n{request}"
    research = Task(
        description="Conduct a thorough research about the customer and competitors in the context of the project. "
        "Use the product catalog and knowledge base for Lumen & Co products and the web for the market.\n\n" + project,
        expected_output="A concise report (at most 400 words) on the customer, their customers and competitors: "
        "demographics, preferences, market positioning and audience engagement.",
    )
    strategy = Task(
        description="Understand the project details and the target audience, then formulate a comprehensive marketing "
        "strategy for the project, using the insights from the research.\n\n" + project,
        expected_output="A marketing strategy with name, tactics, channels and KPIs (at most 6 items each).",
        context=[research],
        output_pydantic=MarketStrategy,
    )
    campaign = Task(
        description="Develop 2-3 creative marketing campaign ideas aligned with the strategy, then create one marketing "
        "copy for each idea. Ensure the copies are compelling, clear, and tailored to the target audience. Do not "
        "promise prices, discounts or features the research did not confirm.\n\n" + project,
        expected_output="Campaign ideas (name, description, audience, channel) and one copy (title, body of at most "
        "120 words) per idea.",
        context=[research, strategy],
        output_pydantic=CampaignPackage,
    )
    return Crew(
        agents=[analyst, strategist, creator],
        tasks=[research, strategy, campaign],
        process=Process.hierarchical,
        manager_agent=director,
        verbose=False,
    )


def run(task: dict) -> str:
    crew = build_crew(cc.run_id_for(AGENT.id, task))
    out = crew.kickoff(inputs={"request": task["input"]})
    strategy = out.tasks_output[1]
    return f"## Strategy\n{cc.result_text(strategy)}\n\n## Campaign\n{cc.result_text(out)}"
