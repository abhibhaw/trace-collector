"""Meeting preparation crew, adapted from crewAI-examples/crews/prep-for-a-meeting.

Shape kept from the source: research specialist and industry analyst run in parallel (async_execution=True),
then the meeting strategy advisor and the briefing coordinator consume both via task context.
Adapted: Exa search is replaced by web search plus the internal CRM, orders, tickets, HR directory and calendar,
so participants can be Lumen & Co customers, business accounts, or colleagues.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from crewai import Crew, Process, Task

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-meeting-prep",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/crews/prep-for-a-meeting",
    description="Four-agent crew with two parallel research tasks that prepares a meeting briefing from CRM, calendar, and web data.",
    complexity="simple",
    model=cc.HAIKU,
    tracing="openinference",
)


def setup() -> None:
    cc.setup_tracing()


def build_crew(run_id: str) -> Crew:
    researcher = cc.mk_agent(
        role="Research Specialist",
        goal="Conduct thorough research on people and companies involved in the meeting",
        backstory="As a Research Specialist, your mission is to uncover detailed information about the individuals and "
        "entities participating in the meeting: CRM profile, recent orders, open support tickets, colleagues' roles, "
        "and their calendars. Your insights will lay the groundwork for strategic meeting preparation. " + cc.WORLD,
        tools=cc.make_tools([T.crm_search_customers, T.crm_get_customer, T.orders_list, T.tickets_search,
                             T.hr_search_employees, T.calendar_get_events], run_id),
        max_iter=5,
    )
    industry = cc.mk_agent(
        role="Industry Analyst",
        goal="Analyze the current industry trends, challenges, and opportunities",
        backstory="As an Industry Analyst, your analysis will identify key trends, challenges facing the industry, and "
        "potential opportunities that could be leveraged during the meeting for strategic advantage. " + cc.WORLD,
        tools=cc.make_tools([T.web_search, T.wikipedia_search], run_id),
        max_iter=3,
    )
    strategist = cc.mk_agent(
        role="Meeting Strategy Advisor",
        goal="Develop talking points, questions, and strategic angles for the meeting",
        backstory="As a Strategy Advisor, your expertise will guide the development of talking points, insightful "
        "questions, and strategic angles to ensure the meeting's objectives are achieved. " + cc.WORLD,
        tools=cc.make_tools([T.kb_search], run_id),
        max_iter=3,
    )
    briefer = cc.mk_agent(
        role="Briefing Coordinator",
        goal="Compile all gathered information into a concise, informative briefing document",
        backstory="As the Briefing Coordinator, your role is to consolidate the research, analysis, and strategic "
        "insights. " + cc.WORLD,
        max_iter=2,
    )
    ctx = "Meeting request (participants, context, objective):\n{request}"
    research = Task(
        description="Conduct comprehensive research on each of the individuals and companies involved in the upcoming "
        "meeting. Gather information on account history, recent orders, open issues, roles, and relevant business "
        "activities. If a participant cannot be found, say so instead of guessing.\n\n" + ctx,
        expected_output="A detailed report summarizing key findings about each participant and company, highlighting "
        "information that could be relevant for the meeting.",
        async_execution=True,
        agent=researcher,
    )
    industry_analysis = Task(
        description="Analyze the current industry trends, challenges, and opportunities relevant to the meeting's "
        "context. Consider market reports, recent developments, and expert opinions.\n\n" + ctx,
        expected_output="An insightful analysis that identifies major trends, potential challenges, and strategic "
        "opportunities.",
        async_execution=True,
        agent=industry,
    )
    strategy = Task(
        description="Develop strategic talking points, questions, and discussion angles for the meeting based on the "
        "research and industry analysis conducted.\n\n" + ctx,
        expected_output="A list of key talking points and strategic questions to ask to help achieve the meeting's "
        "objective.",
        agent=strategist,
        context=[research, industry_analysis],
    )
    briefing = Task(
        description="Compile all the research findings, industry analysis, and strategic talking points into a concise, "
        "comprehensive briefing document for the meeting. Ensure the briefing is easy to digest.\n\n" + ctx,
        expected_output="A well-structured briefing document with sections for participant bios, industry overview, "
        "talking points, and strategic recommendations.",
        agent=briefer,
        context=[research, industry_analysis, strategy],
    )
    return Crew(
        agents=[researcher, industry, strategist, briefer],
        tasks=[research, industry_analysis, strategy, briefing],
        process=Process.sequential,
        verbose=False,
    )


def run(task: dict) -> str:
    crew = build_crew(cc.run_id_for(AGENT.id, task))
    return cc.result_text(crew.kickoff(inputs={"request": task["input"]}))
