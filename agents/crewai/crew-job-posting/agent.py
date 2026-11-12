"""Job posting crew, adapted from crewAI-examples/crews/job-posting.

Shape kept from the source: research analyst, job description writer, and review specialist; five sequential
tasks (company culture, role requirements as structured JSON, draft, review/edit, industry analysis).
Adapted: Serper/WebsiteSearch/FileRead tools are replaced with web search, URL fetch, and the HR API
(employee directory for team context, policy lookup for real benefits). The hiring company is Lumen & Co.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from typing import List

from crewai import Crew, Process, Task
from pydantic import BaseModel, Field

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-job-posting",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting",
    description="Sequential research, writer, and reviewer crew that turns a hiring manager's request into a polished job posting.",
    complexity="simple",
    model=cc.HAIKU,
    tracing="openinference",
)


class ResearchRoleRequirements(BaseModel):
    """Research role requirements model"""

    skills: List[str] = Field(..., description="Recommended skills for the ideal candidate.")
    experience: List[str] = Field(..., description="Recommended experience for the ideal candidate.")
    qualities: List[str] = Field(..., description="Recommended qualities for the ideal candidate.")


def setup() -> None:
    cc.setup_tracing()


def build_crew(run_id: str) -> Crew:
    researcher = cc.mk_agent(
        role="Research Analyst",
        goal="Analyze the company website and provided description to extract insights on culture, values, and "
        "specific needs.",
        backstory="Expert in analyzing company cultures and identifying key values and needs from various sources, "
        "including websites, the internal employee directory, and brief descriptions. " + cc.WORLD,
        tools=cc.make_tools([T.web_search, T.fetch_url, T.hr_search_employees], run_id),
        max_iter=3,
    )
    writer = cc.mk_agent(
        role="Job Description Writer",
        goal="Use insights from the Research Analyst to create a detailed, engaging, and enticing job posting.",
        backstory="Skilled in crafting compelling job descriptions that resonate with the company's values and attract "
        "the right candidates. You only promise benefits that the HR policies confirm. " + cc.WORLD,
        tools=cc.make_tools([T.hr_policy_lookup], run_id),
        max_iter=3,
    )
    reviewer = cc.mk_agent(
        role="Review and Editing Specialist",
        goal="Review the job posting for clarity, engagement, grammatical accuracy, and alignment with company values "
        "and refine it to ensure perfection.",
        backstory="A meticulous editor with an eye for detail, ensuring every piece of content is clear, engaging, "
        "grammatically perfect, inclusive, and legally safe (no age, gender, or nationality preferences). " + cc.WORLD,
        tools=cc.make_tools([T.hr_policy_lookup], run_id),
        max_iter=3,
    )
    req = "Hiring manager's request:\n{request}"
    culture = Task(
        description="Analyze the hiring company (Lumen & Co, unless the request says otherwise) and the hiring manager's "
        "description. Focus on understanding the company's culture, values, and mission, and the team the role joins "
        "(look it up in the employee directory). Identify unique selling points relevant to the role.\n\n" + req,
        expected_output="A report on culture, values, mission, and the hiring team, with suggestions on how to use them "
        "in the job posting.",
        agent=researcher,
    )
    requirements = Task(
        description="Based on the hiring manager's needs, identify the key skills, experiences, and qualities the ideal "
        "candidate should possess for the role. Consider the company's current projects, its competitive landscape, "
        "and industry trends.\n\n" + req,
        expected_output="A list of recommended skills, experiences, and qualities for the ideal candidate.",
        agent=researcher,
        output_pydantic=ResearchRoleRequirements,
    )
    draft = Task(
        description="Draft a job posting for the role described by the hiring manager. Start with a compelling "
        "introduction, followed by a detailed role description, responsibilities, and required skills and "
        "qualifications. Look up the relevant HR policies (for example pto, remote-work, parental-leave, equipment) "
        "and include only benefits they confirm, plus any the manager lists.\n\n" + req,
        expected_output="A detailed, engaging job posting with introduction, role description, responsibilities, "
        "requirements, and benefits.",
        agent=writer,
        context=[culture, requirements],
    )
    review = Task(
        description="Review the draft job posting. Check for clarity, engagement, grammatical accuracy, inclusive "
        "language, and alignment with the company's culture and values. Edit and refine the content. If the manager "
        "asked for something illegal or discriminatory, remove it and say why in the feedback.\n\n" + req,
        expected_output="A polished, error-free job posting in markdown, followed by short feedback and approval status.",
        agent=reviewer,
        context=[draft],
    )
    industry = Task(
        description="Conduct a short analysis of the industry related to the role and company (smart home / IoT "
        "consumer electronics unless the request says otherwise). Assess current trends, challenges, and "
        "opportunities that affect how attractive the role is to candidates.\n\n" + req,
        expected_output="A concise industry analysis with strategic hints for positioning the role, then the final "
        "reviewed job posting reproduced in full at the end.",
        agent=researcher,
        context=[review],
    )
    return Crew(
        agents=[researcher, writer, reviewer],
        tasks=[culture, requirements, draft, review, industry],
        process=Process.sequential,
        verbose=False,
    )


def run(task: dict) -> str:
    crew = build_crew(cc.run_id_for(AGENT.id, task))
    return cc.result_text(crew.kickoff(inputs={"request": task["input"]}))
