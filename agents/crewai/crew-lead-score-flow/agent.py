"""Lead score flow, adapted from crewAI-examples/flows/lead-score-flow.

Shape kept from the source: a Flow with typed state; @start loads the leads; score_leads runs one LeadScoreCrew
per lead concurrently (asyncio.gather of kickoff_async) with a pydantic CandidateScore output; a @router
"human_in_the_loop" step ranks the leads and either loops back to scoring with feedback
("scored_leads_feedback", via or_) or goes on to "generate_emails", where a LeadResponseCrew per lead writes a
follow-up email (proceed for the top leads, polite decline for the rest).
Adapted: candidates from a CSV become B2B leads from the CRM (an upsell to the Business plan or a sales offer
named in the request); a LiteAgent sources up to 3 leads; the human reviewer is a deterministic auto-approver
that asks for one re-score when every score is weak; emails go to the outbox with email_send instead of files.
The scoring agent runs on Sonnet; sourcing and email writing run on Haiku.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

import asyncio
from typing import List

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-lead-score-flow",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/flows/lead-score-flow",
    description="CrewAI Flow that sources CRM leads, scores them in parallel crews (Sonnet, pydantic), routes through an auto-approver loop, and writes follow-up emails.",
    complexity="complex",
    model=cc.SONNET,
    tracing="openinference",
)

MAX_LEADS = 3
TOP_N = 2
WEAK_SCORE = 35


class Candidate(BaseModel):
    id: str = Field(..., description="CRM customer id, for example CUST-10042")
    name: str
    email: str = ""
    bio: str = Field("", description="One-line profile: company, city, tier, plan")


class LeadList(BaseModel):
    leads: List[Candidate] = Field(default_factory=list, description=f"At most {MAX_LEADS} leads")
    offer: str = Field(..., description="The offer or job the leads are scored against, in one sentence")


class CandidateScore(BaseModel):
    id: str
    score: int = Field(..., description="1-100")
    reason: str


class ScoredCandidate(BaseModel):
    id: str
    name: str
    email: str
    bio: str
    score: int
    reason: str


class LeadScoreState(BaseModel):
    request: str = ""
    run_id: str = ""
    offer: str = ""
    candidates: List[Candidate] = []
    candidate_score: List[CandidateScore] = []
    hydrated_candidates: List[ScoredCandidate] = []
    scored_leads_feedback: str = ""
    rescored: bool = False
    emails: List[str] = []


def setup() -> None:
    cc.setup_tracing()


def lead_score_crew(run_id: str) -> Crew:
    evaluator = cc.mk_agent(
        role="Senior Lead Evaluation Expert",
        goal="Analyze leads' account data and compare it against the offer to provide a score and reasoning.",
        backstory="As a Senior Lead Evaluation Expert, you have extensive experience in assessing B2B and consumer "
        "accounts. You weigh purchase history, lifetime value, open support issues, plan, and fit with the offer. "
        + cc.WORLD,
        tools=cc.make_tools([T.crm_get_customer, T.orders_list, T.tickets_search], run_id),
        model=cc.SONNET,
        max_iter=4,
    )
    evaluate = Task(
        description="Evaluate a lead against the offer.\n\nLEAD\n----\nLead ID: {candidate_id}\nName: {name}\nProfile: {bio}\n\n"
        "OFFER\n-----\n{offer}\n\nUse the CRM, orders and tickets to check the account. Your final answer MUST include:\n"
        "- The lead's unique ID\n- A score between 1 and 100. Don't use numbers like 100, 75, or 50. Instead, use specific "
        "numbers like 87, 63, or 42.\n- A detailed reasoning, considering fit, purchase history, account health, and "
        "growth potential.\n{additional_instructions}",
        expected_output="A very specific score from 1 to 100 for the lead, along with a detailed reasoning.",
        agent=evaluator,
        output_pydantic=CandidateScore,
    )
    return Crew(agents=[evaluator], tasks=[evaluate], process=Process.sequential, verbose=False)


def lead_response_crew(run_id: str) -> Crew:
    writer = cc.mk_agent(
        role="Account Coordinator",
        goal="Compose personalized follow-up emails to leads based on their profile and whether we are pursuing them. "
        "If we are proceeding, request availability for a call. Otherwise, send a polite, non-committal note.",
        backstory="You are Sarah from the Lumen & Co sales team, with excellent communication skills and a talent for "
        "crafting personalized and thoughtful emails. " + cc.WORLD,
        tools=cc.make_tools([T.email_send], run_id),
        max_iter=2,
    )
    send = Task(
        description="Compose and send a personalized follow-up email to this lead.\n\nLEAD\n----\nLead ID: {candidate_id}\n"
        "Name: {name}\nEmail: {email}\nProfile: {bio}\nOffer: {offer}\n\nPROCEEDING WITH LEAD: {proceed_with_candidate}\n\n"
        "- If proceeding, ask for their availability for a short call in the next few days.\n"
        "- If not, send a short, warm note that keeps the door open without a hard pitch.\n"
        "Send it with email_send. If the email address is empty, do not send; return the draft instead.",
        expected_output="The email that was sent (or drafted), with the message id if sent.",
        agent=writer,
    )
    return Crew(agents=[writer], tasks=[send], process=Process.sequential, verbose=False)


class LeadScoreFlow(Flow[LeadScoreState]):
    @start()
    async def load_leads(self):
        sourcer = cc.mk_agent(
            role="Lead Sourcer",
            goal=f"Find at most {MAX_LEADS} CRM accounts that fit the sales request",
            backstory="You turn a sales request into a short lead list using CRM search. Use the customer ids the "
            "request names, if any. Prefer active accounts. " + cc.WORLD,
            tools=cc.make_tools([T.crm_search_customers], self.state.run_id),
            max_iter=3,
        )
        out = await sourcer.kickoff_async(
            f"Sales request:\n{self.state.request}\n\nReturn the offer and up to {MAX_LEADS} leads with id, name, email, "
            "and a one-line profile.",
            response_format=LeadList,
        )
        if out.pydantic is None:
            raise RuntimeError(f"lead sourcing returned no structured output: {str(out.raw)[:300]}")
        self.state.offer = out.pydantic.offer
        self.state.candidates = out.pydantic.leads[:MAX_LEADS]

    @listen(or_(load_leads, "scored_leads_feedback"))
    async def score_leads(self):
        self.state.candidate_score = []

        async def score_single_candidate(candidate: Candidate):
            result = await lead_score_crew(self.state.run_id).kickoff_async(
                inputs={
                    "candidate_id": candidate.id,
                    "name": candidate.name,
                    "bio": candidate.bio,
                    "offer": self.state.offer,
                    "additional_instructions": self.state.scored_leads_feedback,
                }
            )
            if result.pydantic is not None:
                score = result.pydantic
                score.id = candidate.id
                self.state.candidate_score.append(score)

        await asyncio.gather(*(score_single_candidate(c) for c in self.state.candidates))

    @router(score_leads)
    def human_in_the_loop(self):
        scores = {s.id: s for s in self.state.candidate_score}
        hydrated = [
            ScoredCandidate(**c.model_dump(), score=scores[c.id].score, reason=scores[c.id].reason)
            for c in self.state.candidates
            if c.id in scores
        ]
        self.state.hydrated_candidates = sorted(hydrated, key=lambda c: c.score, reverse=True)
        if not hydrated:
            return "no_leads"
        # Auto-approver standing in for the source's input() prompt: one re-score round if every lead is weak.
        if not self.state.rescored and all(c.score < WEAK_SCORE for c in hydrated):
            self.state.rescored = True
            self.state.scored_leads_feedback = (
                "Reviewer feedback: every score was low. Re-check whether the account could still be a fit for a "
                "smaller version of the offer, and weigh recent orders more than old tickets."
            )
            return "scored_leads_feedback"
        return "generate_emails"

    @listen("generate_emails")
    async def write_and_save_emails(self):
        top_ids = {c.id for c in self.state.hydrated_candidates[:TOP_N]}

        async def write_email(c: ScoredCandidate):
            result = await lead_response_crew(self.state.run_id).kickoff_async(
                inputs={
                    "candidate_id": c.id,
                    "name": c.name,
                    "email": c.email,
                    "bio": c.bio,
                    "offer": self.state.offer,
                    "proceed_with_candidate": c.id in top_ids,
                }
            )
            return f"### {c.name} ({c.id}) score={c.score} proceed={c.id in top_ids}\n{c.reason}\n\n{result.raw}"

        self.state.emails = list(await asyncio.gather(*(write_email(c) for c in self.state.hydrated_candidates)))

    @listen("no_leads")
    def report_no_leads(self):
        self.state.emails = ["No matching leads were found or scored for this request."]


def run(task: dict) -> str:
    flow = LeadScoreFlow()
    flow.kickoff(inputs={"request": task["input"], "run_id": cc.run_id_for(AGENT.id, task)})
    if not flow.state.emails:
        raise RuntimeError("flow finished without emails or a report")
    return f"Offer: {flow.state.offer}\n\n" + "\n\n".join(flow.state.emails)
