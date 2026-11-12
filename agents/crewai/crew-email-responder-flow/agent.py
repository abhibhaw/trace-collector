"""Email auto-responder flow, adapted from crewAI-examples/flows/email_auto_responder_flow.

Shape kept from the source: a Flow whose @start step fetches new emails, then an EmailFilterCrew with three agents
(Senior Email Analyst -> Email Action Specialist -> Email Response Writer) filters them, pulls each thread,
and drafts replies. The source's filter/action/draft task prompts (from its CrewAI-LangGraph sibling) are kept.
Adapted: Gmail is replaced by the support-ticket inbox (a ticket is an email thread; the customer's address comes
from the CRM) and drafts go to the email outbox; Tavily is replaced by the knowledge base. The 180-second polling
loop is replaced by one pass. A @router decides between "empty inbox" and "draft replies" paths, and the inbox is
selected by a single LiteAgent (Agent.kickoff) with a pydantic output.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from typing import List

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-email-responder-flow",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/flows/email_auto_responder_flow",
    description="CrewAI Flow that picks inbox threads, routes on whether any need attention, and runs a filter/action/draft crew that writes replies.",
    complexity="medium",
    model=cc.HAIKU,
    tracing="openinference",
)


class Email(BaseModel):
    id: str = Field(..., description="Ticket id, used as the thread id, for example TCK-40017")
    sender: str = Field(..., description="Customer id or name of the sender")
    snippet: str = Field(..., description="Subject or first line of the message")


class InboxBatch(BaseModel):
    emails: List[Email] = Field(default_factory=list, description="At most 3 threads that match the request")
    note: str = Field("", description="Why these threads were chosen, or why none were")


class AutoResponderState(BaseModel):
    request: str = ""
    run_id: str = ""
    emails: List[Email] = []
    note: str = ""
    result: str = ""


def setup() -> None:
    cc.setup_tracing()


def format_emails(emails: List[Email]) -> str:
    rows = []
    for e in emails:
        rows.append("\n".join([f"ID: {e.id}", f"- Thread ID: {e.id}", f"- Snippet: {e.snippet}", f"- From: {e.sender}",
                               "--------"]))
    return "\n".join(rows)


class EmailFilterCrew:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def crew(self) -> Crew:
        thread_tools = cc.make_tools([T.tickets_get, T.crm_get_customer], self.run_id)
        filter_agent = cc.mk_agent(
            role="Senior Email Analyst",
            goal="Filter out non-essential emails like newsletters and promotional content",
            backstory="As a Senior Email Analyst, you have extensive experience in email content analysis. You are adept "
            "at distinguishing important emails from spam, newsletters, and other irrelevant content. " + cc.WORLD,
            max_iter=2,
        )
        action_agent = cc.mk_agent(
            role="Email Action Specialist",
            goal="Identify action-required emails and compile a list of their IDs",
            backstory="With a keen eye for detail and a knack for understanding context, you specialize in identifying "
            "emails that require immediate action. " + cc.WORLD,
            tools=thread_tools + cc.make_tools([T.orders_get], self.run_id),
            max_iter=5,
        )
        writer = cc.mk_agent(
            role="Email Response Writer",
            goal="Draft responses to action-required emails",
            backstory="You are a skilled writer, adept at crafting clear, concise, and effective email responses on behalf "
            "of the Lumen & Co support team. You never promise refunds or credits; you say a specialist will confirm. "
            + cc.WORLD,
            tools=cc.make_tools([T.kb_search, T.email_send, T.tickets_update], self.run_id),
            max_iter=6,
        )
        filter_task = Task(
            description="Analyze a batch of emails and filter out non-essential ones such as newsletters, promotional "
            "content and notifications. Make sure to filter for the messages actually directed at the support team.\n\n"
            "Operator instructions: {request}\n\nEMAILS\n-------\n{emails}",
            expected_output="The relevant thread_ids and the sender, as bullet points.",
            agent=filter_agent,
        )
        action_task = Task(
            description="For each email thread, pull and analyze the complete thread using only the actual Thread ID, and "
            "the sender's CRM profile for their email address. Understand the context, key points, and the overall "
            "sentiment. Identify the main query or concerns that need to be addressed in the response for each.",
            expected_output="A list for all emails with: the thread_id, a summary of the thread, the main points, who will "
            "be answered, communication style, and the sender's email address (or 'unknown').",
            agent=action_agent,
            context=[filter_task],
        )
        draft_task = Task(
            description="Based on the action-required emails identified, draft responses for each. Ensure each response "
            "is tailored to the specific needs and context. Research the knowledge base first IF NECESSARY. Send each "
            "response with email_send (to, subject, body) when the sender's address is known, then add a short internal "
            "note to the ticket with tickets_update. Do not change ticket status unless the operator asked for it.\n\n"
            "Operator instructions: {request}",
            expected_output="A confirmation listing each thread, whether a reply was sent (message id) or why not.",
            agent=writer,
            context=[action_task],
        )
        return Crew(agents=[filter_agent, action_agent, writer], tasks=[filter_task, action_task, draft_task],
                    process=Process.sequential, verbose=False)


class EmailAutoResponderFlow(Flow[AutoResponderState]):
    @start()
    async def fetch_new_emails(self):
        scout = cc.mk_agent(
            role="Inbox Fetcher",
            goal="Pick the support inbox threads that the operator's instructions refer to",
            backstory="You read the support inbox (tickets) and return at most 3 matching threads. If the operator names "
            "ticket ids, use exactly those. To filter by customer tier or plan, check the "
            "senders with crm_get_customer. " + cc.WORLD,
            tools=cc.make_tools([T.tickets_search, T.crm_get_customer], self.state.run_id),
            max_iter=5,
        )
        out = await scout.kickoff_async(
            f"Operator instructions:\n{self.state.request}\n\nReturn the threads to process (max 3).",
            response_format=InboxBatch,
        )
        batch = out.pydantic if out.pydantic is not None else InboxBatch(note=str(out.raw))
        self.state.emails = batch.emails[:3]
        self.state.note = batch.note

    @router(fetch_new_emails)
    def check_queue(self):
        return "has_emails" if self.state.emails else "empty_inbox"

    @listen("has_emails")
    async def generate_draft_responses(self):
        result = await EmailFilterCrew(self.state.run_id).crew().kickoff_async(
            inputs={"emails": format_emails(self.state.emails), "request": self.state.request}
        )
        self.state.emails = []
        self.state.result = result.raw
        return self.state.result

    @listen("empty_inbox")
    def nothing_to_do(self):
        self.state.result = f"No threads need a reply. {self.state.note}".strip()
        return self.state.result


def run(task: dict) -> str:
    flow = EmailAutoResponderFlow()
    flow.kickoff(inputs={"request": task["input"], "run_id": cc.run_id_for(AGENT.id, task)})
    if not flow.state.result:
        raise RuntimeError("flow finished without a result")
    return flow.state.result
