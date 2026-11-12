"""Write-a-book flow, adapted from crewAI-examples/flows/write_a_book_with_flows.

Shape kept from the source: a Flow with BookState; @start runs the OutlineCrew (researcher -> outliner, pydantic
BookOutline); write_chapters runs one WriteBookChapterCrew per chapter concurrently (researcher -> writer,
pydantic Chapter); join_and_save_chapter assembles the markdown book.
Adapted: SerperDevTool is replaced with web search and Wikipedia; topic and goal come from the request; to bound
cost the outline is capped at 3 chapters of about 400-600 words; a @router after the outline skips chapter
writing when the outliner returns no usable chapters (refused or unclear topic); the book is returned, not saved.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

import asyncio
from typing import List

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-book-flow",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/flows/write_a_book_with_flows",
    description="CrewAI Flow that outlines a short book with a research crew, writes chapters in parallel crews, and joins them.",
    complexity="complex",
    model=cc.HAIKU,
    tracing="openinference",
)

MAX_CHAPTERS = 3


class ChapterOutline(BaseModel):
    title: str
    description: str


class BookOutline(BaseModel):
    title: str = Field(..., description="Book title")
    chapters: List[ChapterOutline] = Field(..., description=f"1 to {MAX_CHAPTERS} chapters in order; empty if the topic cannot be written")
    note: str = Field("", description="Assumptions, or why no outline was produced")


class Chapter(BaseModel):
    title: str
    content: str


class BookState(BaseModel):
    request: str = ""
    run_id: str = ""
    title: str = ""
    note: str = ""
    book: List[Chapter] = []
    book_outline: List[ChapterOutline] = []
    output: str = ""


def setup() -> None:
    cc.setup_tracing()


def outline_crew(run_id: str) -> Crew:
    researcher = cc.mk_agent(
        role="Research Agent",
        goal="Gather comprehensive information about the requested topic that will be used to create an organized and "
        "well-structured book outline.",
        backstory="You're a seasoned researcher, known for gathering the best sources and understanding the key elements "
        "of any topic. You aim to collect all relevant information so the book outline can be accurate and informative. "
        + cc.WORLD,
        tools=cc.make_tools([T.web_search, T.wikipedia_search, T.wikipedia_page], run_id, max_chars=5000),
        max_iter=4,
    )
    outliner = cc.mk_agent(
        role="Book Outlining Agent",
        goal="Based on the research, generate a book outline about the requested topic, with all chapters in sequential "
        "order and a title and description for each chapter.",
        backstory="You are a skilled organizer, great at turning scattered information into a structured format. Your "
        "goal is to create clear, concise chapter outlines with all key topics and subtopics covered.",
        max_iter=2,
    )
    research_topic = Task(
        description="Research the topic of the book to gather the most important information that will be useful in "
        "creating a book outline. Ensure you focus on high-quality, reliable sources.\n\nAuthor's request (topic, "
        "audience, goal):\n{request}",
        expected_output="A set of key points and important information about the topic that will be used to create the "
        "outline.",
        agent=researcher,
    )
    generate_outline = Task(
        description=f"Create a short book outline with at most {MAX_CHAPTERS} chapters in sequential order based on the "
        "research findings. Ensure that each chapter has a title and a brief description that highlights the topics and "
        "subtopics to be covered. Each chapter is only going to be about 500 words. Do not duplicate chapters or topics. "
        "If the request is harmful or impossible to write, return no chapters and explain in note.\n\n"
        "Author's request:\n{request}",
        expected_output="A book title and an outline of chapters, with titles and descriptions.",
        agent=outliner,
        context=[research_topic],
        output_pydantic=BookOutline,
    )
    return Crew(agents=[researcher, outliner], tasks=[research_topic, generate_outline], process=Process.sequential,
                verbose=False)


def chapter_crew(run_id: str) -> Crew:
    researcher = cc.mk_agent(
        role="Research Agent",
        goal="Gather comprehensive information about {chapter_title} that will be used to enhance the content of the "
        "chapter.",
        backstory="You are an experienced researcher skilled in finding the most relevant and up-to-date information on "
        "any given topic. Your job is to provide insightful data that supports and enriches the writing process. "
        + cc.WORLD,
        tools=cc.make_tools([T.web_search, T.wikipedia_page], run_id, max_chars=4000),
        max_iter=3,
    )
    writer = cc.mk_agent(
        role="Chapter Writer",
        goal="Write a well-structured chapter for the book based on the provided chapter title, goal, and outline.",
        backstory="You are an exceptional writer, known for producing engaging, well-researched, and informative content. "
        "You excel at transforming complex ideas into readable and well-organized chapters.",
        max_iter=2,
    )
    research_chapter = Task(
        description="Research the chapter topic, title, and outline to gather content that will be helpful in writing the "
        "chapter. Focus on reliable sources. Two or three searches are enough.\n\nAuthor's request: {request}\n"
        "Chapter title: {chapter_title}\nChapter description: {chapter_description}\n\nOutline of the entire book:\n"
        "{book_outline}",
        expected_output="A set of insights and facts that can be used in writing the chapter.",
        agent=researcher,
    )
    write_chapter = Task(
        description="Write a well-structured chapter based on the chapter title, goal, and outline description. Write it "
        "in markdown, about 400-600 words. It needs to fit in well with the rest of the chapters in the book.\n\n"
        "Author's request: {request}\nChapter title: {chapter_title}\nChapter description: {chapter_description}\n\n"
        "Outline of the entire book:\n{book_outline}",
        expected_output="A markdown chapter of about 500 words covering the chapter title and description.",
        agent=writer,
        context=[research_chapter],
        output_pydantic=Chapter,
    )
    return Crew(agents=[researcher, writer], tasks=[research_chapter, write_chapter], process=Process.sequential,
                verbose=False)


class BookFlow(Flow[BookState]):
    @start()
    async def generate_book_outline(self):
        out = await outline_crew(self.state.run_id).kickoff_async(inputs={"request": self.state.request})
        outline = out.pydantic
        if outline is None:
            raise RuntimeError(f"outline crew returned no structured outline: {str(out.raw)[:300]}")
        self.state.title = outline.title
        self.state.note = outline.note
        self.state.book_outline = outline.chapters[:MAX_CHAPTERS]

    @router(generate_book_outline)
    def check_outline(self):
        return "write" if self.state.book_outline else "no_outline"

    @listen("write")
    async def write_chapters(self):
        outline_json = "\n".join(c.model_dump_json() for c in self.state.book_outline)

        async def write_single_chapter(chapter_outline: ChapterOutline) -> Chapter:
            out = await chapter_crew(self.state.run_id).kickoff_async(
                inputs={
                    "request": self.state.request,
                    "chapter_title": chapter_outline.title,
                    "chapter_description": chapter_outline.description,
                    "book_outline": outline_json,
                }
            )
            if out.pydantic is not None:
                return out.pydantic
            return Chapter(title=chapter_outline.title, content=str(out.raw))

        chapters = await asyncio.gather(*(write_single_chapter(c) for c in self.state.book_outline))
        self.state.book.extend(chapters)

    @listen(write_chapters)
    def join_and_save_chapter(self):
        parts = [f"# {self.state.title}\n"]
        for chapter in self.state.book:
            parts.append(f"## {chapter.title}\n\n{chapter.content}\n")
        self.state.output = "\n".join(parts)

    @listen("no_outline")
    def explain_no_outline(self):
        self.state.output = f"No book was written. {self.state.note}".strip()


def run(task: dict) -> str:
    flow = BookFlow()
    flow.kickoff(inputs={"request": task["input"], "run_id": cc.run_id_for(AGENT.id, task)})
    if not flow.state.output:
        raise RuntimeError("flow finished without output")
    return flow.state.output
