"""Open Deep Research, adapted from smolagents examples/open_deep_research.

Source shape (examples/open_deep_research/run.py):
    manager CodeAgent(tools=[visualizer, TextInspectorTool], planning_interval=4, managed_agents=[search_agent])
      -> search_agent ToolCallingAgent(tools=[GoogleSearchTool, VisitTool, PageUpTool, PageDownTool,
         FinderTool, FindNextTool, ArchiveSearchTool, TextInspectorTool], planning_interval=4,
         provide_run_summary=True) with the extra `prompt_templates["managed_agent"]["task"]` suffix.
Adaptation:
- GoogleSearchTool (Serper) -> tools-service web search. ArchiveSearchTool (Wayback) -> arXiv search,
  plus Wikipedia, which are free and keyless.
- SimpleTextBrowser is re-implemented in-process over the tools-service fetch: a per-run browser keeps
  the current page and a viewport so page_up / page_down / find_on_page_ctrl_f / find_next keep their roles.
- TextInspectorTool keeps its design (a nested LLM call that answers a question about a document),
  reading URLs instead of local files. The visualizer (image QA) is dropped: no images in this setup.
- additional_authorized_imports=["*"] is replaced by a short safe list. Step budgets are reduced.
"""
from __future__ import annotations

import re
import uuid

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import CodeAgent, Tool, ToolCallingAgent
from smolagents.models import MessageRole, Model

AGENT = AgentMeta(
    id="smol-open-deep-research",
    framework="smolagents",
    source_url="https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research",
    description="Open Deep Research: a planning manager CodeAgent with a text inspector and a managed browsing ToolCallingAgent.",
    complexity="complex",
    model="claude-sonnet-5-5",
    tracing="openinference",
)

VIEWPORT = 5000
TEXT_LIMIT = 40000


class SimpleTextBrowser:
    """Minimal text browser: one current page split into fixed-size viewports."""

    def __init__(self) -> None:
        self.address = "about:blank"
        self.title = ""
        self.content = ""
        self.pages: list[str] = [""]
        self.idx = 0
        self.find_query: str | None = None
        self.find_pos = -1

    def visit(self, url: str) -> str:
        res = T.fetch_url(url, max_chars=TEXT_LIMIT)
        if "error" in res:
            return f"Error visiting {url}: {res['error']}"
        self.address, self.title = res.get("url", url), res.get("title", "")
        self.content = re.sub(r"\n{3,}", "\n\n", res.get("content", ""))
        self.pages = [self.content[i : i + VIEWPORT] for i in range(0, max(len(self.content), 1), VIEWPORT)] or [""]
        self.idx, self.find_query, self.find_pos = 0, None, -1
        return self.state()

    def state(self) -> str:
        header = f"Address: {self.address}\nTitle: {self.title}\n"
        header += f"Viewport position: Showing page {self.idx + 1} of {len(self.pages)}.\n"
        return header + "=======================\n" + self.pages[self.idx]

    def scroll(self, delta: int) -> str:
        self.idx = max(0, min(len(self.pages) - 1, self.idx + delta))
        return self.state()

    def find(self, query: str, start: int) -> str:
        if not self.content:
            return "No page is open. Use visit_page first."
        pos = self.content.lower().find(query.lower(), start)
        if pos < 0:
            return f"The search string '{query}' was not found on this page."
        self.find_query, self.find_pos = query, pos
        self.idx = pos // VIEWPORT
        return self.state()


class SearchInformationTool(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        res = T.web_search(query, max_results=8)
        if "error" in res:
            return f"Search error: {res['error']}"
        rows = [f"{i}. [{r.get('title')}]({r.get('url')})\n{r.get('snippet', '')}" for i, r in enumerate(res.get("results", []), 1)]
        return f"A search for '{query}' found {len(rows)} results:\n\n## Web Results\n" + "\n\n".join(rows)


class VisitTool(Tool):
    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text."
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the webpage to visit."}}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        return self.browser.visit(url)


class PageUpTool(Tool):
    name = "page_up"
    description = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    inputs = {}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        return self.browser.scroll(-1)


class PageDownTool(Tool):
    name = "page_down"
    description = (
        "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        return self.browser.scroll(1)


class FinderTool(Tool):
    name = "find_on_page_ctrl_f"
    description = "Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F."
    inputs = {
        "search_string": {
            "type": "string",
            "description": "The string to search for on the page. This search string supports wildcards like '*'",
        }
    }
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser):
        super().__init__()
        self.browser = browser

    def forward(self, search_string: str) -> str:
        return self.browser.find(search_string.replace("*", ""), 0)


class FindNextTool(Tool):
    name = "find_next"
    description = "Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search."
    inputs = {}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        if not self.browser.find_query:
            return "No active search. Use find_on_page_ctrl_f first."
        return self.browser.find(self.browser.find_query, self.browser.find_pos + 1)


class PaperSearchTool(Tool):
    name = "arxiv_search"
    description = "Search arXiv for scientific papers. Returns titles, authors, dates, abstracts and links."
    inputs = {"query": {"type": "string", "description": "Keywords for the paper search."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        res = T.arxiv_search(query, max_results=5)
        if "error" in res:
            return f"arXiv error: {res['error']}"
        return C._stringify(res.get("results", res), 6000)


class WikipediaTool(Tool):
    name = "wikipedia_page"
    description = "Read the introduction and main text of an English Wikipedia article by exact title."
    inputs = {"title": {"type": "string", "description": "The article title, e.g. 'Large Hadron Collider'."}}
    output_type = "string"

    def forward(self, title: str) -> str:
        res = T.wikipedia_page(title, max_chars=6000)
        if "error" in res:
            return f"Wikipedia error: {res['error']}"
        return f"{res.get('title')} ({res.get('url')})\n\n{res.get('content') or res.get('extract') or ''}"


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load documents yourself: instead call this tool to read a document (web page or PDF URL) as markdown text and ask questions about it.
This tool handles HTML pages and PDF files, for example arXiv PDFs. IT DOES NOT HANDLE IMAGES."""
    inputs = {
        "file_path": {
            "description": "The URL of the document you want to read as text, for example 'https://arxiv.org/pdf/1706.03762'.",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, text_limit: int = TEXT_LIMIT):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def forward(self, file_path: str, question: str | None = None) -> str:
        res = T.fetch_url(file_path, max_chars=self.text_limit)
        if "error" in res:
            return f"Error reading {file_path}: {res['error']}"
        text = res.get("content", "")
        if not question:
            return text[:VIEWPORT * 2]
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {"type": "text", "text": "You will have to write a short caption for this file, then answer this question:" + question}
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### " + str(res.get("title")) + "\n\n" + text[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        return self.model(messages).content


def setup() -> None:
    C.setup_tracing()


def create_agent() -> CodeAgent:
    model = C.make_model(AGENT.model)
    browser = SimpleTextBrowser()
    web_tools = [
        SearchInformationTool(),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        PaperSearchTool(),
        WikipediaTool(),
        TextInspectorTool(model),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=4,
        verbosity_level=0,
        planning_interval=4,
        # The paging tools share one browser state, so tool calls run one at a time.
        max_tool_threads=1,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    return CodeAgent(
        model=model,
        tools=[TextInspectorTool(model)],
        max_steps=4,
        verbosity_level=0,
        additional_authorized_imports=["math", "statistics", "datetime", "re", "json", "collections"],
        # Managed agents run inside the code executor: lift its 30 s default so a delegation is not killed and retried.
        executor_kwargs={"timeout_seconds": 300},
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    return C.last_text(create_agent().run(task["input"]))
