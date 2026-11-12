"""DSPy ReAct agent for multi-hop Wikipedia browsing (claim -> supporting titles + verdict).

Derived from the DSPy "Tutorial: Agents" (docs/docs/tutorials/agents): a dspy.ReAct with the two tools
`search_wikipedia` and `lookup_wikipedia`, instructed to find all Wikipedia titles needed to verify a claim.
Changed: the ColBERTv2 wiki17-abstracts server is replaced by the tools-service Wikipedia search/page tools,
the signature also asks for a short verdict, max_iters is 8 (source: 20), and no optimization is run.
"""

import dspy
import dsh_common
from peira_harness import AgentMeta, tools

AGENT = AgentMeta(
    id="dspy-wiki-react",
    framework="dspy",
    source_url="https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/agents/index.ipynb",
    description="dspy.ReAct agent that browses Wikipedia to find the pages needed to verify or refute a claim.",
    complexity="simple",
    model="claude-haiku-5-5",
    tracing="openinference",
)

instructions = (
    "Find all Wikipedia titles relevant to verifying (or refuting) the claim. "
    "Then give a one-paragraph verdict: SUPPORTED, REFUTED, or NOT ENOUGH INFO, with the key evidence."
)
signature = dspy.Signature("claim -> titles: list[str], verdict: str", instructions)


def build_tools():
    DOCS: dict[str, str] = {}

    def search(query: str, k: int) -> list[str]:
        res = tools.wikipedia_search(query=query, max_results=k)
        if "error" in res:
            return [f"Search failed: {res['error']}"]
        out = []
        for r in res.get("results", []):
            title = r.get("title", "")
            text = r.get("snippet") or ""
            DOCS.setdefault(title, text)
            out.append(f"{title} | {text}")
        return out

    def search_wikipedia(query: str) -> list[str]:
        """Returns top-5 results and then the titles of the top-5 to top-10 results."""
        topK = search(query, 10)
        titles, topK = [f"`{x.split(' | ')[0]}`" for x in topK[5:10]], topK[:5]
        return topK + [f"Other retrieved pages have titles: {', '.join(titles)}."]

    def lookup_wikipedia(title: str) -> str:
        """Returns the text of the Wikipedia page, if it exists."""
        page = tools.wikipedia_page(title=title, max_chars=3000)
        if "error" in page:
            if title in DOCS:
                return DOCS[title]
            return f"No Wikipedia page found for title: {title} ({page['error']})"
        text = page.get("content") or ""
        DOCS[page.get("title", title)] = text
        return f"{page.get('title', title)} | {text}"

    return [search_wikipedia, lookup_wikipedia]


def setup() -> None:
    dsh_common.setup_dspy(AGENT.model)


def run(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    react = dspy.ReAct(signature, tools=build_tools(), max_iters=8)
    pred = react(claim=task["input"])
    titles = ", ".join(pred.titles or [])
    return f"{pred.verdict}\n\nRelevant Wikipedia titles: {titles}"
