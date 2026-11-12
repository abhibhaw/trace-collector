"""DSPy multi-hop retrieval program over Wikipedia (ChainOfThought + retrieval loop).

Derived from the DSPy "Tutorial: Multi-Hop Retrieval" (docs/docs/tutorials/multihop_search): the `Hop`
module with two ChainOfThought sub-modules, `generate_query` ('claim, notes -> query') and
`append_notes` ('claim, notes, context -> new_notes: list[str], titles: list[str]'), run for num_hops.
Changed: the local BM25 index over 2017 wiki abstracts is replaced by the tools-service Wikipedia search;
num_hops is 3 (source: 4) to bound cost; a final ChainOfThought turns the notes into an answer;
no optimization is run.
"""

import dspy
import dsh_common
from peira_harness import AgentMeta, tools

AGENT = AgentMeta(
    id="dspy-multihop-rag",
    framework="dspy",
    source_url="https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/multihop_search/index.ipynb",
    description="Multi-hop Wikipedia RAG: per hop a CoT writes a query, retrieval runs, and a CoT appends notes; a final CoT answers.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="openinference",
)


def search(query: str, k: int) -> dict[str, float]:
    """Wikipedia search; returns {"title | text": score} like the tutorial's BM25 search."""
    res = tools.wikipedia_search(query=query, max_results=k)
    if "error" in res:
        return {f"search error | {res['error']}": 0.0}
    run = {}
    for rank, r in enumerate(res.get("results", [])):
        text = r.get("snippet") or ""
        run[f"{r.get('title', '')} | {text}"] = float(k - rank)
    return run


class Hop(dspy.Module):
    def __init__(self, num_docs=5, num_hops=3):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought("claim, notes -> query")
        self.append_notes = dspy.ChainOfThought("claim, notes, context -> new_notes: list[str], titles: list[str]")
        self.answer = dspy.ChainOfThought("claim, notes: list[str] -> answer")

    def forward(self, claim: str) -> dspy.Prediction:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)

        answer = self.answer(claim=claim, notes=notes).answer
        return dspy.Prediction(notes=notes, titles=list(set(titles)), answer=answer)


def setup() -> None:
    dsh_common.setup_dspy(AGENT.model)


def run(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    pred = Hop()(claim=task["input"])
    return f"{pred.answer}\n\nSources: {', '.join(pred.titles)}"
