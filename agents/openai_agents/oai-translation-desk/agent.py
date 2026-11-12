"""Translation desk: an orchestrator calls translator agents as tools, then a synthesizer
checks and assembles the final answer.

Adapted from openai-agents-python examples/agent_patterns/agents_as_tools.py. Kept the
orchestrator + translator sub-agents + synthesizer shape and prompts. Added German and
Japanese translators and a local glossary tool for Lumen & Co product terms.
"""

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, function_tool, trace

from _oai_common import HAIKU, model, refusal_as_answer, settings, setup_family
from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="oai-translation-desk",
    framework="openai-agents",
    source_url="https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/agents_as_tools.py",
    description="Orchestrator agent calls per-language translator agents as tools, then a synthesizer agent reviews and concatenates the translations.",
    complexity="simple",
    model=HAIKU,
    tracing="openinference",
)

# Product names and terms that must stay consistent across languages.
GLOSSARY = {
    "lumen cloud": "Lumen Cloud (product name, never translate)",
    "hub pro": "Hub Pro (product name, never translate)",
    "smart lock": "es: cerradura inteligente | fr: serrure connectée | it: serratura smart | de: Smart Lock | ja: スマートロック",
    "subscription": "es: suscripción | fr: abonnement | it: abbonamento | de: Abonnement | ja: サブスクリプション",
    "firmware update": "es: actualización de firmware | fr: mise à jour du firmware | it: aggiornamento firmware | de: Firmware-Update | ja: ファームウェア更新",
    "warranty": "es: garantía | fr: garantie | it: garanzia | de: Garantie | ja: 保証",
    "refund": "es: reembolso | fr: remboursement | it: rimborso | de: Rückerstattung | ja: 返金",
    "motion sensor": "es: sensor de movimiento | fr: détecteur de mouvement | it: sensore di movimento | de: Bewegungsmelder | ja: モーションセンサー",
}


@function_tool
def glossary_lookup(term: str) -> str:
    """Look up the approved translation of a Lumen & Co product term.

    Args:
        term: The English term, for example "smart lock" or "Lumen Cloud".
    """
    hit = GLOSSARY.get(term.strip().lower())
    return hit or f"No glossary entry for '{term}'. Translate it naturally."


LANGS = {
    "spanish": "An english to spanish translator",
    "french": "An english to french translator",
    "italian": "An english to italian translator",
    "german": "An english to german translator",
    "japanese": "An english to japanese translator",
}

_a: dict = {}


def setup() -> None:
    setup_family()
    tools = []
    for lang, desc in LANGS.items():
        sub = Agent(
            name=f"{lang}_agent",
            instructions=(f"You translate the user's message to {lang.capitalize()}. "
                          "Use glossary_lookup for Lumen & Co product terms before translating them."),
            handoff_description=desc,
            model=model(HAIKU),
            model_settings=settings(),
            tools=[glossary_lookup],
        )
        tools.append(sub.as_tool(tool_name=f"translate_to_{lang}",
                                 tool_description=f"Translate the user's message to {lang.capitalize()}",
                                 max_turns=3))
    _a["orchestrator"] = Agent(
        name="orchestrator_agent",
        instructions=(
            "You are a translation agent. You use the tools given to you to translate."
            "If asked for multiple translations, you call the relevant tools in order."
            "You never translate on your own, you always use the provided tools. "
            "If a requested language has no tool, say so instead of translating it yourself."
        ),
        model=model(HAIKU),
        model_settings=settings(),
        tools=tools,
    )
    _a["synthesizer"] = Agent(
        name="synthesizer_agent",
        instructions="You inspect translations, correct them if needed, and produce a final concatenated response.",
        model=model(HAIKU),
        model_settings=settings(),
    )


@refusal_as_answer
async def arun(task: dict) -> str:
    with trace("Orchestrator evaluator"):
        orch = await Runner.run(_a["orchestrator"], task["input"], max_turns=8)
        steps = [ItemHelpers.text_message_output(i) for i in orch.new_items if isinstance(i, MessageOutputItem)]
        if not any(steps) and not orch.new_items:
            raise RuntimeError("orchestrator produced no output")
        # Claude rejects a conversation that ends with an assistant turn (no prefill), so the
        # synthesizer gets an explicit user turn after the orchestrator transcript.
        synth_input = orch.to_input_list() + [{"role": "user", "content": "Inspect the translations above, correct them if needed, and produce the final response."}]
        synth = await Runner.run(_a["synthesizer"], synth_input, max_turns=2)
    return str(synth.final_output)
