"""Marketing coordinator with four AgentTool specialists (domain, website, marketing, logo).

Derived from google/adk-samples python/agents/marketing-agency (removed from main in #2591; pinned).
Adaptations:
- domain_create: Google Search availability checks -> a local DNS-lookup tool (check_domains,
  free and keyless) plus web_search.
- logo_create: Imagen -> a local generate_image tool that renders an SVG logo from the prompt and
  saves it with tool_context.save_artifact (InMemoryRunner's artifact service); load_artifacts kept.
- The coordinator is interactive (keywords, domain choice); a simulated user supplies the brand
  keywords from the request and picks the first suggested domain.
"""
from __future__ import annotations

import hashlib
import html
import re
import socket
from concurrent.futures import ThreadPoolExecutor

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext, load_artifacts
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

import adk_common as C
from peira_harness import AgentMeta

prompts = C.load_local(__file__)

AGENT = AgentMeta(
    id="adk-marketing-agency",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/marketing-agency",
    description="Marketing coordinator that calls domain, website, campaign and logo sub-agents as tools; logos are saved as SVG artifacts.",
    complexity="medium",
    model=C.HAIKU,
    tracing="openinference",
)

M = C.claude(C.HAIKU)
_DOMAIN_RE = re.compile(r"^(?=.{4,63}$)[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$")


def _resolves(domain: str) -> bool:
    try:
        socket.getaddrinfo(domain, None)
        return True
    except (socket.gaierror, UnicodeError, OSError):
        return False


def check_domains(domains: list[str]) -> dict:
    """Check whether domain names already resolve in DNS (a sign they are registered and in use).

    Args:
        domains: Up to 25 domain names such as "brightbrew.com".

    Returns:
        A dict with `registered` (resolving) and `possibly_available` (no DNS records) lists.
    """
    names = [d.strip().lower() for d in domains if d and d.strip()][:25]
    bad = [d for d in names if not _DOMAIN_RE.match(d)]
    good = [d for d in names if d not in bad]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = dict(zip(good, ex.map(_resolves, good)))
    return {
        "registered": [d for d, r in res.items() if r],
        "possibly_available": [d for d, r in res.items() if not r],
        "invalid": bad,
        "note": "No DNS records does not guarantee the name is unregistered; confirm at a registrar.",
    }


async def generate_image(img_prompt: str, tool_context: ToolContext) -> dict:
    """Generates a logo image (SVG) based on the prompt and stores it as an artifact."""
    h = hashlib.sha256(img_prompt.encode()).hexdigest()
    colors = re.findall(r"#[0-9a-fA-F]{6}", img_prompt) or [f"#{h[:6]}", f"#{h[6:12]}"]
    c1, c2 = colors[0], colors[1] if len(colors) > 1 else f"#{h[6:12]}"
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", img_prompt)
    name = next((w for w in words if w[0].isupper() and w.lower() not in {"a", "the", "logo", "minimal", "modern"}), words[0] if words else "Brand")
    shape = int(h[12], 16) % 3
    mark = (
        f'<circle cx="64" cy="64" r="44" fill="{c1}"/>'
        if shape == 0
        else f'<rect x="22" y="22" width="84" height="84" rx="18" fill="{c1}"/>'
        if shape == 1
        else f'<polygon points="64,16 112,104 16,104" fill="{c1}"/>'
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="128" viewBox="0 0 420 128">'
        f"<desc>{html.escape(img_prompt[:500])}</desc>{mark}"
        f'<text x="64" y="76" font-family="Helvetica" font-size="40" font-weight="700" fill="#ffffff" text-anchor="middle">{html.escape(name[:1].upper())}</text>'
        f'<text x="130" y="80" font-family="Helvetica" font-size="44" font-weight="600" fill="{c2}">{html.escape(name[:14])}</text></svg>'
    )
    version = await tool_context.save_artifact(
        "logo.svg", types.Part.from_bytes(data=svg.encode(), mime_type="image/svg+xml")
    )
    return {
        "status": "success",
        "detail": "Image generated successfully and stored in artifacts.",
        "filename": "logo.svg",
        "version": version,
        "colors": [c1, c2],
        "wordmark": name[:14],
    }


domain_create_agent = LlmAgent(
    model=M,
    name="domain_create_agent",
    description="Suggests available, brandable domain names for a topic.",
    instruction=prompts.DOMAIN_CREATE_PROMPT,
    output_key="domain_create_output",
    tools=[C.async_tool(check_domains), C.AT.web_search],
)

website_create_agent = LlmAgent(
    model=M,
    name="website_create_agent",
    description="Builds a professional website for the chosen domain.",
    instruction=prompts.WEBSITE_CREATE_PROMPT,
    output_key="website_create_output",
)

marketing_create_agent = LlmAgent(
    model=M,
    name="marketing_create_agent",
    description="Creates an online marketing campaign strategy for the chosen domain.",
    instruction=prompts.MARKETING_CREATE_PROMPT,
    output_key="marketing_create_output",
)

logo_create_agent = LlmAgent(
    model=M,
    name="logo_create_agent",
    description="An agent that generates images and answers questions about the images.",
    instruction=prompts.LOGO_CREATE_PROMPT,
    output_key="logo_create_output",
    tools=[generate_image, load_artifacts],
)

root_agent = LlmAgent(
    name="marketing_coordinator",
    model=M,
    description=(
        "Establish a powerful online presence and connect with your audience "
        "effectively. Guide you through defining your digital identity, from "
        "choosing the perfect domain name and crafting a professional "
        "website, to strategizing online marketing campaigns, "
        "designing a memorable logo, and creating engaging short videos"
    ),
    instruction=prompts.MARKETING_COORDINATOR_PROMPT
    + "\nIf the user's message already describes the brand, use it as the keywords instead of asking again."
    + " Keep the website output to one compact HTML page.\n",
    tools=[
        AgentTool(agent=domain_create_agent),
        AgentTool(agent=website_create_agent),
        AgentTool(agent=marketing_create_agent),
        AgentTool(agent=logo_create_agent),
    ],
)


def _client(turn: int, last_text: str, state: dict) -> str | None:
    """Simulated client: provides keywords, then picks the first domain, then asks to finish."""
    if state.get("logo_create_output") and state.get("marketing_create_output") and state.get("website_create_output"):
        return None
    if not state.get("domain_create_output"):
        return "Use the brand description from my first message as the keywords and go ahead with the domain search."
    m = re.search(r"\b([a-z0-9-]+\.[a-z]{2,})\b", state.get("domain_create_output", ""))
    choice = m.group(1) if m else "the first domain on your list"
    return f"Let's go with {choice}. Please do the website, the marketing campaign and the logo now, without further questions."


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(root_agent, task, app_name="marketing_agency", auto_user=_client, max_turns=3, max_llm_calls=22)
