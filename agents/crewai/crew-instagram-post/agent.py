"""Instagram post crews, adapted from crewAI-examples/crews/instagram_post.

Shape kept from the source: two crews run back to back. The copy crew (lead market analyst, chief marketing
strategist, creative content creator) does product analysis, competitor analysis, campaign development and
ad copy. The image crew (senior photographer, chief creative director with delegation) turns the copy into
three photo briefs and reviews them, delegating follow-up work when needed.
Adapted: browser/search tools are replaced by the product catalog, inventory, knowledge base, and web search.
The "product website" is a Lumen & Co product (or category) named in the request.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from crewai import Crew, Process, Task

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-instagram-post",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/crews/instagram_post",
    description="Two chained crews: a copy crew analyses a product and writes Instagram copy, then an image crew drafts and reviews photo briefs with delegation.",
    complexity="medium",
    model=cc.HAIKU,
    tracing="openinference",
)


def setup() -> None:
    cc.setup_tracing()


def copy_crew(run_id: str) -> Crew:
    product_tools = cc.make_tools([T.products_search, T.inventory_check, T.kb_search], run_id)
    web = cc.make_tools([T.web_search, T.fetch_url], run_id)
    analyst = cc.mk_agent(
        role="Lead Market Analyst",
        goal="Conduct amazing analysis of the products and competitors, providing in-depth insights to guide marketing "
        "strategies.",
        backstory="As the Lead Market Analyst at a premier digital marketing firm, you specialize in dissecting online "
        "business landscapes. " + cc.WORLD,
        tools=product_tools + web,
        max_iter=4,
    )
    strategist = cc.mk_agent(
        role="Chief Marketing Strategist",
        goal="Synthesize amazing insights from product analysis to formulate incredible marketing strategies.",
        backstory="You are the Chief Marketing Strategist at a leading digital marketing agency, known for crafting "
        "bespoke strategies that drive success. " + cc.WORLD,
        tools=web,
        max_iter=2,
    )
    creator = cc.mk_agent(
        role="Creative Content Creator",
        goal="Develop compelling and innovative content for social media campaigns, with a focus on creating "
        "high-impact Instagram ad copies.",
        backstory="As a Creative Content Creator at a top-tier digital marketing agency, you excel in crafting narratives "
        "that resonate with audiences on social media. " + cc.WORLD,
        max_iter=2,
    )
    brief = "Customer brief (product and post details):\n{request}"
    product_analysis = Task(
        description="Analyze the product(s) named in the brief using the product catalog, inventory, and knowledge base. "
        "Focus on identifying unique features, benefits, price, availability, and the overall narrative. If the product "
        "does not exist or is discontinued, say so.\n\n" + brief,
        expected_output="A report on the product's key selling points, market appeal, and positioning suggestions.",
        agent=analyst,
    )
    competitor_analysis = Task(
        description="Explore competitors of the product. Identify the top 3 competitors and analyze their strategies, "
        "market positioning, and customer perception.\n\n" + brief,
        expected_output="A report with all context about the product and a comparison to its main competitors.",
        agent=analyst,
        context=[product_analysis],
    )
    campaign = Task(
        description="Create a targeted marketing campaign for the product. It should be designed to captivate and engage "
        "the product's target audience. Based on your ideas your co-workers will create the content.\n\n" + brief,
        expected_output="Campaign ideas that will resonate with the audience, including all context about the product "
        "and the customer.",
        agent=strategist,
        context=[product_analysis, competitor_analysis],
    )
    ad_copy = Task(
        description="Craft an engaging Instagram post copy. The copy should be punchy, captivating, concise, and aligned "
        "with the product marketing strategy. Respect any constraints in the brief (tone, length, hashtags, claims). "
        "Never invent discounts or claims not supported by the analysis.\n\n" + brief,
        expected_output="3 options for an Instagram ad copy.",
        agent=creator,
        context=[campaign],
    )
    return Crew(agents=[analyst, strategist, creator], tasks=[product_analysis, competitor_analysis, campaign, ad_copy],
                process=Process.sequential, verbose=False)


def image_crew() -> Crew:
    photographer = cc.mk_agent(
        role="Senior Photographer",
        goal="Take the most amazing photographs for instagram ads that capture emotions and convey a compelling message.",
        backstory="As a Senior Photographer at a leading digital marketing agency, you are an expert at taking amazing "
        "photographs that inspire and engage.",
        max_iter=2,
    )
    director = cc.mk_agent(
        role="Chief Creative Director",
        goal="Oversee the work done by your team to make sure it's the best possible and aligned with the product's "
        "goals, review, approve, ask clarifying question or delegate follow up work if necessary to make decisions",
        backstory="You're the Chief Content Officer of leading digital marketing specialized in product branding. You "
        "make sure your team is crafting the best possible content for the customer.",
        allow_delegation=True,
        max_iter=3,
    )
    take_photo = Task(
        description="You are working on a new campaign for a super important customer, and you MUST describe the most "
        "amazing photo for an instagram post regarding the product. You have the following copy:\n{copy}\n\n"
        "Customer brief:\n{request}\n\n"
        "Describe the photo you want to take in a paragraph, like these examples:\n"
        "- high tech airplane in a beautiful blue sky in a beautiful sunset super crispy beautiful 4k, professional wide shot\n"
        "- an bearded old man in the snows, using very warm clothing, with mountains full of snow behind him, soft "
        "lighting, 4k, crisp, close up to the camera\n"
        "Think creatively. Don't show the actual product on the photo.",
        expected_output="3 options of photographs, each with 1 paragraph describing the photograph like the examples.",
        agent=photographer,
    )
    review = Task(
        description="Review the photos you got from the senior photographer. Make sure they are the best possible and "
        "aligned with the product's goals; review, approve, ask clarifying questions or delegate follow-up work if "
        "necessary. When delegating work send the full draft as part of the information.\n\nCustomer brief:\n{request}",
        expected_output="The final 3 approved photo descriptions plus the chosen ad copy, ready for the customer.",
        agent=director,
        context=[take_photo],
    )
    return Crew(agents=[photographer, director], tasks=[take_photo, review], process=Process.sequential, verbose=False)


def run(task: dict) -> str:
    run_id = cc.run_id_for(AGENT.id, task)
    ad_copy = copy_crew(run_id).kickoff(inputs={"request": task["input"]})
    image = image_crew().kickoff(inputs={"request": task["input"], "copy": ad_copy.raw})
    return f"## Post copy\n{ad_copy.raw}\n\n## Photo briefs\n{image.raw}"
