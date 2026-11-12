"""Stock analysis crew, adapted from crewAI-examples/crews/stock_analysis.

Shape kept from the source: three agents (research analyst, financial analyst, investment advisor)
and four sequential tasks (research -> financial analysis -> filings analysis -> recommendation).
Adapted: SEC/EDGAR scraping and WebsiteSearchTool are replaced by the synthetic finance API and web
search; the source's CalculatorTool is kept as a local tool; the "filings" task works on the price
history (the only filing-like data available) with python_exec for computations.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from crewai import Crew, Process, Task

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-stock-analysis",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis",
    description="Sequential 3-agent crew that researches a ticker, analyses its price data, and writes an investment recommendation.",
    complexity="simple",
    model=cc.HAIKU,
    tracing="openinference",
)


def setup() -> None:
    cc.setup_tracing()


def calculator(operation: str) -> str:
    """Useful to perform any mathematical calculation, like sum, minus, multiplication, division, etc.

    Args:
        operation: A mathematical expression, for example "200*7" or "5000/2*10".
    """
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
           ast.USub: op.neg, ast.Mod: op.mod}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("only arithmetic is allowed")

    try:
        return str(ev(ast.parse(operation.replace(",", ""), mode="eval").body))
    except Exception as e:  # noqa: BLE001 - message goes back to the model
        return f"Error: {e}"


def build_crew(run_id: str) -> Crew:
    calc = cc.local_tool(calculator)
    research_analyst = cc.mk_agent(
        role="Staff Research Analyst",
        goal="Being the best at gathering, interpreting data and amazing your customer with it",
        backstory="Known as the BEST research analyst, you're skilled in sifting through news, company announcements, "
        "and market sentiments. Now you're working on a super important customer. " + cc.WORLD,
        tools=cc.make_tools([T.web_search, T.fetch_url, T.finance_quote], run_id),
        max_iter=4,
    )
    financial_analyst = cc.mk_agent(
        role="The Best Financial Analyst",
        goal="Impress all customers with your financial data and market trends analysis",
        backstory="The most seasoned financial analyst with lots of expertise in stock market analysis and investment "
        "strategies that is working for a super important customer. " + cc.WORLD,
        tools=cc.make_tools([T.finance_quote, T.finance_history, T.python_exec], run_id) + [calc],
        max_iter=5,
    )
    investment_advisor = cc.mk_agent(
        role="Private Investment Advisor",
        goal="Impress your customers with full analyses over stocks and complete investment recommendations",
        backstory="You're the most experienced investment advisor and you combine various analytical insights to "
        "formulate strategic investment advice. You are now working for a super important customer you need to impress. "
        + cc.WORLD,
        tools=cc.make_tools([T.finance_quote], run_id) + [calc],
        max_iter=3,
    )

    research = Task(
        description="Customer request: {request}\n\n"
        "Collect and summarize recent news, press releases, and market analyses related to the stock(s) in the "
        "request and their industry. Pay special attention to any significant events, market sentiments, and "
        "analysts' opinions. Also include upcoming events like earnings and others. Get the latest quote first. "
        "If the request names no ticker, or a ticker the quote tool does not know, say so clearly.",
        expected_output="A report that includes a comprehensive summary of the latest news, any notable shifts in market "
        "sentiment, and potential impacts on the stock. Also return the stock ticker(s).",
        agent=research_analyst,
    )
    financial_analysis = Task(
        description="Customer request: {request}\n\n"
        "Conduct a thorough analysis of the stock's financial health and market performance. Examine the key "
        "metrics available (P/E ratio, market cap, dividend yield, 52-week range, recent returns and volatility). "
        "Also analyze the stock's performance in comparison to its industry peers and overall market trends.",
        expected_output="A clear assessment of the stock's financial standing, its strengths and weaknesses, and how it "
        "fares against its competitors. Use the most recent data possible.",
        agent=financial_analyst,
        context=[research],
    )
    filings_analysis = Task(
        description="Customer request: {request}\n\n"
        "Analyze the price history (the data we have in place of 10-Q and 10-K filings) for the stock in question. "
        "Look for trend changes, drawdowns, unusual volume, gaps, splits or trading halts. Use python_exec for any "
        "non-trivial computation (returns, volatility, max drawdown) on the bars you fetched.",
        expected_output="An expanded report that highlights significant findings from the price history, including any "
        "red flags or positive indicators for the customer.",
        agent=financial_analyst,
        context=[research, financial_analysis],
    )
    recommend = Task(
        description="Customer request: {request}\n\n"
        "Review and synthesize the analyses provided by the Financial Analyst and the Research Analyst. Combine "
        "these insights to form a comprehensive investment recommendation. You MUST consider all aspects, including "
        "financial health, market sentiment, and the price-history findings. Note that prices are synthetic.",
        expected_output="A recommendation for the customer: a well formatted report with a clear investment stance and "
        "strategy with supporting evidence. Answer exactly what the customer asked.",
        agent=investment_advisor,
        context=[research, financial_analysis, filings_analysis],
    )
    return Crew(
        agents=[research_analyst, financial_analyst, investment_advisor],
        tasks=[research, financial_analysis, filings_analysis, recommend],
        process=Process.sequential,
        verbose=False,
    )


def run(task: dict) -> str:
    crew = build_crew(cc.run_id_for(AGENT.id, task))
    return cc.result_text(crew.kickoff(inputs={"request": task["input"]}))
