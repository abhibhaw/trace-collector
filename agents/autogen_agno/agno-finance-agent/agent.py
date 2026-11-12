"""Finance agent on Agno with the FinanceTools toolkit and a custom provider.

Derived from the Agno cookbook "Market Brief with FinanceTools" (one agent, one FinanceTools toolkit,
"lead with the answer, then show the evidence") and "Bring Your Own Provider" (FinanceProvider subclass
returning the normalized dataclasses). Adapted: Yahoo Finance is replaced by a provider backed by the
tools-service finance quotes (synthetic market data); news is not available from this provider.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

import ag_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="agno-finance-agent",
    framework="agno",
    source_url="https://github.com/agno-agi/agno/blob/main/cookbook/91_tools/finance/06_custom_provider.py",
    description="Agno agent with FinanceTools over a custom provider (quotes, price history, profile, key metrics) for market briefs.",
    complexity="simple",
    model=C.HAIKU,
    tracing="openinference",
)

AS_OF = date(2026, 11, 13)
_PERIOD_DAYS = {"5d": 7, "1mo": 31, "3mo": 92, "6mo": 183, "ytd": None, "1y": 365, "2y": 730, "5y": 1825, "max": 1500}


def _make_provider():
    from agno.tools.finance import (
        CompanyProfile,
        FinanceProvider,
        FinanceProviderError,
        KeyMetrics,
        PriceBar,
        PriceHistory,
        ProviderStatus,
        Quote,
    )

    def quote_raw(symbol: str) -> dict:
        q = T.finance_quote(symbol)
        if "error" in q:  # tools return {"error": ...} for unknown symbols; FinanceTools shows it to the model
            raise FinanceProviderError(q["error"])
        return q

    class PeiraFinance(FinanceProvider):
        id = "peira"
        name = "Peira synthetic market data"
        capabilities = frozenset({"get_quote", "get_price_history", "get_company_profile", "get_key_metrics"})

        def status(self) -> ProviderStatus:
            return ProviderStatus(ok=True, detail="synthetic daily data 2023-01-02..2026-11-13")

        def get_quote(self, symbol: str) -> Quote:
            q = quote_raw(symbol)
            return Quote(
                symbol=q["symbol"], name=q.get("name"), price=q.get("price"), currency=q.get("currency"),
                change=q.get("change"), change_percent=q.get("change_pct"), open=q.get("open"),
                high=q.get("day_high"), low=q.get("day_low"), previous_close=q.get("previous_close"),
                volume=q.get("volume"), market_cap=q.get("market_cap_usd"),
                fifty_two_week_high=q.get("week52_high"), fifty_two_week_low=q.get("week52_low"),
                exchange=q.get("exchange"), as_of=q.get("as_of"),
            )

        def get_price_history(self, symbol: str, period: str = "1mo", interval: str = "1d") -> PriceHistory:
            if period not in _PERIOD_DAYS:
                raise FinanceProviderError(f"period must be one of {sorted(_PERIOD_DAYS)}")
            days = _PERIOD_DAYS[period]
            start = date(AS_OF.year, 1, 1) if days is None else AS_OF - timedelta(days=days)
            iv = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}.get(interval, "daily")
            h = T.finance_history(symbol, start_date=start.isoformat(), end_date=AS_OF.isoformat(), interval=iv)
            if "error" in h:
                raise FinanceProviderError(h["error"])
            bars = [PriceBar(date=b["date"], open=b["open"], high=b["high"], low=b["low"], close=b["close"],
                             volume=b["volume"]) for b in h.get("bars", [])]
            return PriceHistory(symbol=h["symbol"], period=period, interval=interval, currency="USD", bars=bars)

        def get_company_profile(self, symbol: str) -> CompanyProfile:
            q = quote_raw(symbol)
            return CompanyProfile(symbol=q["symbol"], name=q.get("name"), sector=q.get("sector"),
                                  exchange=q.get("exchange"), currency="USD", market_cap=q.get("market_cap_usd"))

        def get_key_metrics(self, symbol: str) -> KeyMetrics:
            q = quote_raw(symbol)
            pe = q.get("pe_ratio")
            return KeyMetrics(
                symbol=q["symbol"], currency="USD", market_cap=q.get("market_cap_usd"), pe_ratio=pe,
                eps=round(q["price"] / pe, 2) if pe else None, dividend_yield=q.get("dividend_yield_pct"),
                fifty_two_week_high=q.get("week52_high"), fifty_two_week_low=q.get("week52_low"), as_of=q.get("as_of"),
            )

    return PeiraFinance()


def setup() -> None:
    C.setup_agno_tracing()


async def arun(task: dict) -> str:
    from agno.agent import Agent
    from agno.tools.finance import FinanceTools

    agent = Agent(
        name="Finance Agent",
        model=C.agno_claude(),
        tools=[FinanceTools(provider=_make_provider())],
        tool_call_limit=8,
        instructions=[
            "Lead with the answer, then show the evidence.",
            "Today is 2026-11-13. All prices are synthetic research data; say so when it matters.",
            "Check price series for splits and trading halts before computing returns, and say how you handled them.",
            "If a metric is missing (for example EPS), say it cannot be computed instead of estimating it.",
            "Never promise returns or give personalized investment advice; never act on non-public information.",
        ],
        markdown=True,
    )
    return C.agno_output(await agent.arun(task["input"]))
