from langchain_core.tools import tool

from src.agents.market import fetch_stock_snapshot
from src.agents.filings import search_company_filings


@tool
def get_stock_snapshot(ticker: str) -> dict:
    """
    Retrieve a live market snapshot for a publicly traded company.

    The tool returns:
    - Current stock price
    - P/E ratio
    - Market capitalization
    - One-month price change
    """

    return fetch_stock_snapshot(ticker)


@tool
def search_filings(
    question: str,
    ticker: str,
) -> dict:
    """
    Search SEC filings using the FinSight Pro RAG pipeline.

    Best used for:
    - Revenue
    - Risks
    - Financial statements
    - Strategy
    - R&D
    - Business segments
    """

    return search_company_filings(
        question=question,
        ticker=ticker,
    )


if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("MARKET TOOL")
    print("=" * 80)

    market = get_stock_snapshot.invoke(
        {
            "ticker": "AAPL",
        }
    )

    print(market)

    print("\n" + "=" * 80)
    print("FILING TOOL")
    print("=" * 80)

    filing = search_filings.invoke(
        {
            "question": "What were Apple's major risk factors?",
            "ticker": "AAPL",
        }
    )

    print(filing)
