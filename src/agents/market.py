import logging
from typing import Any

import yfinance as yf

from src.config import (
    DEFAULT_HISTORY_PERIOD,
    MARKET_CAP_DIVISOR,
)

logger = logging.getLogger(__name__)


def _calculate_one_month_change(history) -> float | None:
    """
    Calculate the percentage price change over the last month.
    """

    if history.empty or len(history) < 2:
        return None

    start_price = float(history["Close"].iloc[0])
    end_price = float(history["Close"].iloc[-1])

    if start_price == 0:
        return None

    return round(((end_price - start_price) / start_price) * 100, 2)


def fetch_stock_snapshot(ticker: str) -> dict[str, Any]:
    """
    Fetch a live market snapshot for a stock.

    Returns:
        {
            "success": bool,
            "ticker": str,
            "company": str,
            "current_price": float,
            "currency": str,
            "pe_ratio": float | None,
            "market_cap": int | None,
            "market_cap_billions": float | None,
            "one_month_change_pct": float | None
        }
    """

    ticker = ticker.strip().upper()

    if not ticker:
        return {
            "success": False,
            "error": "Ticker symbol cannot be empty."
        }

    logger.info("Fetching live market data for %s", ticker)

    try:
        stock = yf.Ticker(ticker)

        fast_info = stock.fast_info
        info = stock.info
        history = stock.history(period=DEFAULT_HISTORY_PERIOD)

        if history.empty:
            logger.warning("No history found for %s", ticker)

            return {
                "success": False,
                "error": f"No market data found for ticker '{ticker}'."
            }

        current_price = fast_info.get("lastPrice", info.get("currentPrice"))
        

        market_cap = fast_info.get("marketCap", info.get("marketCap"))

        pe_ratio = info.get("trailingPE")

        company = (
            info.get("longName")
            or info.get("shortName")
            or ticker
        )

        currency = info.get("currency", "USD")

        one_month_change = _calculate_one_month_change(history)

        logger.info("Successfully fetched market data for %s", ticker)

        if market_cap is not None:
          market_cap = int(market_cap)

        return {
            "success": True,
            "ticker": ticker,
            "company": company,
            "current_price": current_price,
            "currency": currency,
            "pe_ratio": pe_ratio,
            "market_cap": market_cap,
            "market_cap_billions": (
                round(market_cap / MARKET_CAP_DIVISOR, 2)
                if market_cap
                else None
            ),
            "one_month_change_pct": one_month_change,
        }

    except Exception:
        logger.exception("Failed fetching market data for %s", ticker)

        return {
            "success": False,
            "error": "Unable to retrieve market data."
        }