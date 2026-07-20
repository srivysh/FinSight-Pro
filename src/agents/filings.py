import logging
from typing import Any

from src.rag.generate import answer_question
from src.config import (
    DEFAULT_TOP_K,
    DEFAULT_RETRIEVAL_METHOD,
)

logger = logging.getLogger(__name__)


def search_company_filings(
    question: str,
    ticker: str,
    retrieval_method: str = "DEFAULT_RETRIEVAL_METHOD",
    k: int = "DEFAULT_TOP_K",
) -> dict[str, Any]:
    """
    Search SEC filings using the FinSight Pro RAG pipeline.

    Args:
        question:
            User's financial question.

        ticker:
            Company ticker (e.g. AAPL).

        retrieval_method:
            "vector" or "hybrid".

        k:
            Number of chunks to retrieve.

    Returns:
        Dictionary containing:
            answer
            confidence
            citations
            sources
    """

    ticker = ticker.strip().upper()

    if not question.strip():
        return {
            "success": False,
            "error": "Question cannot be empty.",
        }

    if not ticker:
        return {
            "success": False,
            "error": "Ticker cannot be empty.",
        }

    logger.info(
        "Searching filings | ticker=%s | method=%s",
        ticker,
        retrieval_method,
    )

    try:

        result = answer_question(
            question=question,
            ticker_filter=ticker,
            retrieval_method=DEFAULT_RETRIEVAL_METHOD,
            k=DEFAULT_TOP_K,
        )

        return {
            "success": True,
            "ticker": ticker,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "citations": result["citations"],
            "sources": result["sources"],
        }

    except Exception:

        logger.exception(
            "Filing search failed for %s",
            ticker,
        )

        return {
            "success": False,
            "error": "Unable to search company filings.",
        }