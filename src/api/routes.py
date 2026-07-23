from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    AnalyzeRequest,
    CompareRequest,
    AnalyzeResponse,
    HealthResponse,
    CompaniesResponse,
    CompareResponse,
)

from src.agents.orchestrator import run_pipeline
from src.agents.compare import compare_companies
from src.ingest.edgar_client import COMPANY_CIKS

router = APIRouter()


@router.get("/health", tags=["Health"],response_model=HealthResponse,)
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "indexed_companies": len(COMPANY_CIKS),
    }


@router.get("/companies", tags=["Companies"], response_model=CompaniesResponse,)
async def companies():
    return {
        "tickers": sorted(COMPANY_CIKS.keys())
    }


@router.post("/analyze", tags=["Analysis"], response_model=AnalyzeResponse,)
async def analyze(req: AnalyzeRequest):

    ticker = req.ticker.upper()

    if ticker not in COMPANY_CIKS:
        raise HTTPException(
            status_code=400,
            detail=f"{ticker} is not indexed.",
        )

    result = run_pipeline(
        query=req.query,
        ticker=ticker,
    )

    return result


@router.post("/compare", tags=["Comparison"], response_model=CompareResponse,)
async def compare(req: CompareRequest):

    result = compare_companies(
        query=req.query,
        ticker_a=req.ticker_a.upper(),
        ticker_b=req.ticker_b.upper(),
    )

    return result