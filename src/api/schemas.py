from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Financial question",
    )

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Company ticker",
    )


class CompareRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
    )

    ticker_a: str

    ticker_b: str


class HealthResponse(BaseModel):
    status: str
    version: str
    indexed_companies: int
class AnalyzeResponse(BaseModel):
    success: bool
    ticker: str
    query: str
    research: str
    analysis: str
    report: str
    research_latency: float
    analysis_latency: float
    report_latency: float
    total_latency: float

class HealthResponse(BaseModel):
    status: str
    version: str
    indexed_companies: int


class CompaniesResponse(BaseModel):
    tickers: list[str]

class CompareResponse(BaseModel):
    success: bool
    ticker_a: str
    ticker_b: str
    report_a: str
    report_b: str
    comparison: str
    total_latency: float