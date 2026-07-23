import logging
import time
from fastapi import FastAPI, Request

from src.api.routes import router

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinSight Pro API",
    description=  ("Production-grade AI Financial Research API powered by "
                 "Multi-Agent Retrieval-Augmented Generation (RAG), "
                 "hybrid retrieval, and financial document analysis."),
    version="1.0.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",},
    license_info={
        "name": "MIT",
          },    

          )

app.include_router(router)

@app.middleware("http")
async def log_requests(request: Request, call_next):

    start = time.perf_counter()

    response = await call_next(request)

    latency = time.perf_counter() - start

    logger.info(
        "%s %s -> %s (%.2fs)",
        request.method,
        request.url.path,
        response.status_code,
        latency,
    )

    return response