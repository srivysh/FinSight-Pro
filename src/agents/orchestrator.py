"""
FinSight Pro - Orchestrator

Coordinates the complete multi-agent workflow:

User Query
      │
      ▼
Research Agent
      │
      ▼
Analysis Agent
      │
      ▼
Report Agent
      │
      ▼
Final Investor Report
"""

import logging
import time

from src.agents.analysis_agent import analyze_research
from src.agents.report_agent import generate_report
from src.agents.research_agent import run_research

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def run_pipeline(query: str, ticker: str) -> dict:
    """
    Execute the complete FinSight Pro pipeline.

    Returns:
        dict containing:
        - success
        - ticker
        - query
        - research
        - analysis
        - report
        - latency
    """

    overall_start = time.perf_counter()

    logger.info("Pipeline started | ticker=%s", ticker)

    # ----------------------------
    # Research
    # ----------------------------
    research_start = time.perf_counter()

    research = run_research(
        query=query,
        ticker=ticker,
     )

    research_time = round(
          time.perf_counter() - research_start,
          2,
        )
    
    if (
        not research
        or research.startswith("Unable")
        or research.startswith("Research is temporarily")
    ):
        logger.warning("Pipeline stopped during research stage.")

        return {
            "success": False,
            "ticker": ticker,
            "query": query,
            "stage": "research",
            "message": research,
            "latency": round(time.perf_counter() - overall_start, 2),
        }

    # ----------------------------
    # Analysis
    # ----------------------------
    analysis_start = time.perf_counter()

    analysis = analyze_research(research)

    analysis_time = round(
          time.perf_counter() - analysis_start,
         2,
        )

    if (
        not analysis
        or analysis.startswith("Unable")
        or analysis.startswith("Analysis is temporarily")
    ):
        logger.warning("Pipeline stopped during analysis stage.")

        return {
            "success": False,
            "ticker": ticker,
            "query": query,
            "stage": "analysis",
            "message": analysis,
            "latency": round(time.perf_counter() - overall_start, 2),
        }

    # ----------------------------
    # Report
    # ----------------------------
    report_start = time.perf_counter()

    report = generate_report(analysis)

    report_time = round(
         time.perf_counter() - report_start,
         2,
        )

    if (
        not report
        or report.startswith("Unable")
        or report.startswith("Report generation is temporarily")
    ):
        logger.warning("Pipeline stopped during report stage.")

        return {
            "success": False,
            "ticker": ticker,
            "query": query,
            "stage": "report",
            "message": report,
            "latency": round(time.perf_counter() - overall_start, 2),
        }

    total_latency = round(time.perf_counter() - overall_start, 2)

    logger.info(
        "Pipeline completed successfully | latency=%.2fs",
        total_latency,
    )

    return {
        "success": True,
        "ticker": ticker,
        "query": query,
        "research": research,
        "analysis": analysis,
        "report": report,
        "research_latency": research_time,
        "analysis_latency": analysis_time,
        "report_latency": report_time,
        "total_latency": total_latency,
    }


def main():
    print("=" * 80)
    print("FinSight Pro - Multi-Agent Financial Research")
    print("=" * 80)

    while True:

        ticker = input("\nTicker (or 'exit'): ").strip()

        if ticker.lower() == "exit":
            break

        query = input("Question: ").strip()

        result = run_pipeline(
            query=query,
            ticker=ticker,
        )

        print("\n" + "=" * 80)

        if not result["success"]:
            print(f"Pipeline failed during: {result['stage']}")
            print(result["message"])
            print(f"\nLatency: {result['latency']} sec")
            continue

        print("FINAL INVESTOR REPORT")
        print("=" * 80)
        print(result["report"])

        print("\n")
        print("\nPerformance Summary")
        print("=" * 80)
        print(f"Research : {result['research_latency']} sec")
        print(f"Analysis : {result['analysis_latency']} sec")
        print(f"Report   : {result['report_latency']} sec")
        print("-" * 80)
        print(f"Total    : {result['total_latency']} sec")
        print("=" * 80)

if __name__ == "__main__":
    main()