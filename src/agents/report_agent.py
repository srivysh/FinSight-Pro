"""
FinSight Pro - Report Agent

This module converts a structured financial analysis into a concise,
professional investor report.

Architecture
------------
Research Agent
      │
      ▼
Analysis Agent
      │
      ▼
Report Agent
      │
      ▼
Investor Report
"""

from __future__ import annotations

import logging
import os
import time

from groq import RateLimitError
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.config import (
    GROQ_MODEL,
)

load_dotenv()

logger = logging.getLogger(__name__)

MAX_REPORT_CHARS = 12000

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

REPORT_PROMPT = """
You are FinSight Pro's Report Agent.

ROLE
-----
Your ONLY responsibility is to convert a structured financial analysis
into a concise, professional investor report.

You are NOT another analyst.

=========================================================
STRICT RULES
=========================================================

1. Use ONLY the supplied analysis.

2. Never:
   - invent facts
   - invent numbers
   - invent risks
   - invent financial metrics
   - invent dates

3. Never change:
   - stock prices
   - market capitalization
   - P/E ratios
   - percentages
   - company names
   - currencies

4. Never perform additional financial reasoning.

5. Never provide investment advice.

6. Preserve every factual statement.

7. Do NOT introduce adjectives or conclusions that are not explicitly
   stated in the supplied analysis. Examples include:
   - strong
   - weak
   - healthy
   - stable
   - resilient
   - attractive
   - overvalued
   - undervalued
   - financially strong
   - market leader

8. You may reorganize, shorten, or combine information for readability,
   but you must not reinterpret, summarize beyond the supplied meaning,
   or introduce new conclusions.

9. If the supplied analysis already contains an "Executive Summary",
reuse it with only minor grammatical or formatting changes.
Do not rewrite its meaning.   

=========================================================
ANALYSIS
=========================================================

{analysis}

=========================================================
OUTPUT FORMAT
=========================================================

## Executive Summary

Write 2–3 concise sentences.

---

## Key Findings

• Bullet points

---

## Risk Considerations

• Bullet points

---

## Final Takeaway

Write ONE factual sentence that summarizes the supplied analysis.
Do not add new conclusions or interpretations.

Keep the report below 200 words.
"""


def generate_report(analysis: str) -> str:
    """
    Convert executive analysis into a concise investor report.
    """

    analysis = analysis.strip()

    failure_messages = (
        "Unable to",
        "temporarily unavailable",
    )

    if any(msg in analysis for msg in failure_messages):
        logger.warning(
            "Skipping report generation because analysis failed."
        )
        return analysis

    logger.info("Starting report generation")

    start = time.perf_counter()

    try:

        prompt = REPORT_PROMPT.format(
            analysis=analysis[:MAX_REPORT_CHARS]
        )

        response = llm.invoke(prompt)

        latency = time.perf_counter() - start

        logger.info(
            "Report generated | latency=%.2fs",
            latency,
        )

        report = response.content.strip()

        if not report:
            return "The report agent returned an empty response."

        return report

    except RateLimitError:

        logger.warning(
            "Groq rate limit exceeded during report generation."
        )

        return (
            "Report generation is temporarily unavailable because "
            "the AI service rate limit has been reached."
        )

    except Exception:

        logger.exception(
            "Report generation failed."
        )

        return (
            "Unable to generate the investor report due to an "
            "unexpected error."
        )


if __name__ == "__main__":

    sample_analysis = '''
## Executive Summary

Apple reported stable financial performance.

## Key Financial Metrics

- Stock Price: $326
- Market Cap: $4,796.74 billion

## Business Risks

- Supply chain disruptions
- Regulatory risks

## Overall Assessment

Apple remains financially strong.
'''

    print("=" * 80)
    print("REPORT AGENT")
    print("=" * 80)

    print(generate_report(sample_analysis))