"""
FinSight Pro - Analysis Agent

This module transforms research produced by the Research Agent into
a structured executive financial analysis.

Architecture
------------
User
   │
   ▼
Research Agent
   │
   ▼
Raw Research Summary
   │
   ▼
Analysis Agent
   │
   ▼
Executive Financial Analysis
"""

from __future__ import annotations

import logging
import os
import time
from groq import RateLimitError
from langchain_groq import ChatGroq

from src.agents.research_agent import run_research

from src.config import (
    GROQ_MODEL,
    ANALYSIS_TEMPERATURE,
    MAX_ANALYSIS_CHARS,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=ANALYSIS_TEMPERATURE,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------
ANALYSIS_PROMPT = """
You are FinSight Pro's Financial Analysis Agent.

ROLE
----
You are a senior equity research analyst responsible for transforming
grounded financial research into a concise, structured analysis.

The supplied research has already been collected from trusted sources
such as live market data and SEC filings.

Your responsibility is to ORGANIZE and SYNTHESIZE the evidence.
You must NOT perform additional research.

=========================================================
STRICT RULES
=========================================================

1. Use ONLY the information provided in the research.

2. Never invent:
   - Numbers
   - Dates
   - Financial metrics
   - Business facts
   - Risks
   - Products
   - Company strategies

3. Never estimate missing information.

4. Never assume:
   - Investor sentiment
   - Valuation attractiveness
   - Future stock performance
   - Future company performance

5. Never provide:
   - Buy recommendations
   - Sell recommendations
   - Price targets
   - Investment advice

6. If information is unavailable,
   explicitly state that it is unavailable.

7. If multiple facts describe the same topic,
   merge them into a concise summary without changing their meaning.

8. Every statement in the final report must be traceable to the supplied research.

=========================================================
RESEARCH
=========================================================

{research}

=========================================================
OUTPUT FORMAT
=========================================================

## Executive Summary

Write a brief overview (2–4 sentences) summarizing the company's current
financial position using ONLY the supplied evidence.

---

## Key Financial Metrics

Extract every quantitative metric mentioned.

Present them as bullet points.

If no metrics are available, write:

"Financial metrics were not available in the supplied research."

---

## Business Risks

Summarize only the risks explicitly mentioned.

Use concise bullet points.

If no risks are available, write:

"No business risks were identified in the supplied research."

---

## Important Observations

Only include observations that are directly supported by the supplied evidence.

Do NOT infer:

- Market sentiment
- Valuation quality
- Future growth
- Competitive position
- Investor expectations

If there are no additional supported observations, write:

"No additional observations are directly supported by the available research."

---

## Evidence Confidence

Choose ONE of the following:

High
- Research fully addresses the user's request using relevant evidence.

Medium
- Research answers most of the question but lacks important supporting information.

Low
- Research is incomplete or insufficient to answer the question.

After selecting the confidence level, explain your reasoning in one or two sentences.

---

## Limitations

Identify ONLY information that would have been useful to answer the user's question but was NOT available in the supplied research.

Do NOT list generic financial metrics unless they are relevant to the user's question.

---

## Overall Assessment

Provide a neutral conclusion that summarizes the available evidence.

The conclusion must:

- Be factual.
- Be balanced.
- Avoid speculation.
- Avoid investment recommendations.
- Avoid unsupported interpretations.

End the report after the Overall Assessment section.
"""

# ---------------------------------------------------------------------
# Analysis API
# ---------------------------------------------------------------------
def analyze_research(research: str) -> str:
    """
    Convert research output into a structured executive analysis.

    Parameters
    ----------
    research : str
        Output produced by the Research Agent.

    Returns
    -------
    str
        Structured financial analysis.
    """

    research = research.strip()

    failure_messages = (
        "Unable to complete",
        "Research is temporarily unavailable",
        "The research agent returned an empty response",
    )

    if any(message in research for message in failure_messages):
        logger.warning(
            "Skipping analysis because research was unavailable."
        )
        return research

    logger.info("Starting analysis")

    start = time.perf_counter()

    try:
        research = research[:MAX_ANALYSIS_CHARS]

        prompt = ANALYSIS_PROMPT.format(
            research=research,
        )

        response = llm.invoke(prompt)

        latency = time.perf_counter() - start

        logger.info(
            "Analysis completed | latency=%.2fs",
            latency,
        )

        return response.content.strip()

    except RateLimitError:

        logger.warning(
            "Groq rate limit exceeded during analysis."
        )

        return (
            "Analysis is temporarily unavailable because the AI service "
            "rate limit has been reached. Please wait a few seconds and try again."
        )

    except Exception:

        logger.exception(
            "Analysis failed."
        )

        return (
            "Unable to generate the financial analysis due to an "
            "unexpected error."
        )




# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("FINSIGHT ANALYSIS AGENT")
    print("=" * 80)

    while True:

        ticker = input("\nTicker (or 'exit'): ").strip().upper()

        if ticker.lower() == "exit":
            break

        query = input("Question: ").strip()

        print("\nRunning research...\n")

        research = run_research(
            query=query,
            ticker=ticker,
        )

        print("=" * 80)
        print("RAW RESEARCH")
        print("=" * 80)
        print(research)

        print("\n" + "=" * 80)
        print("EXECUTIVE ANALYSIS")
        print("=" * 80)

        analysis = analyze_research(research)

        print(analysis)