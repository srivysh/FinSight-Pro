import os
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.rag.retrieve import search
logger = logging.getLogger(__name__)

load_dotenv()

def get_llm() -> ChatGroq:
    """
    Create and return a Groq LLM instance.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Add it to your .env file."
        )

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=api_key,
    )


PROMPT_TEMPLATE = """
You are a financial research assistant.

Answer the user's question using ONLY the context provided below.

Rules:
- Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly:
  "I don't have enough information in the provided filings to answer this."
- Be concise and factual.
- Cite numerical values exactly as they appear.
- Mention the company ticker and filing date when relevant.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def build_context(results) -> str:
    """
    Convert retrieved documents into a formatted context string.
    """

    context_parts = []

    for doc, _ in results:

        ticker = doc.metadata.get("ticker", "Unknown")
        filing_date = doc.metadata.get("filing_date", "Unknown")

        context_parts.append(
            f"[{ticker} | Filed: {filing_date}]\n"
            f"{doc.page_content}"
        )

    separator = "\n\n" + ("-" * 80) + "\n\n"

    return separator.join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """
    Create the final prompt sent to the LLM.
    """

    return PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

def compute_confidence(
    answer: str,
    sources: list,
) -> float:
    """
    Estimate confidence using a simple retrieval-based heuristic.

    Notes:
        This is NOT model uncertainty.
        It is based on retrieval quality only.
    """

    if not sources:
        return 0.0

    if "don't have enough information" in answer.lower():
        return 0.1

    best_score = min(source["score"] for source in sources)

    confidence = max(
        0.0,
        min(
            1.0,
            1.0 - (best_score / 2.0),
        ),
    )

    return round(confidence, 2)

def format_citations(sources: list) -> str:
    """
    Create clean, deduplicated source citations.
    """

    if not sources:
        return "No sources."

    seen = set()
    citations = []

    for source in sources:

        key = (
            source["ticker"],
            source["filing_date"],
        )

        if key not in seen:

            citations.append(
                f"- {source['ticker']} 10-K (Filed: {source['filing_date']})"
            )

            seen.add(key)

    return "\n".join(citations)



def answer_question(
    question: str,
    ticker_filter: str | None = None,
    k: int = 8,
) -> dict:
    """
    Retrieve relevant documents and generate a grounded answer using Groq.

    Args:
        question: User's question.
        ticker_filter: Optional company ticker (e.g., "AAPL").
        k: Number of documents to retrieve.

    Returns:
        Dictionary containing the generated answer and source metadata.
    """

    logger.info("Searching for relevant documents...")

    results = search(
        query=question,
        k=k,
        ticker_filter=ticker_filter,
    )

    if not results:
        logger.warning("No relevant documents found.")

        return {
            "answer": "No relevant documents were found.",
            "sources": [],
        }

    logger.info("Building context from %d retrieved documents.", len(results))

    context = build_context(results)

    prompt = build_prompt(
        question=question,
        context=context,
    )

    llm = get_llm()

    logger.info("Generating response from Groq...")

    try:
        response = llm.invoke(prompt)

    except Exception as exc:
        logger.exception("Groq request failed.")

        return {
            "answer": f"LLM generation failed: {exc}",
            "sources": [],
        }

    sources = [
    {
        "ticker": doc.metadata.get("ticker"),
        "filing_date": doc.metadata.get("filing_date"),
        "chunk_id": doc.metadata.get("chunk_id"),
        "source": doc.metadata.get("source"),
        "score": float(score),
    }
    for doc, score in results
    ]

    confidence = compute_confidence(
    answer=response.content,
    sources=sources,
    )

    citations = format_citations(
    sources=sources,
    )

    logger.info(
    "Answer generated successfully | Confidence: %.2f | Sources Used: %d",
    confidence,
    len(sources),
    )

    return {
    "answer": response.content,
    "confidence": confidence,
    "citations": citations,
    "sources": sources,
     }


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    DEBUG = False

    TICKER = "AMZN"

    QUESTIONS = [
          "What was Amazon's operating income in 2025?"
          ]
    
    TOP_K = 8

    for question in QUESTIONS:

      print("\n" + "#" * 100)
      print(f"QUESTION: {question}")
      print("#" * 100)

      result = answer_question(
        question=question,
        ticker_filter=TICKER,
        k=TOP_K,
        )

      print("\n" + "=" * 80)
      print("ANSWER")
      print("=" * 80)
      print(result["answer"])

      print("\n" + "=" * 80)
      print("CONFIDENCE")
      print("=" * 80)
      print(f"{result['confidence']:.2f}")

      print("\n" + "=" * 80)
      print("CITATIONS")
      print("=" * 80)
      print(result["citations"])

      if DEBUG:

          print("\n" + "=" * 80)
          print("RAW SOURCES")
          print("=" * 80)

          for i, source in enumerate(result["sources"], start=1):
              print(f"\nSource {i}")
              print("-" * 60)
              print(source)