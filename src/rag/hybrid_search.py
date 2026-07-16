import logging
from pathlib import Path
import re

from rank_bm25 import BM25Okapi

from src.ingest.chunker import chunk_filing
from src.rag.retrieve import search as vector_search

DEFAULT_VECTOR_WEIGHT = 0.6

logger = logging.getLogger(__name__)

def tokenize(text: str) -> list[str]:
    """
    Normalize text before BM25 indexing.
    """

    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

def build_bm25_index():
    """
    Build a BM25 index over all filing chunks.
    """

    logger.info("Building BM25 index...")

    all_chunks = []
    if not list(Path("data/filings").glob("*.txt")):
       raise FileNotFoundError(
          "No filing text files found in data/filings."
         )



    for file in Path("data/filings").glob("*.txt"):

        all_chunks.extend(
            chunk_filing(str(file))
        )

    tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in all_chunks
        ]
    
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info(
        "Indexed %d chunks for BM25.",
        len(all_chunks),
    )

    return bm25, all_chunks

BM25_INDEX: BM25Okapi
BM25_CHUNKS: list

BM25_INDEX, BM25_CHUNKS = build_bm25_index()

KNOWN_TICKERS = {
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "AMZN",
}


def preprocess_query(query: str) -> list[str]:
    """
    Tokenize a query and remove ticker symbols when a ticker filter
    already limits the search.
    """

    return [
        token
        for token in tokenize(query)
        if token.upper() not in KNOWN_TICKERS
    ]


def bm25_search(
    query: str,
    k: int = 5,
    ticker_filter: str | None = None,
):
    """
    Search using BM25 keyword matching.

    Returns:
        List of (chunk, score) tuples.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    logger.debug("Running BM25 search...")

    query_tokens = (
    preprocess_query(query)
    if ticker_filter
    else tokenize(query)
    )

    scores = BM25_INDEX.get_scores(query_tokens)

    ranked_results = sorted(
        zip(BM25_CHUNKS, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    if ticker_filter:

        ranked_results = [
            (chunk, score)
            for chunk, score in ranked_results
            if chunk["metadata"]["ticker"] == ticker_filter
        ]

    return ranked_results[:k]


def get_chunk_key(metadata: dict) -> tuple:
    """
    Return a unique identifier for a chunk.
    """

    return (
        metadata["ticker"],
        metadata["filing_date"],
        metadata["chunk_id"],
    )


def hybrid_search(
    query: str,
    k: int = 5,
    ticker_filter: str | None = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
):
    """
    Combine vector search and BM25 search using Reciprocal Rank Fusion (RRF).

    Returns:
        List of (chunk, fusion_score) tuples.
    """

    # Validate inputs
    if k <= 0:
       raise ValueError(
          "k must be greater than 0."
      )

    if not 0 <= vector_weight <= 1:
       raise ValueError(
          "vector_weight must be between 0 and 1."
        )

    logger.info("Running hybrid search...")

    vector_results = vector_search(
        query=query,
        k=k * 2,
        ticker_filter=ticker_filter,
    )

    bm25_results = bm25_search(
        query=query,
        k=k * 2,
        ticker_filter=ticker_filter,
    )

    fused = {}

    # ---------------- Vector ----------------

    for rank, (doc, _) in enumerate(vector_results):

        key = get_chunk_key(doc.metadata)

        if key not in fused:

            fused[key] = {
                "chunk": doc,
                "fusion_score": 0.0,
            }

        fused[key]["fusion_score"] += vector_weight * (1 / (rank + 1))

    # ---------------- BM25 ----------------

    for rank, (chunk, _) in enumerate(bm25_results):

        key = get_chunk_key(chunk["metadata"])

        if key not in fused:

            fused[key] = {
                "chunk": chunk,
                "fusion_score": 0.0,
            }

        fused[key]["fusion_score"] += (1 - vector_weight) * (1 / (rank + 1))

    results = sorted(
        fused.values(),
        key=lambda item: item["fusion_score"],
        reverse=True,
    )

    return results[:k]


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    TEST_QUERIES = [
        "Research and development expense",
        "Risk factors",
        "Deferred revenue",
    ]

    TICKER = "AAPL"
    TOP_K = 5

    def print_results(title, results, search_type):

        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)

        if not results:
            print("No results found.")
            return

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 80)

            if search_type == "vector":

                doc, score = result

                print(f"Distance   : {score:.3f}")
                print(f"Ticker     : {doc.metadata['ticker']}")
                print(f"Chunk ID   : {doc.metadata['chunk_id']}")

                print("\nPreview:")
                print(doc.page_content[:250])

            elif search_type == "bm25":

                chunk, score = result

                print(f"BM25 Score : {score:.2f}")
                print(f"Ticker     : {chunk['metadata']['ticker']}")
                print(f"Chunk ID   : {chunk['metadata']['chunk_id']}")

                print("\nPreview:")
                print(chunk["text"][:250])

            elif search_type == "hybrid":

                fusion_score = result["fusion_score"]
                chunk = result["chunk"]

                if hasattr(chunk, "metadata"):
                    metadata = chunk.metadata
                    text = chunk.page_content
                else:
                    metadata = chunk["metadata"]
                    text = chunk["text"]

                print(f"Fusion Score : {fusion_score:.3f}")
                print(f"Ticker       : {metadata['ticker']}")
                print(f"Chunk ID     : {metadata['chunk_id']}")

                print("\nPreview:")
                print(text[:250])

    for query in TEST_QUERIES:

        print("\n")
        print("#" * 100)
        print(f"QUERY: {query}")
        print("#" * 100)

        vector_results = vector_search(
            query=query,
            ticker_filter=TICKER,
            k=TOP_K,
        )

        bm25_results = bm25_search(
            query=query,
            ticker_filter=TICKER,
            k=TOP_K,
        )

        hybrid_results = hybrid_search(
            query=query,
            ticker_filter=TICKER,
            k=TOP_K,
        )

        print_results(
            "VECTOR SEARCH",
            vector_results,
            "vector",
        )

        print_results(
            "BM25 SEARCH",
            bm25_results,
            "bm25",
        )

        print_results(
            "HYBRID SEARCH",
            hybrid_results,
            "hybrid",
        )