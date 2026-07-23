from src.rag.vectorstore import load_vectorstore

DEFAULT_K = 5


def search(query: str, k: int = DEFAULT_K, ticker_filter: str = None, collection_name: str = None,):
    """
    Search the vector database using semantic similarity.

    Args:
        query: User's search query.
        k: Number of results to return.
        ticker_filter: Restrict search to a specific company ticker.

    Returns:
        List of (Document, score) tuples.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    vs = load_vectorstore(collection_name=collection_name,)

    filter_dict = (
        {"ticker": ticker_filter}
        if ticker_filter
        else None
    )

    results = vs.similarity_search_with_score(
        query=query,
        k=k,
        filter=filter_dict,
    )

    return results


if __name__ == "__main__":

    test_queries = [
        "What was the revenue growth?",
        "What are the main risk factors?",
        "How much did the company spend on research and development?",
        "What is the company's competitive position?",
    ]

    for query in test_queries:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = search(query=query, k=3,ticker_filter="AAPL",)

        if not results:
            print("No matching chunks found.")
            continue

        for i, (doc, score) in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 80)

            print(f"Ticker            : {doc.metadata['ticker']}")
            print(f"Similarity Score  : {score:.3f}")
            print(f"Chunk ID          : {doc.metadata['chunk_id']}")
            print(f"Source File       : {doc.metadata['source']}")

            print("\nPreview:")
            print(doc.page_content[:250])

        print("\n")