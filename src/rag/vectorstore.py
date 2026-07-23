import logging
from pathlib import Path

from langchain_chroma import Chroma

from src.ingest.chunker import chunk_filing
from src.rag.embeddings import get_embedder

logger = logging.getLogger(__name__)

PERSIST_DIR = "data/chroma_db"


def build_vectorstore():

    embedder = get_embedder()

    all_chunks = []

    for file in Path("data/filings").glob("*.txt"):

        logger.info("Reading %s", file.name)

        all_chunks.extend(
            chunk_filing(str(file))
        )

    texts = [
        c["text"]
        for c in all_chunks
        if c["text"].strip()
    ]

    metadatas = [
        c["metadata"]
        for c in all_chunks
        if c["text"].strip()
    ]

    logger.info("Embedding %d chunks...", len(texts))

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
    )

    logger.info("Indexed %d chunks.", len(texts))

    return vectorstore

VECTORSTORE = None

VECTORSTORES = {}


def load_vectorstore(collection_name: str = None):
    """
    Load a Chroma vector database only once.

    Args:
        collection_name: Optional Chroma collection name.
                         If None, uses the default collection.
    """

    global VECTORSTORES

    cache_key = collection_name or "default"

    if cache_key not in VECTORSTORES:

        embedder = get_embedder()

        kwargs = {
            "persist_directory": PERSIST_DIR,
            "embedding_function": embedder,
        }

        if collection_name:
            kwargs["collection_name"] = collection_name

        VECTORSTORES[cache_key] = Chroma(**kwargs)

    return VECTORSTORES[cache_key]

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    build_vectorstore()