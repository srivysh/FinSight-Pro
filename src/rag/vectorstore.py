import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma

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


def load_vectorstore():

    embedder = get_embedder()

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedder,
    )


if __name__ == "__main__":

    build_vectorstore()