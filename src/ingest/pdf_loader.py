"""
FinSight Pro
PDF Upload & Indexing Module

Supports indexing arbitrary financial PDF reports into the
existing Chroma vector database.
"""

import logging
import re
import time
from pathlib import Path

from pypdf import PdfReader

from src.ingest.chunker import chunk_text
from langchain_chroma import Chroma

from src.rag.embeddings import get_embedder
from src.rag.vectorstore import PERSIST_DIR

logger = logging.getLogger(__name__)


def extract_pdf_text(filepath: str) -> str:
    """
    Extract all text from a PDF.
    """

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    start = time.perf_counter()

    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    text = "\n".join(pages)

    text = re.sub(r"\n{3,}", "\n\n", text)

    latency = round(time.perf_counter() - start, 2)

    logger.info(
        "Extracted PDF | pages=%d | chars=%d | latency=%.2fs",
        len(reader.pages),
        len(text),
        latency,
    )

    if not text.strip():
        raise ValueError(
            "No extractable text found. "
            "This PDF is likely scanned or image-only."
        )

    return text


def prepare_pdf_chunks(
    filepath: str,
    company_label: str,
):
    """
    Extract text and convert it into reusable chunks.
    """

    text = extract_pdf_text(filepath)

    metadata = {
        "ticker": company_label,
        "filing_date": "user_upload",
        "source": filepath,
    }

    return chunk_text(
        text=text,
        metadata=metadata,
    )

def index_uploaded_pdf(
    filepath: str,
    company_label: str,
    collection_name: str = "user_uploads",
):
    """
    Index a user-uploaded PDF into Chroma.
    """

    start = time.perf_counter()

    chunks = prepare_pdf_chunks(
        filepath=filepath,
        company_label=company_label,
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    embedder = get_embedder()

    Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )

    latency = round(
        time.perf_counter() - start,
        2,
    )

    logger.info(
        "Indexed PDF | company=%s | chunks=%d | latency=%.2fs",
        company_label,
        len(chunks),
        latency,
    )

    return {
        "success": True,
        "company": company_label,
        "filename": Path(filepath).name,
        "chunks": len(chunks),
        "collection": collection_name,
        "latency": latency,
    }


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("FinSight Pro - PDF Indexer")
    print("=" * 70)

    pdf = input("PDF path: ").strip()

    company = input(
        "Company label: "
    ).strip()

    result = index_uploaded_pdf(
        filepath=pdf,
        company_label=company,
    )

    print()

    print("=" * 70)

    print("Indexing Complete")

    print("=" * 70)

    print(f"File       : {result['filename']}")
    print(f"Company    : {result['company']}")
    print(f"Chunks     : {result['chunks']}")
    print(f"Collection : {result['collection']}")
    print(f"Latency    : {result['latency']} sec")

    print("=" * 70)