from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Split raw text into chunks while preserving metadata.

    This function is reusable for TXT files, PDFs, Word documents,
    or any future document source.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=DEFAULT_SEPARATORS,
    )

    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "metadata": {
                **metadata,
                "chunk_id": i,
            },
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_filing(
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Chunk an SEC filing stored as a text file.
    """

    path = Path(filepath)

    text = path.read_text(encoding="utf-8")

    ticker, filing_date = path.stem.split("_")

    metadata = {
        "ticker": ticker,
        "filing_date": filing_date,
        "source": str(path),
    }

    return chunk_text(
        text=text,
        metadata=metadata,
        chunk_size=chunk_size,
        overlap=overlap,
    )


if __name__ == "__main__":
    import glob

    for f in glob.glob("data/filings/*.txt"):
        chunks = chunk_filing(f)

        avg = (
            sum(len(c["text"]) for c in chunks)
            // len(chunks)
        )

        print(f"{f}: {len(chunks)} chunks, avg {avg} chars")