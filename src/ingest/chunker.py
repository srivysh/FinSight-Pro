from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def chunk_filing(filepath: str, chunk_size=800, overlap=120):
    text = Path(filepath).read_text(encoding="utf-8")
    ticker, date = Path(filepath).stem.split("_")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "metadata": {
                "ticker": ticker,
                "filing_date": date,
                "chunk_id": i,
                "source": filepath,
            }
        }
        for i, chunk in enumerate(chunks)
    ]

if __name__ == "__main__":
    import glob
    for f in glob.glob("data/filings/*.txt"):
        chunks = chunk_filing(f)
        print(f"{f}: {len(chunks)} chunks, avg {sum(len(c['text']) for c in chunks)//len(chunks)} chars")