import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# SEC Headers
# ----------------------------------------------------------------------

HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "Research research@example.com"
    )
}


REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.3
MAX_RETRIES = 3

PIPELINE_VERSION = "1.0"

# ----------------------------------------------------------------------
# HTTP Session with retries
# ----------------------------------------------------------------------

session = requests.Session()

retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)
session.mount("http://", adapter)

# ----------------------------------------------------------------------
# Companies
# ----------------------------------------------------------------------

COMPANY_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
}


# ----------------------------------------------------------------------
# Filing discovery
# ----------------------------------------------------------------------

def get_company_filings(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 1,
) -> List[Dict]:
    """
    Fetch recent SEC filings for a company.

    Parameters
    ----------
    ticker : str
        Stock ticker.
    form_type : str
        SEC form type (default: 10-K).
    limit : int
        Maximum number of filings to return.

    Returns
    -------
    list[dict]
    """

    cik = COMPANY_CIKS[ticker]

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    response = session.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    recent = data["filings"]["recent"]

    results = []

    for i, form in enumerate(recent["form"]):

        if form == form_type:

            results.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "form": form,
                    "filingDate": recent["filingDate"][i],
                    "accessionNumber": recent["accessionNumber"][i],
                    "primaryDocument": recent["primaryDocument"][i],
                }
            )

        if len(results) >= limit:
            break

    return results


# ----------------------------------------------------------------------
# Filing URL
# ----------------------------------------------------------------------

def get_filing_document_url(filing: Dict) -> str:
    """
    Build SEC filing document URL.
    """

    acc = filing["accessionNumber"].replace("-", "")
    cik = int(filing["cik"])

    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{acc}/{filing['primaryDocument']}"
    )


# ----------------------------------------------------------------------
# Download + Clean
# ----------------------------------------------------------------------

def download_and_clean(
    filing: Dict,
    save_dir: str = "data/filings",
) -> str:
    """
    Download a filing, save raw HTML, clean it,
    save cleaned text and metadata.
    """

    logger.info("Processing %s...", filing["ticker"])

    url = get_filing_document_url(filing)

    logger.info("Downloading filing...")

    response = session.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    base_name = f"{filing['ticker']}_{filing['filingDate']}"

    html_path = Path(save_dir) / f"{base_name}.html"
    txt_path = Path(save_dir) / f"{base_name}.txt"
    metadata_path = Path(save_dir) / f"{base_name}.metadata.json"

    # ------------------------------------------------------------------
    # Save raw HTML
    # ------------------------------------------------------------------

    with open(html_path, "wb") as f:
        f.write(response.content)

    logger.info("Saved raw HTML")

    logger.info("Cleaning HTML...")

    soup = BeautifulSoup(response.content, "html.parser")

    removed = {
        "scripts": 0,
        "styles": 0,
        "tables": 0,
        "svg": 0,
        "meta": 0,
        "link": 0,
        "hidden_xbrl": 0,
        "inline_xbrl": 0,
    }

    # Scripts
    tags = soup.find_all("script")
    removed["scripts"] = len(tags)
    for tag in tags:
        tag.decompose()

    # Styles
    tags = soup.find_all("style")
    removed["styles"] = len(tags)
    for tag in tags:
        tag.decompose()

    # Tables
    tags = soup.find_all("table")
    removed["tables"] = len(tags)
    for tag in tags:
        tag.decompose()

    # SVG
    tags = soup.find_all("svg")
    removed["svg"] = len(tags)
    for tag in tags:
        tag.decompose()

    # Meta
    tags = soup.find_all("meta")
    removed["meta"] = len(tags)
    for tag in tags:
        tag.decompose()

    # Link
    tags = soup.find_all("link")
    removed["link"] = len(tags)
    for tag in tags:
        tag.decompose()

    # Hidden XBRL

    hidden = soup.select('div[style*="display:none"]')
    removed["hidden_xbrl"] = len(hidden)

    for tag in hidden:
        tag.decompose()

    # Inline XBRL

    for tag_name in [
        "ix:header",
        "ix:hidden",
        "ix:references",
        "ix:resources",
    ]:

        tags = soup.find_all(tag_name)

        removed["inline_xbrl"] += len(tags)

        for tag in tags:
            tag.decompose()

    logger.info("Extracting visible text...")

    text = soup.get_text(separator="\n")
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove boilerplate

    text = text.replace("Table of Contents", "")

    # Remove weird symbols

    text = re.sub(r"[☒☐®™]", "", text)

    # Remove page headers

    text = re.sub(
        r".*Form\s+10-K\s*\|\s*\d+\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Clean whitespace

    text = re.sub(r"\r", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    lines = []

    previous_blank = False

    for line in text.splitlines():

        line = line.strip()

        if not line:

            if not previous_blank:
                lines.append("")
                previous_blank = True

        else:

            lines.append(line)
            previous_blank = False

    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    logger.info("Saving cleaned text...")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    characters = len(text)
    words = len(text.split())
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "ticker": filing["ticker"],
        "cik": filing["cik"],
        "form": filing["form"],
        "filingDate": filing["filingDate"],
        "downloaded_at": datetime.now().isoformat(),
        "url": url,
        "characters": characters,
        "words": words,
        "paragraphs": paragraphs,
        "raw_html": str(html_path),
        "clean_text": str(txt_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Removed HTML elements:")
    logger.info("  Scripts      : %d", removed["scripts"])
    logger.info("  Styles       : %d", removed["styles"])
    logger.info("  Tables       : %d", removed["tables"])
    logger.info("  SVGs         : %d", removed["svg"])
    logger.info("  Meta tags    : %d", removed["meta"])
    logger.info("  Links        : %d", removed["link"])
    logger.info("  Hidden XBRL  : %d", removed["hidden_xbrl"])
    logger.info("  Inline XBRL  : %d", removed["inline_xbrl"])

    logger.info("Statistics")
    logger.info("Ticker      : %s", filing["ticker"])
    logger.info("Characters  : %s", f"{characters:,}")
    logger.info("Words       : %s", f"{words:,}")
    logger.info("Paragraphs  : %s", f"{paragraphs:,}")
    logger.info("Output      : %s", txt_path)

    logger.info("✓ Saved %s", filing["ticker"])

    return str(txt_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":

    for ticker in COMPANY_CIKS:

        try:

            filings = get_company_filings(
                ticker,
                limit=1,
            )

            if filings:
                download_and_clean(filings[0])
            else:
                logger.warning("No 10-K found for %s", ticker)

        except Exception as e:
            logger.exception("Error processing %s: %s", ticker, e)

        # SEC recommends <= 10 requests/sec
        time.sleep(REQUEST_DELAY)