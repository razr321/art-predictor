#!/usr/bin/env python3
"""Discover Sotheby's Modern & Contemporary South Asian Art auctions (2015-present).

Paginates through Sotheby's search results to find relevant past auctions.
Cards are <li> inside .SearchModule-results, with <a> containing <h3> (title)
and <p> (date | time | location).

Output: data/raw/auctions_sothebys.csv
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW, append_ndjson, load_ndjson

logger = setup_logger(__name__, "discover_auctions_sothebys.log")

PROGRESS_FILE = DATA_RAW / "auctions_sothebys_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "auctions_sothebys.csv"

DEPARTMENT_FILTER = "00000164-609b-d1db-a5e6-e9ff07220000"

# Search multiple queries and date ranges to catch all SA auctions
SEARCH_QUERIES = [
    "south+asian+art",
    "indian+art",
    "contemporary+south+asian",
    "modern+south+asian",
]

# Search in 2-year windows to find older auctions that might not appear in default results
YEAR_RANGES = [
    ("2015-01-01", "2016-12-31"),
    ("2017-01-01", "2018-12-31"),
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
    ("2025-01-01", "2026-12-31"),
]


def build_search_url(query: str, from_date: str = "", to_date: str = "") -> str:
    return (
        f"https://www.sothebys.com/en/results?"
        f"from={from_date}&to={to_date}"
        f"&f2={DEPARTMENT_FILTER}"
        f"&q={query}"
    )

SA_TITLE_PATTERNS = [
    r"south\s+asian",
    r"modern.*contemporary.*south\s+asian",
    r"indian\s+(?:and\s+south\s+asian|art)",
    r"arts?\s+of\s+(?:the\s+)?(?:islamic\s+world|india)",
    r"gods\s+and\s+guardians",
]


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def dismiss_overlays(driver: webdriver.Chrome) -> None:
    selectors = [
        "#onetrust-accept-btn-handler",
        "[data-testid='cookie-accept']",
        "button[class*='cookie']",
        "[aria-label='Close']",
    ]
    for sel in selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass


def is_relevant_auction(title: str) -> bool:
    if not title:
        return False
    title_lower = title.lower()
    for pattern in SA_TITLE_PATTERNS:
        if re.search(pattern, title_lower):
            return True
    return False


def parse_card_element(li_el) -> dict | None:
    """Parse an <li> search result into auction metadata.

    Structure: li > div > a[href*='/auction/'] containing h3 (title) and p (date|time|location)
    """
    try:
        # Find all links that point to auction pages
        links = li_el.find_elements(By.CSS_SELECTOR, "a[href*='/auction/'], a[href*='/buy/auction/']")
        if not links:
            # Try any link
            links = li_el.find_elements(By.TAG_NAME, "a")
        if not links:
            return None

        # Get URL from first auction link
        url = ""
        for link in links:
            href = link.get_attribute("href") or ""
            if "/auction/" in href:
                url = href
                break

        if not url:
            return None

        # Parse title from card text. Layout:
        #   Type: auction
        #   CATEGORY:
        #   PAST AUCTION (or SELLING EXHIBITION)
        #   <Title>                    <-- this is what we want
        #   DATE | TIME | LOCATION
        #   VIEW RESULTS
        title = ""
        full_text = li_el.text or ""
        lines = [l.strip() for l in full_text.split("\n") if l.strip()]
        for i, line in enumerate(lines):
            if line.upper() in ("PAST AUCTION", "SELLING EXHIBITION", "UPCOMING AUCTION"):
                # Title is the next line
                if i + 1 < len(lines):
                    title = lines[i + 1]
                break

        if not title:
            return None

        # Only keep relevant SA auctions
        if not is_relevant_auction(title):
            return None

        # Extract date/time/location from full card text
        # Format: "... DATE | TIME | LOCATION" or "DATE–DATE | TIME | LOCATION"
        date_display = ""
        location = ""
        try:
            full_text = li_el.text or ""
            # Find the line with pipe separators
            for line in full_text.split("\n"):
                line = line.strip()
                if "|" in line:
                    parts = [s.strip() for s in line.split("|")]
                    date_display = parts[0] if parts else ""
                    for part in parts:
                        if any(city in part.upper() for city in ["LONDON", "NEW YORK", "MUMBAI", "HONG KONG", "PARIS", "DOHA"]):
                            location = part.strip()
                            break
                    break
        except Exception:
            pass

        # Parse year from date text or URL
        year = None
        m = re.search(r"20[12]\d", date_display)
        if m:
            year = int(m.group(0))
        if not year:
            m = re.search(r"/auction/(\d{4})/", url)
            if m:
                year = int(m.group(1))

        if year and year < 2015:
            return None

        # Sale type
        is_online = "online" in url.lower() or "online" in title.lower()
        sale_type = "online" if is_online else "live"

        # Auction ID from URL: /en/buy/auction/YEAR/SLUG — use YEAR/SLUG as ID to avoid cross-year dedup
        auction_id = ""
        m = re.search(r"/auction/(\d{4}/[^/?#]+)", url)
        if m:
            auction_id = m.group(1)

        if not auction_id:
            return None

        logger.info(f"    Found: {title} ({year}) — {auction_id}")

        return {
            "auction_id": auction_id,
            "title": title,
            "url": url,
            "date_display": date_display,
            "start_date": "",
            "end_date": "",
            "year": year,
            "location": location,
            "sale_type": sale_type,
            "sale_total": "",
            "source": "sothebys",
        }

    except Exception as e:
        logger.debug(f"Error parsing card: {e}")
        return None


def discover_auctions(driver: webdriver.Chrome) -> list[dict]:
    """Search multiple queries and date ranges to discover auctions."""
    all_auctions = []
    seen_ids = set()

    for query in SEARCH_QUERIES:
        for from_date, to_date in YEAR_RANGES:
            base_url = build_search_url(query, from_date, to_date)
            logger.info(f"\nSearching: q={query}, {from_date} to {to_date}")

            page = 1
            empty_pages = 0
            while empty_pages < 2:
                url = f"{base_url}&p={page}" if page > 1 else base_url
                logger.info(f"  Loading search page {page}...")

                try:
                    driver.get(url)
                    time.sleep(4)
                    dismiss_overlays(driver)

                    try:
                        WebDriverWait(driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".SearchModule-results li, ul li a[href*='/auction/']"))
                        )
                    except Exception:
                        logger.info(f"  No results container on page {page}")
                        empty_pages += 1
                        page += 1
                        continue

                    cards = driver.find_elements(By.CSS_SELECTOR, ".SearchModule-results li")
                    if not cards:
                        cards = driver.find_elements(By.CSS_SELECTOR, "li")
                        cards = [c for c in cards if c.find_elements(By.CSS_SELECTOR, "a[href*='/auction/']")]

                    logger.info(f"  Found {len(cards)} cards on page {page}")

                    if not cards:
                        empty_pages += 1
                        page += 1
                        continue

                    page_auctions = []
                    for card in cards:
                        parsed = parse_card_element(card)
                        if parsed and parsed["auction_id"] not in seen_ids:
                            seen_ids.add(parsed["auction_id"])
                            page_auctions.append(parsed)

                    if page_auctions:
                        empty_pages = 0
                    else:
                        empty_pages += 1

                    logger.info(f"  → {len(page_auctions)} new relevant auctions on page {page}")
                    all_auctions.extend(page_auctions)

                    page += 1
                    time.sleep(3)

                except Exception as e:
                    logger.error(f"Error on page {page}: {e}")
                    empty_pages += 1
                    page += 1

    logger.info(f"\nTotal unique auctions discovered: {len(all_auctions)}")
    return all_auctions


def main():
    logger.info("=" * 60)
    logger.info("Discovering Sotheby's South Asian M&CA auctions (2015-present)")
    logger.info("=" * 60)

    # Clear stale progress from broken run
    if PROGRESS_FILE.exists():
        existing = load_ndjson(PROGRESS_FILE)
    else:
        existing = []
    seen_ids = {r["auction_id"] for r in existing}
    all_auctions = list(existing)
    logger.info(f"Loaded {len(existing)} previously discovered auctions")

    driver = make_driver()
    try:
        found = discover_auctions(driver)
        new = [a for a in found if a["auction_id"] not in seen_ids]
        for a in new:
            seen_ids.add(a["auction_id"])
            all_auctions.append(a)
            append_ndjson(PROGRESS_FILE, a)
        logger.info(f"\n→ {len(new)} new auctions discovered")

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress...")
    finally:
        driver.quit()

    if all_auctions:
        df = pd.DataFrame(all_auctions)
        df = df.drop_duplicates(subset=["auction_id"])
        df = df.sort_values("year", ascending=False)
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"\nSaved {len(df)} auctions to {OUTPUT_FILE}")
        if "year" in df.columns:
            logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    else:
        logger.warning("No auctions found!")


if __name__ == "__main__":
    main()
