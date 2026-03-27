#!/usr/bin/env python3
"""Discover all Christie's South Asian Modern & Contemporary Art auctions (2015-present).

Uses Selenium to navigate Christie's results pages and extract auction metadata
from window.chrComponents JavaScript objects.

Output: data/raw/auctions.csv
"""

import json
import re
import sys
import time
from datetime import datetime
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

logger = setup_logger(__name__, "discover_auctions.log")

PROGRESS_FILE = DATA_RAW / "auctions_progress.ndjson"
OUTPUT_FILE = DATA_RAW / "auctions.csv"

# Search terms to find SA auctions
SEARCH_QUERIES = [
    "south asian modern contemporary art",
    "south asian modern + contemporary art",
]

# Known auction slugs/patterns for SA art
SA_TITLE_PATTERNS = [
    r"south\s+asian",
    r"indian\s+(?:and\s+south\s+asian|modern|contemporary)",
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


def extract_calendar_events(driver: webdriver.Chrome) -> list[dict]:
    """Extract auction events from window.chrComponents on current page."""
    try:
        data = driver.execute_script("""
            if (window.chrComponents) {
                for (const key of Object.keys(window.chrComponents)) {
                    const comp = window.chrComponents[key];
                    if (comp && comp.data && comp.data.events) {
                        return JSON.stringify(comp.data);
                    }
                }
            }
            return null;
        """)
        if data:
            parsed = json.loads(data)
            return parsed.get("events", [])
    except Exception as e:
        logger.warning(f"Failed to extract calendar events: {e}")
    return []


def extract_next_page_url(driver: webdriver.Chrome) -> str | None:
    """Get URL for next page of results."""
    try:
        data = driver.execute_script("""
            if (window.chrComponents) {
                for (const key of Object.keys(window.chrComponents)) {
                    const comp = window.chrComponents[key];
                    if (comp && comp.data && comp.data.page_next) {
                        return comp.data.page_next;
                    }
                }
            }
            return null;
        """)
        return data
    except Exception:
        return None


def is_south_asian_auction(title: str) -> bool:
    """Check if auction title matches South Asian art patterns."""
    if not title:
        return False
    title_lower = title.lower()
    for pattern in SA_TITLE_PATTERNS:
        if re.search(pattern, title_lower):
            return True
    return False


def parse_event(event: dict) -> dict | None:
    """Parse a Christie's calendar event into a standardized record."""
    title = event.get("title_txt", "")
    if not is_south_asian_auction(title):
        return None

    # Parse date
    date_str = event.get("date_display_txt", "")
    start_date = event.get("start_date", "")
    end_date = event.get("end_date", "")

    # Determine sale type
    landing_url = event.get("landing_url", "")
    is_online = "onlineonly" in landing_url.lower() or "online" in title.lower()
    sale_type = "online" if is_online else "live"

    # Extract sale ID
    sale_id = event.get("event_id", "")
    if not sale_id:
        m = re.search(r"-(\d+)/?$", landing_url)
        if m:
            sale_id = m.group(1)

    # Parse year from date
    year = None
    if start_date:
        try:
            year = int(start_date[:4])
        except (ValueError, IndexError):
            pass

    # Filter: 2015+
    if year and year < 2015:
        return None

    location = event.get("location_txt", "")
    sale_total = event.get("sale_total_value_txt", "")

    return {
        "auction_id": str(sale_id),
        "title": title,
        "url": landing_url,
        "date_display": date_str,
        "start_date": start_date,
        "end_date": end_date,
        "year": year,
        "location": location,
        "sale_type": sale_type,
        "sale_total": sale_total,
    }


def discover_from_search(driver: webdriver.Chrome, query: str) -> list[dict]:
    """Search Christie's results page and paginate through all results."""
    auctions = []
    url = f"https://www.christies.com/en/results?Keyword={query.replace(' ', '+')}&isautosuggestclick=false&saession=past"
    page = 1

    while url:
        logger.info(f"Loading search page {page}: {url[:100]}...")
        try:
            driver.get(url)
            time.sleep(3)

            # Dismiss cookie banner if present
            try:
                cookie_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='cookie-accept'], .chr-cookie-banner button, #onetrust-accept-btn-handler"))
                )
                cookie_btn.click()
                time.sleep(1)
            except Exception:
                pass

            events = extract_calendar_events(driver)
            logger.info(f"  Found {len(events)} events on page {page}")

            for event in events:
                parsed = parse_event(event)
                if parsed:
                    auctions.append(parsed)

            url = extract_next_page_url(driver)
            if url and not url.startswith("http"):
                url = f"https://www.christies.com{url}"
            page += 1
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error on page {page}: {e}")
            break

    return auctions


def discover_from_browse(driver: webdriver.Chrome) -> list[dict]:
    """Browse Christie's by year/month to find SA auctions we might have missed."""
    auctions = []
    current_year = datetime.now().year

    for year in range(2015, current_year + 1):
        for month in [3, 4, 9, 10]:  # Main SA auction months + adjacent
            url = f"https://www.christies.com/en/results?month={month}&year={year}&language=en"
            logger.info(f"Browsing {year}-{month:02d}...")
            try:
                driver.get(url)
                time.sleep(3)

                events = extract_calendar_events(driver)
                for event in events:
                    parsed = parse_event(event)
                    if parsed:
                        auctions.append(parsed)

                time.sleep(1)
            except Exception as e:
                logger.warning(f"Error browsing {year}-{month}: {e}")
                continue

    return auctions


def discover_online_auctions(driver: webdriver.Chrome) -> list[dict]:
    """Check onlineonly.christies.com for online SA auctions."""
    auctions = []
    url = "https://onlineonly.christies.com/s/south-asian-modern-contemporary-art-online/lots/3500"

    try:
        logger.info("Checking onlineonly.christies.com...")
        driver.get(url)
        time.sleep(3)

        # Extract sale data
        data = driver.execute_script("""
            if (window.chrComponents) {
                for (const key of Object.keys(window.chrComponents)) {
                    const comp = window.chrComponents[key];
                    if (comp && comp.data && comp.data.sale) {
                        return JSON.stringify(comp.data.sale);
                    }
                }
            }
            return null;
        """)
        if data:
            sale = json.loads(data)
            logger.info(f"  Found online sale: {sale.get('title_txt', 'unknown')}")
    except Exception as e:
        logger.warning(f"Error checking online auctions: {e}")

    return auctions


def main():
    logger.info("=" * 60)
    logger.info("Discovering Christie's South Asian M&CA auctions (2015-present)")
    logger.info("=" * 60)

    # Load existing progress
    existing = load_ndjson(PROGRESS_FILE)
    seen_ids = {r["auction_id"] for r in existing}
    all_auctions = list(existing)
    logger.info(f"Loaded {len(existing)} previously discovered auctions")

    driver = make_driver()
    try:
        # Phase 1: Search-based discovery
        for query in SEARCH_QUERIES:
            logger.info(f"\n--- Searching: '{query}' ---")
            found = discover_from_search(driver, query)
            new = [a for a in found if a["auction_id"] not in seen_ids]
            for a in new:
                seen_ids.add(a["auction_id"])
                all_auctions.append(a)
                append_ndjson(PROGRESS_FILE, a)
            logger.info(f"  → {len(new)} new auctions from search")

        # Phase 2: Browse by year/month
        logger.info("\n--- Browsing by year/month ---")
        found = discover_from_browse(driver)
        new = [a for a in found if a["auction_id"] not in seen_ids]
        for a in new:
            seen_ids.add(a["auction_id"])
            all_auctions.append(a)
            append_ndjson(PROGRESS_FILE, a)
        logger.info(f"  → {len(new)} new auctions from browse")

        # Phase 3: Check online platform
        discover_online_auctions(driver)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress...")
    finally:
        driver.quit()

    # Save final CSV
    if all_auctions:
        df = pd.DataFrame(all_auctions)
        df = df.drop_duplicates(subset=["auction_id"])
        df = df.sort_values("start_date", ascending=False)
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"\nSaved {len(df)} auctions to {OUTPUT_FILE}")
        logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
        logger.info(f"Sale types: {df['sale_type'].value_counts().to_dict()}")
    else:
        logger.warning("No auctions found!")


if __name__ == "__main__":
    main()
