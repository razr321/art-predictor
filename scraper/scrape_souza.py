#!/usr/bin/env python3
"""Scrape all F.N. Souza lots from Christie's search results.

Navigates through Christie's search pages using Selenium, extracts lot data
from window.defined JS objects and page HTML, then visits each lot detail page
for provenance/literature/exhibition data.

Output: data/raw/lots_souza.csv
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW, append_ndjson, load_ndjson
from utils.data_cleaning import (
    parse_artist_name,
    parse_medium,
    parse_dimensions,
    parse_year_created,
    is_signed,
    is_dated,
    count_provenance_entries,
    count_literature_entries,
    count_exhibition_entries,
    parse_currency_amount,
)
from utils.currency import to_usd

logger = setup_logger(__name__, "scrape_souza.log")

OUTPUT_FILE = DATA_RAW / "lots_souza.csv"
PROGRESS_FILE = DATA_RAW / "lots_souza_progress.ndjson"
DELAY = 2.0


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


def dismiss_overlays(driver):
    selectors = [
        "#onetrust-accept-btn-handler",
        "[data-testid='cookie-accept']",
        ".chr-cookie-banner button",
        ".chr-modal__close",
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


def extract_search_lots(driver) -> list[dict]:
    """Extract lot cards from Christie's search results page."""
    lots = []
    try:
        # Wait for results to render
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR,
                "[class*='search-results'], [class*='SearchResults'], [class*='lot-tile'], [class*='LotTile'], ul[class*='result']"
            ))
        )
        time.sleep(2)

        # Try extracting from JS state
        js_data = driver.execute_script("""
            // Check for Next.js / React state
            if (window.__NEXT_DATA__) return JSON.stringify(window.__NEXT_DATA__);
            if (window.__INITIAL_STATE__) return JSON.stringify(window.__INITIAL_STATE__);
            if (window.chrComponents) {
                for (const key of Object.keys(window.chrComponents)) {
                    const comp = window.chrComponents[key];
                    if (comp && comp.data && (comp.data.results || comp.data.lots)) {
                        return JSON.stringify(comp.data);
                    }
                }
            }
            return null;
        """)

        if js_data:
            parsed = json.loads(js_data)
            # Navigate to search results in __NEXT_DATA__
            if "props" in parsed:
                page_props = parsed.get("props", {}).get("pageProps", {})
                results = page_props.get("results", [])
                if results:
                    return results
                # Try deeper nesting
                search_data = page_props.get("searchData", {})
                results = search_data.get("results", [])
                if results:
                    return results
            elif "results" in parsed:
                return parsed["results"]
            elif "lots" in parsed:
                return parsed["lots"]

        # Fallback: parse from HTML
        logger.info("  Falling back to HTML parsing...")
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find lot links on the page
        lot_links = soup.select("a[href*='/lot/']")
        seen_urls = set()
        for link in lot_links:
            href = link.get("href", "")
            if href in seen_urls or not href:
                continue
            if "/lot/" not in href:
                continue
            seen_urls.add(href)

            # Try to extract basic info from the card
            card = link.find_parent("li") or link.find_parent("div", class_=re.compile(r"lot|tile|card", re.I))
            lot_data = {"url": href, "_from_html": True}

            if card:
                # Title
                title_el = card.select_one("[class*='title'], h2, h3")
                if title_el:
                    lot_data["title"] = title_el.get_text(strip=True)

                # Price
                price_el = card.select_one("[class*='price'], [class*='Price']")
                if price_el:
                    lot_data["price_text"] = price_el.get_text(strip=True)

                # Estimate
                est_el = card.select_one("[class*='estimate'], [class*='Estimate']")
                if est_el:
                    lot_data["estimate_text"] = est_el.get_text(strip=True)

                # Artist
                artist_el = card.select_one("[class*='artist'], [class*='Artist'], [class*='maker']")
                if artist_el:
                    lot_data["artist_text"] = artist_el.get_text(strip=True)

            lots.append(lot_data)

        logger.info(f"  Extracted {len(lots)} lot links from HTML")
        return lots

    except Exception as e:
        logger.warning(f"Error extracting search lots: {e}")
        return []


def extract_lot_detail_page(driver) -> dict:
    """Extract full lot details from an individual lot page."""
    result = {
        "artist_raw": "",
        "title": "",
        "details_text": "",
        "provenance_text": "",
        "literature_text": "",
        "exhibited_text": "",
        "image_url": "",
        "estimate_low": None,
        "estimate_high": None,
        "estimate_currency": "USD",
        "hammer_price": None,
        "hammer_currency": "USD",
        "is_sold": False,
        "auction_title": "",
        "auction_date": "",
        "auction_location": "",
        "sale_type": "",
        "lot_number": "",
    }

    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR,
                ".chr-lot-header, [class*='lot-header'], [class*='LotHeader'], main"
            ))
        )
        time.sleep(1.5)
        dismiss_overlays(driver)

        # Try JS extraction first
        js_data = driver.execute_script("""
            if (!window.chrComponents) return null;
            for (const key of Object.keys(window.chrComponents)) {
                const comp = window.chrComponents[key];
                if (comp && comp.data && comp.data.lots && comp.data.lots.length > 0) {
                    return JSON.stringify(comp.data);
                }
            }
            // Try lotHeader specifically
            for (const key of Object.keys(window.chrComponents)) {
                if (key.startsWith('lotHeader') || key.startsWith('lot')) {
                    const comp = window.chrComponents[key];
                    if (comp && comp.data) {
                        return JSON.stringify(comp.data);
                    }
                }
            }
            return null;
        """)

        if js_data:
            parsed = json.loads(js_data)
            lots = parsed.get("lots", [parsed] if "object_id" in parsed else [])
            if lots:
                lot = lots[0]
                result["artist_raw"] = lot.get("title_primary_txt", "")
                result["title"] = lot.get("title_secondary_txt", "")
                result["details_text"] = lot.get("description_txt", "")
                result["lot_number"] = lot.get("lot_id_txt", "")

                # Estimates
                result["estimate_low"] = lot.get("estimate_low")
                result["estimate_high"] = lot.get("estimate_high")
                est_txt = lot.get("estimate_txt", "")
                if est_txt:
                    _, cur = parse_currency_amount(est_txt)
                    result["estimate_currency"] = cur or "USD"

                # Price
                result["hammer_price"] = lot.get("price_realised")
                price_txt = lot.get("price_realised_txt", "")
                if price_txt:
                    _, cur = parse_currency_amount(price_txt)
                    result["hammer_currency"] = cur or "USD"

                result["is_sold"] = not lot.get("is_unsold", True)

                # Image
                assets = lot.get("lot_assets", [])
                if assets:
                    result["image_url"] = assets[0].get("image_src", "") or assets[0].get("image_desktop_src", "")
                elif lot.get("image"):
                    result["image_url"] = lot["image"].get("image_src", "")

            # Sale info
            sale = parsed.get("sale", {})
            if not sale:
                for key in ["sale_title_txt", "sale_title"]:
                    if key in parsed:
                        sale = parsed
                        break
            if sale:
                result["auction_title"] = sale.get("sale_title_txt", "") or sale.get("title_txt", "")
                result["auction_date"] = sale.get("start_date", "") or sale.get("date_txt", "")
                result["auction_location"] = sale.get("location_txt", "")
                st = sale.get("sale_type_txt", "").lower()
                result["sale_type"] = "online" if "online" in st else "live"

        # HTML fallback / supplement for details sections
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Details text from HTML if not from JS
        if not result["details_text"]:
            desc_el = soup.select_one("[class*='lot-description'], [class*='LotDescription']")
            if desc_el:
                result["details_text"] = desc_el.get_text(separator="\n", strip=True)

        # Title from HTML if not from JS
        if not result["title"]:
            title_el = soup.select_one("[class*='lot-header'] h1, [class*='LotHeader'] h1")
            if title_el:
                result["title"] = title_el.get_text(strip=True)

        # Artist from HTML if not from JS
        if not result["artist_raw"]:
            artist_el = soup.select_one("[class*='lot-header'] [class*='maker'], [class*='artist']")
            if artist_el:
                result["artist_raw"] = artist_el.get_text(strip=True)

        # Accordion sections (provenance, literature, exhibited)
        all_section_divs = soup.select(".chr-lot-details section, .chr-lot-details .chr-lot-section, [class*='lot-section']")
        for div in all_section_divs:
            label_el = div.select_one("h3, [class*='section__title'], [class*='title']")
            content_el = div.select_one("[class*='accordion--content'], [class*='content']")
            if not label_el or not content_el:
                continue
            label = label_el.get_text(strip=True).lower()
            content = content_el.get_text(separator="\n", strip=True)

            if "provenance" in label:
                result["provenance_text"] = content
            elif "literature" in label:
                result["literature_text"] = content
            elif "exhibit" in label:
                result["exhibited_text"] = content
            elif "detail" in label and not result["details_text"]:
                result["details_text"] = content

        # Estimate from HTML if not from JS
        if not result["estimate_low"]:
            est_el = soup.select_one("[class*='estimate'], [class*='Estimate']")
            if est_el:
                est_text = est_el.get_text(strip=True)
                # Parse "GBP 10,000 - GBP 15,000" or similar
                amounts = re.findall(r'[\d,]+', est_text)
                if len(amounts) >= 2:
                    result["estimate_low"] = int(amounts[0].replace(",", ""))
                    result["estimate_high"] = int(amounts[1].replace(",", ""))
                    _, cur = parse_currency_amount(est_text)
                    result["estimate_currency"] = cur or "USD"

        # Price from HTML if not from JS
        if not result["hammer_price"]:
            price_el = soup.select_one("[class*='price-realised'], [class*='PriceRealised'], [class*='hammer']")
            if price_el:
                price_text = price_el.get_text(strip=True)
                amounts = re.findall(r'[\d,]+', price_text)
                if amounts:
                    result["hammer_price"] = int(amounts[0].replace(",", ""))
                    _, cur = parse_currency_amount(price_text)
                    result["hammer_currency"] = cur or "USD"
                    result["is_sold"] = True

        # Auction info from HTML if not from JS
        if not result["auction_title"]:
            sale_el = soup.select_one("[class*='sale-title'], [class*='SaleTitle'], [class*='auction-title']")
            if sale_el:
                result["auction_title"] = sale_el.get_text(strip=True)

        if not result["auction_date"]:
            date_el = soup.select_one("[class*='sale-date'], [class*='SaleDate'], [class*='auction-date'], time")
            if date_el:
                result["auction_date"] = date_el.get("datetime", "") or date_el.get_text(strip=True)

    except Exception as e:
        logger.warning(f"Error extracting lot detail: {e}")

    return result


def build_record(detail: dict, lot_url: str) -> dict:
    """Convert extracted detail dict into a clean CSV-ready record."""
    artist_name, birth_year, death_year = parse_artist_name(detail["artist_raw"])
    if not artist_name:
        artist_name = "FRANCIS NEWTON SOUZA"

    details_text = detail.get("details_text", "")
    medium = parse_medium(details_text)
    height_cm, width_cm = parse_dimensions(details_text)
    year_created = parse_year_created(details_text)
    signed = is_signed(details_text)
    dated = is_dated(details_text)

    prov_text = detail.get("provenance_text", "")
    lit_text = detail.get("literature_text", "")
    exh_text = detail.get("exhibited_text", "")

    auction_date = detail.get("auction_date", "")
    est_low_usd = to_usd(detail["estimate_low"], detail["estimate_currency"], auction_date)
    est_high_usd = to_usd(detail["estimate_high"], detail["estimate_currency"], auction_date)
    hammer_usd = to_usd(detail["hammer_price"], detail["hammer_currency"], auction_date)

    # Generate a lot_id from URL
    lot_id = ""
    m = re.search(r'/lot/(\d+)', lot_url)
    if m:
        lot_id = m.group(1)
    else:
        lot_id = str(hash(lot_url))[:10]

    return {
        "lot_id": lot_id,
        "lot_number": detail.get("lot_number", ""),
        "auction_id": "",
        "auction_title": detail.get("auction_title", ""),
        "auction_date": auction_date,
        "auction_location": detail.get("auction_location", ""),
        "sale_type": detail.get("sale_type", ""),
        "artist_name": artist_name,
        "artist_birth_year": birth_year,
        "artist_death_year": death_year,
        "title": detail.get("title", ""),
        "medium_raw": details_text[:500],
        "medium_category": medium,
        "height_cm": height_cm,
        "width_cm": width_cm,
        "year_created": year_created,
        "is_signed": signed,
        "is_dated": dated,
        "provenance_text": prov_text[:2000],
        "provenance_count": count_provenance_entries(prov_text),
        "literature_text": lit_text[:2000],
        "literature_count": count_literature_entries(lit_text),
        "exhibited_text": exh_text[:2000],
        "exhibition_count": count_exhibition_entries(exh_text),
        "estimate_low_usd": est_low_usd,
        "estimate_high_usd": est_high_usd,
        "estimate_currency": detail.get("estimate_currency", "USD"),
        "hammer_price_usd": hammer_usd,
        "hammer_currency": detail.get("hammer_currency", "USD"),
        "is_sold": detail.get("is_sold", False),
        "is_withdrawn": False,
        "image_url": detail.get("image_url", ""),
        "lot_url": lot_url,
    }


def get_lot_urls_from_search(driver, base_url: str, max_pages: int = 10) -> list[str]:
    """Paginate through Christie's search results and collect lot URLs."""
    all_urls = []
    seen = set()

    for page in range(1, max_pages + 1):
        url = re.sub(r'page=\d+', f'page={page}', base_url)
        if 'page=' not in url:
            sep = '&' if '?' in url else '?'
            url = f"{url}{sep}page={page}"

        logger.info(f"Loading search page {page}: {url}")
        driver.get(url)
        time.sleep(DELAY + 1)
        dismiss_overlays(driver)

        # Collect lot URLs from the page
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find all links that point to lot detail pages
        lot_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/lot/" in href and href not in seen:
                if not href.startswith("http"):
                    href = f"https://www.christies.com{href}"
                lot_links.add(href)
                seen.add(href)

        if not lot_links:
            logger.info(f"  No lot links found on page {page} — stopping pagination")
            break

        all_urls.extend(sorted(lot_links))
        logger.info(f"  Found {len(lot_links)} lot links (total: {len(all_urls)})")

    return all_urls


def main():
    logger.info("=" * 60)
    logger.info("Scraping F.N. Souza lots from Christie's search")
    logger.info("=" * 60)

    # Load existing progress
    existing = load_ndjson(PROGRESS_FILE)
    scraped_urls = {r["lot_url"] for r in existing}
    all_records = list(existing)
    logger.info(f"Loaded {len(existing)} previously scraped lots")

    driver = make_driver()

    try:
        # Step 1: Collect all lot URLs from search results
        search_url = (
            "https://www.christies.com/en/search?"
            "entry=F.N.%20Souza&page=1&sortby=realized_desc&tab=sold_lots"
        )
        logger.info("Phase 1: Collecting lot URLs from search pages...")
        lot_urls = get_lot_urls_from_search(driver, search_url, max_pages=20)

        # Also try alternate search term
        search_url2 = (
            "https://www.christies.com/en/search?"
            "entry=Francis%20Newton%20Souza&page=1&sortby=realized_desc&tab=sold_lots"
        )
        logger.info("Also searching 'Francis Newton Souza'...")
        lot_urls.extend(get_lot_urls_from_search(driver, search_url2, max_pages=10))

        # Also get unsold lots
        search_url_unsold = (
            "https://www.christies.com/en/search?"
            "entry=F.N.%20Souza&page=1&sortby=relevance&tab=unsold_lots"
        )
        logger.info("Also collecting unsold lot URLs...")
        unsold_urls = get_lot_urls_from_search(driver, search_url_unsold, max_pages=10)
        lot_urls.extend(unsold_urls)

        # Deduplicate
        lot_urls = list(dict.fromkeys(lot_urls))
        logger.info(f"\nTotal unique lot URLs found: {len(lot_urls)}")

        # Filter out already scraped
        new_urls = [u for u in lot_urls if u not in scraped_urls]
        logger.info(f"New lots to scrape: {len(new_urls)}")

        # Step 2: Visit each lot page and extract details
        logger.info("\nPhase 2: Scraping individual lot pages...")
        for i, lot_url in enumerate(tqdm(new_urls, desc="Scraping lots")):
            try:
                driver.get(lot_url)
                time.sleep(DELAY)
                dismiss_overlays(driver)

                detail = extract_lot_detail_page(driver)
                record = build_record(detail, lot_url)
                all_records.append(record)
                append_ndjson(PROGRESS_FILE, record)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i+1}/{len(new_urls)}")

            except Exception as e:
                logger.warning(f"  Failed to scrape {lot_url}: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress...")
    finally:
        driver.quit()

    # Save final CSV
    if all_records:
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["lot_id"])
        df = df.sort_values(["auction_date", "lot_number"], ascending=[False, True])
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"\nSaved {len(df)} lots to {OUTPUT_FILE}")
        logger.info(f"Sold: {df['is_sold'].sum()}, Unsold: {(~df['is_sold']).sum()}")
        if df['hammer_price_usd'].notna().any():
            sold_df = df[df['hammer_price_usd'].notna()]
            logger.info(f"Price range: ${sold_df['hammer_price_usd'].min():,.0f} - ${sold_df['hammer_price_usd'].max():,.0f}")
    else:
        logger.warning("No lots scraped!")


if __name__ == "__main__":
    main()
