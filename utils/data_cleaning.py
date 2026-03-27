"""Parsing utilities for Christie's lot data."""

import re
from typing import Optional


def parse_artist_name(raw: str) -> tuple[str, Optional[int], Optional[int]]:
    """Parse 'ARTIST NAME (1900-1980)' → (name, birth_year, death_year)."""
    if not raw:
        return ("", None, None)
    raw = raw.strip()
    # Match trailing (YYYY-YYYY) or (b. YYYY) or (YYYY)
    m = re.search(r"\((?:b\.\s*)?(\d{4})\s*[-–]\s*(\d{4})\)\s*$", raw)
    if m:
        name = raw[: m.start()].strip()
        return (name, int(m.group(1)), int(m.group(2)))
    m = re.search(r"\((?:b\.\s*)(\d{4})\)\s*$", raw)
    if m:
        name = raw[: m.start()].strip()
        return (name, int(m.group(1)), None)
    m = re.search(r"\((\d{4})\s*[-–]\s*(\d{4})\)\s*$", raw)
    if m:
        name = raw[: m.start()].strip()
        return (name, int(m.group(1)), int(m.group(2)))
    return (raw, None, None)


def normalize_artist_name(name: str) -> str:
    """Lowercase, strip extra spaces, remove accents for matching."""
    if not name:
        return ""
    name = name.strip().upper()
    name = re.sub(r"\s+", " ", name)
    return name


def parse_medium(details_text: str) -> str:
    """Extract and normalize medium from lot details text."""
    if not details_text:
        return "unknown"
    text = details_text.lower()
    # Order matters: check most specific first
    medium_patterns = [
        (r"oil\s+on\s+canvas", "oil_on_canvas"),
        (r"oil\s+on\s+board", "oil_on_board"),
        (r"oil\s+on\s+panel", "oil_on_board"),
        (r"oil\s+on\s+paper", "oil_on_paper"),
        (r"oil\s+on\s+masonite", "oil_on_board"),
        (r"acrylic\s+on\s+canvas", "acrylic_on_canvas"),
        (r"acrylic\s+on\s+board", "acrylic_on_board"),
        (r"acrylic\s+on\s+paper", "acrylic_on_paper"),
        (r"gouache\s+on\s+paper", "gouache_on_paper"),
        (r"gouache\s+on\s+board", "gouache_on_board"),
        (r"gouache\s+on\s+canvas", "gouache_on_canvas"),
        (r"gouache", "gouache"),
        (r"watercolou?r\s+on\s+paper", "watercolor_on_paper"),
        (r"watercolou?r", "watercolor"),
        (r"ink\s+(?:and|&)\s+(?:watercolou?r|wash)\s+on\s+paper", "ink_wash_on_paper"),
        (r"ink\s+on\s+paper", "ink_on_paper"),
        (r"ink\s+on\s+silk", "ink_on_silk"),
        (r"pencil\s+on\s+paper", "pencil_on_paper"),
        (r"charcoal\s+on\s+paper", "charcoal_on_paper"),
        (r"pastel\s+on\s+paper", "pastel_on_paper"),
        (r"tempera\s+on\s+(?:canvas|board|paper)", "tempera"),
        (r"mixed\s+media", "mixed_media"),
        (r"bronze", "bronze_sculpture"),
        (r"marble", "marble_sculpture"),
        (r"stone", "stone_sculpture"),
        (r"wood", "wood_sculpture"),
        (r"ceramic", "ceramic"),
        (r"photograph", "photograph"),
        (r"print", "print"),
        (r"lithograph", "lithograph"),
        (r"etching", "etching"),
        (r"screenprint|serigraph", "screenprint"),
        (r"oil", "oil_other"),
        (r"acrylic", "acrylic_other"),
    ]
    for pattern, label in medium_patterns:
        if re.search(pattern, text):
            return label
    return "other"


def parse_dimensions(details_text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract (height_cm, width_cm) from details text.

    Handles formats like:
    - '76.2 x 101.6 cm'
    - '30 x 40 in.'
    - '76.2 x 101.6 cm. (30 x 40 in.)'
    """
    if not details_text:
        return (None, None)
    # Try cm first
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*cm", details_text, re.I)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    # Try inches, convert to cm
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*in", details_text, re.I)
    if m:
        return (float(m.group(1)) * 2.54, float(m.group(2)) * 2.54)
    return (None, None)


def parse_year_created(details_text: str) -> Optional[int]:
    """Extract year created from details text.

    Handles: 'Painted in 1967', 'circa 1970', 'executed in 1955',
    'signed and dated 1972', '1960s' → 1965.
    """
    if not details_text:
        return None
    text = details_text.lower()
    # 'painted in YYYY', 'executed in YYYY', 'dated YYYY'
    m = re.search(r"(?:painted|executed|dated|signed\s+.*?dated)\s+(?:in\s+)?(?:circa\s+)?(\d{4})", text)
    if m:
        return int(m.group(1))
    # 'circa YYYY' standalone
    m = re.search(r"circa\s+(\d{4})", text)
    if m:
        return int(m.group(1))
    # Decade: '1960s' → 1965
    m = re.search(r"(\d{4})s\b", text)
    if m:
        return int(m.group(1)) + 5
    # Range: 1960-1965 → midpoint
    m = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if 1800 < y1 < 2030 and 1800 < y2 < 2030:
            return (y1 + y2) // 2
    # Bare year in reasonable range (as last resort)
    years = re.findall(r"\b((?:19|20)\d{2})\b", text)
    # Filter to reasonable creation years (not birth/death years)
    valid = [int(y) for y in years if 1850 < int(y) < 2026]
    if valid:
        return valid[-1]  # Last mentioned year is usually creation date
    return None


def is_signed(details_text: str) -> bool:
    """Check if artwork is signed based on details text."""
    if not details_text:
        return False
    text = details_text.lower()
    return bool(re.search(r"\bsigned\b", text))


def is_dated(details_text: str) -> bool:
    """Check if artwork is dated (inscribed with date by artist)."""
    if not details_text:
        return False
    text = details_text.lower()
    return bool(re.search(r"\bdated\b", text))


def count_provenance_entries(text: str) -> int:
    """Count provenance chain length from provenance text."""
    if not text or text.strip() == "":
        return 0
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return len(lines)


def count_literature_entries(text: str) -> int:
    """Count literature references."""
    if not text or text.strip() == "":
        return 0
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return len(lines)


def count_exhibition_entries(text: str) -> int:
    """Count exhibition history entries."""
    if not text or text.strip() == "":
        return 0
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return len(lines)


def parse_currency_amount(txt: str) -> tuple[Optional[float], str]:
    """Parse 'USD 50,000' or 'GBP 30,000' → (50000.0, 'USD')."""
    if not txt:
        return (None, "USD")
    txt = txt.strip()
    m = re.match(r"(USD|GBP|EUR|INR|HKD)\s*([\d,]+(?:\.\d+)?)", txt)
    if m:
        currency = m.group(1)
        amount = float(m.group(2).replace(",", ""))
        return (amount, currency)
    # Try just a number
    m = re.search(r"([\d,]+(?:\.\d+)?)", txt)
    if m:
        return (float(m.group(1).replace(",", "")), "USD")
    return (None, "USD")
