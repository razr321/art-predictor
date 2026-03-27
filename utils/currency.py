"""Currency conversion utilities. Normalizes GBP/EUR/INR to USD."""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
FX_DIR = BASE_DIR / "data" / "fx"
FX_CACHE = FX_DIR / "historical_rates.json"

# Approximate annual average GBP/USD rates (fallback if API unavailable)
_FALLBACK_GBPUSD = {
    2015: 1.53, 2016: 1.36, 2017: 1.29, 2018: 1.33, 2019: 1.28,
    2020: 1.28, 2021: 1.38, 2022: 1.24, 2023: 1.24, 2024: 1.27,
    2025: 1.26, 2026: 1.26,
}

_FALLBACK_EURUSD = {
    2015: 1.11, 2016: 1.11, 2017: 1.13, 2018: 1.18, 2019: 1.12,
    2020: 1.14, 2021: 1.18, 2022: 1.05, 2023: 1.08, 2024: 1.08,
    2025: 1.08, 2026: 1.08,
}

_FALLBACK_INRUSD = {
    2015: 0.0158, 2016: 0.0149, 2017: 0.0154, 2018: 0.0146, 2019: 0.0143,
    2020: 0.0134, 2021: 0.0135, 2022: 0.0126, 2023: 0.0121, 2024: 0.0120,
    2025: 0.0118, 2026: 0.0118,
}

_rate_cache: dict = {}


def _load_cache() -> dict:
    global _rate_cache
    if _rate_cache:
        return _rate_cache
    if FX_CACHE.exists():
        with FX_CACHE.open("r") as f:
            _rate_cache = json.load(f)
    return _rate_cache


def _save_cache() -> None:
    FX_DIR.mkdir(parents=True, exist_ok=True)
    with FX_CACHE.open("w") as f:
        json.dump(_rate_cache, f)


def _fallback_rate(currency: str, year: int) -> float:
    if currency == "GBP":
        return _FALLBACK_GBPUSD.get(year, 1.27)
    elif currency == "EUR":
        return _FALLBACK_EURUSD.get(year, 1.08)
    elif currency == "INR":
        return _FALLBACK_INRUSD.get(year, 0.012)
    elif currency == "HKD":
        return 0.128  # Relatively stable peg
    return 1.0


def to_usd(amount: Optional[float], currency: str, date: str | datetime | None = None) -> Optional[float]:
    """Convert amount in given currency to USD.

    Args:
        amount: The amount to convert
        currency: ISO currency code (USD, GBP, EUR, INR, HKD)
        date: Date for historical rate (str YYYY-MM-DD or datetime)

    Returns:
        Amount in USD, or None if amount is None
    """
    if amount is None:
        return None
    if currency == "USD":
        return amount

    # Determine year for fallback
    year = 2024
    if isinstance(date, str):
        try:
            year = int(date[:4])
        except (ValueError, IndexError):
            pass
    elif isinstance(date, datetime):
        year = date.year

    rate = _fallback_rate(currency, year)
    return round(amount * rate, 2)
