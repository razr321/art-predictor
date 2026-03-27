#!/usr/bin/env python3
"""Clean scraped F.N. Souza data and save to processed CSV."""

import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INPUT = Path("data/raw/lots_souza.csv")
OUTPUT = Path("data/raw/lots_souza_clean.csv")

df = pd.read_csv(INPUT)

# Filter to Souza only
souza_mask = df['artist_name'].str.contains(
    r'SOUZA|Souza|F\.?\s*N\.?\s*Souza|Francis.*Souza',
    case=False, na=False, regex=True
)
# Exclude non-Souza matches
exclude_mask = df['artist_name'].str.contains(
    r'SCHIST|FRIEZE|ANONYMOUS|TANG|DYNASTY|MING|QING|JADE|PORCELAIN|BRONZE|VESSEL',
    case=False, na=False, regex=True
)
souza = df[souza_mask & ~exclude_mask].copy()

# Standardize artist info
souza['artist_name'] = 'FRANCIS NEWTON SOUZA'
souza['artist_birth_year'] = 1924
souza['artist_death_year'] = 2002

# Clean titles
souza['title'] = souza['title'].fillna('Untitled').str.strip()

# Clean auction dates
souza['auction_date'] = pd.to_datetime(souza['auction_date'], errors='coerce', utc=True)
souza = souza[souza['auction_date'].notna() & (souza['auction_date'].dt.year > 1900)]
souza['auction_date'] = souza['auction_date'].dt.strftime('%Y-%m-%d')
souza['auction_year'] = pd.to_datetime(souza['auction_date']).dt.year

# Clean auction locations
def clean_location(loc):
    if pd.isna(loc):
        return 'Unknown'
    loc = str(loc).strip()
    if 'New York' in loc:
        return 'New York'
    if 'South Kensington' in loc:
        return 'London'
    if 'London' in loc:
        return 'London'
    return loc

souza['auction_location'] = souza['auction_location'].apply(clean_location)

# Assign medium based on price and title cues
# Souza: oils on canvas/board (major works), works on paper (drawings, gouache), prints
def assign_medium(row):
    title = str(row['title']).lower()
    price = row.get('hammer_price_usd', 0) or 0

    if any(kw in title for kw in ['lithograph', 'serigraph', 'print', 'etching', 'woodcut']):
        return 'print'
    if any(kw in title for kw in ['drawing', 'sketch', 'pen ', 'ink ']):
        return 'works_on_paper'

    if price > 200000:
        return 'oil_on_canvas'
    elif price > 50000:
        return 'oil_on_canvas'
    elif price > 15000:
        return 'acrylic_on_canvas'
    elif price > 3000:
        return 'works_on_paper'
    else:
        if price > 0:
            return 'works_on_paper'
        est = row.get('estimate_low_usd', 0) or 0
        if est > 80000:
            return 'oil_on_canvas'
        elif est > 20000:
            return 'acrylic_on_canvas'
        else:
            return 'works_on_paper'

def get_medium(row):
    existing = str(row.get('medium_category', 'unknown')).strip()
    if existing not in ('unknown', 'other', '', 'nan'):
        return existing
    return assign_medium(row)

souza['medium_category'] = souza.apply(get_medium, axis=1)

# Estimate dimensions for missing
def estimate_dimensions(row):
    price = row.get('hammer_price_usd', 0) or row.get('estimate_low_usd', 0) or 0
    medium = row['medium_category']

    h = row.get('height_cm')
    w = row.get('width_cm')
    if pd.notna(h) and pd.notna(w) and h > 0 and w > 0:
        return h, w

    if medium in ('oil_on_canvas', 'acrylic_on_canvas'):
        if price > 500000:
            return 120, 90
        elif price > 100000:
            return 90, 70
        elif price > 30000:
            return 70, 55
        else:
            return 50, 40
    else:
        return 35, 25

souza[['height_cm', 'width_cm']] = souza.apply(
    lambda r: pd.Series(estimate_dimensions(r)), axis=1
)
souza['surface_area_cm2'] = souza['height_cm'] * souza['width_cm']

# Extract year from title
def extract_year(title):
    m = re.search(r'\b(19[4-9]\d|200[0-2])\b', str(title))
    return int(m.group(1)) if m else None

souza['year_created'] = souza['title'].apply(extract_year)

# Derived columns
souza['estimate_avg'] = souza[['estimate_low_usd', 'estimate_high_usd']].mean(axis=1)
souza['is_signed'] = True
souza['is_dated'] = False
souza['is_withdrawn'] = False
souza['sale_type'] = 'live'

# Theme classification — Souza's key themes
THEME_PATTERNS = {
    'Heads & Portraits': r'(?i)\b(head|portrait|face|self.portrait|man|woman|girl|boy|figure)\b',
    'Religious': r'(?i)\b(crucifixion|christ|church|cathedral|madonna|pieta|bishop|priest|cardinal|pope|saint|cross|last supper|jesus|christian|nativity|resurrection)\b',
    'Nudes': r'(?i)\b(nude|nudes|reclining|naked|lovers|couple|erotic)\b',
    'Landscape & Cityscape': r'(?i)\b(landscape|city|cityscape|village|street|house|building|goa|church|town)\b',
    'Still Life': r'(?i)\b(still life|flowers|vase|fruit|bottle|table)\b',
}

def classify_theme(title):
    if pd.isna(title):
        return 'Other'
    for theme, pattern in THEME_PATTERNS.items():
        if re.search(pattern, str(title)):
            return theme
    return 'Other'

souza['theme'] = souza['title'].apply(classify_theme)

# Drop duplicates
souza = souza.drop_duplicates(subset=['lot_id'])
souza = souza.sort_values('auction_date', ascending=False)

# Save
souza.to_csv(OUTPUT, index=False)
print(f"Saved {len(souza)} clean F.N. Souza lots to {OUTPUT}")
print(f"Sold: {souza['is_sold'].sum()}")
if souza['hammer_price_usd'].notna().any():
    sold = souza[souza['hammer_price_usd'] > 0]
    print(f"Price range: ${sold['hammer_price_usd'].min():,.0f} - ${sold['hammer_price_usd'].max():,.0f}")
    print(f"Avg price: ${sold['hammer_price_usd'].mean():,.0f}")
    print(f"Median price: ${sold['hammer_price_usd'].median():,.0f}")
print(f"Years: {souza['auction_year'].min()}-{souza['auction_year'].max()}")
print(f"\nMedium distribution:")
print(souza['medium_category'].value_counts().to_string())
print(f"\nTheme distribution:")
print(souza['theme'].value_counts().to_string())
print(f"\nLocation distribution:")
print(souza['auction_location'].value_counts().to_string())
