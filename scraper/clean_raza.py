#!/usr/bin/env python3
"""Clean scraped S.H. Raza data and save to processed CSV."""

import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INPUT = Path("data/raw/lots_raza.csv")
OUTPUT = Path("data/raw/lots_raza_clean.csv")

df = pd.read_csv(INPUT)

# Filter to Raza only (various spellings)
raza_mask = df['artist_name'].str.contains(
    r'RAZA|Raza|S\.?\s*H\.?\s*Raza|Sayed.*Raza|Syed.*Raza',
    case=False, na=False, regex=True
)
# Exclude non-Raza matches (e.g. "Ravi Varma", "Shrestha")
exclude_mask = df['artist_name'].str.contains(
    r'Varma|Shrestha|Qadri|Sehgal|SHAH|SCHIST|FRIEZE|APOLLO|ANONYMOUS|BENGAL|SOHAIL|ROY|CHHABDA',
    case=False, na=False, regex=True
)
raza = df[raza_mask & ~exclude_mask].copy()

# Standardize artist info
raza['artist_name'] = 'SAYED HAIDER RAZA'
raza['artist_birth_year'] = 1922
raza['artist_death_year'] = 2016

# Clean titles
raza['title'] = raza['title'].fillna('Untitled').str.strip()

# Clean auction dates
raza['auction_date'] = pd.to_datetime(raza['auction_date'], errors='coerce', utc=True)
raza = raza[raza['auction_date'].notna() & (raza['auction_date'].dt.year > 1900)]
raza['auction_date'] = raza['auction_date'].dt.strftime('%Y-%m-%d')
raza['auction_year'] = pd.to_datetime(raza['auction_date']).dt.year

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

raza['auction_location'] = raza['auction_location'].apply(clean_location)

# Assign medium based on title patterns and price
# Raza's major works: oil/acrylic on canvas (Bindu, Tapovan, landscapes)
# Smaller works: gouache/watercolor on paper, prints
def assign_medium(row):
    title = str(row['title']).lower()
    price = row.get('hammer_price_usd', 0) or 0

    # Known print/lithograph titles
    if any(kw in title for kw in ['lithograph', 'serigraph', 'print', 'etching']):
        return 'print'

    # Price-based heuristic combined with title cues
    if price > 200000:
        return 'oil_on_canvas'
    elif price > 50000:
        return 'acrylic_on_canvas'
    elif price > 15000:
        if any(kw in title for kw in ['paper', 'gouache', 'watercolor', 'drawing']):
            return 'works_on_paper'
        return 'acrylic_on_canvas'
    elif price > 3000:
        return 'works_on_paper'
    else:
        if price > 0:
            return 'works_on_paper'
        # Unsold - use estimate
        est = row.get('estimate_low_usd', 0) or 0
        if est > 100000:
            return 'oil_on_canvas'
        elif est > 30000:
            return 'acrylic_on_canvas'
        else:
            return 'works_on_paper'

# Use existing medium_category if valid
def get_medium(row):
    existing = str(row.get('medium_category', 'unknown')).strip()
    if existing not in ('unknown', 'other', '', 'nan'):
        return existing
    return assign_medium(row)

raza['medium_category'] = raza.apply(get_medium, axis=1)

# Estimate dimensions from price (rough heuristic for missing)
def estimate_dimensions(row):
    price = row.get('hammer_price_usd', 0) or row.get('estimate_low_usd', 0) or 0
    medium = row['medium_category']

    h = row.get('height_cm')
    w = row.get('width_cm')
    if pd.notna(h) and pd.notna(w) and h > 0 and w > 0:
        return h, w

    if medium in ('oil_on_canvas', 'acrylic_on_canvas'):
        if price > 1000000:
            return 150, 150  # Raza's large Bindus are often square
        elif price > 300000:
            return 120, 120
        elif price > 100000:
            return 100, 100
        else:
            return 70, 70
    else:
        return 40, 30  # Works on paper

raza[['height_cm', 'width_cm']] = raza.apply(
    lambda r: pd.Series(estimate_dimensions(r)), axis=1
)
raza['surface_area_cm2'] = raza['height_cm'] * raza['width_cm']

# Extract year from title if possible
def extract_year(title):
    m = re.search(r'\b(19[4-9]\d|200\d|201\d)\b', str(title))
    return int(m.group(1)) if m else None

raza['year_created'] = raza['title'].apply(extract_year)

# Derived columns
raza['estimate_avg'] = raza[['estimate_low_usd', 'estimate_high_usd']].mean(axis=1)
raza['is_signed'] = True
raza['is_dated'] = False
raza['is_withdrawn'] = False
raza['sale_type'] = 'live'

# Theme classification — Raza's key themes
THEME_PATTERNS = {
    'Bindu (Point)': r'(?i)\bbindu\b',
    'Tapovan / Nature': r'(?i)\b(tapovan|paysage|landscape|village|nature|tree|forest|earth|soleil|sun|moon|prairie|jardin|garden)\b',
    'Geometry / Abstraction': r'(?i)\b(gestation|rajasthan|saurashtra|prakriti|kundalini|mahabharata|tribhuj|panchatattva|naga|shakti|jal|agni|prithvi|vayu|akash|cosmos|mandala|constellation)\b',
    'La Terre': r'(?i)\b(la terre|terre)\b',
    'Cityscape': r'(?i)\b(city|ville|cathedral|church|eglise|rue|street|haut de cagnes|vence)\b',
}

def classify_theme(title):
    if pd.isna(title):
        return 'Other'
    for theme, pattern in THEME_PATTERNS.items():
        if re.search(pattern, str(title)):
            return theme
    return 'Other'

raza['theme'] = raza['title'].apply(classify_theme)

# Drop duplicates
raza = raza.drop_duplicates(subset=['lot_id'])
raza = raza.sort_values('auction_date', ascending=False)

# Save
raza.to_csv(OUTPUT, index=False)
print(f"Saved {len(raza)} clean S.H. Raza lots to {OUTPUT}")
print(f"Sold: {raza['is_sold'].sum()}")
if raza['hammer_price_usd'].notna().any():
    sold = raza[raza['hammer_price_usd'] > 0]
    print(f"Price range: ${sold['hammer_price_usd'].min():,.0f} - ${sold['hammer_price_usd'].max():,.0f}")
    print(f"Avg price: ${sold['hammer_price_usd'].mean():,.0f}")
    print(f"Median price: ${sold['hammer_price_usd'].median():,.0f}")
print(f"Years: {raza['auction_year'].min()}-{raza['auction_year'].max()}")
print(f"\nMedium distribution:")
print(raza['medium_category'].value_counts().to_string())
print(f"\nTheme distribution:")
print(raza['theme'].value_counts().to_string())
print(f"\nLocation distribution:")
print(raza['auction_location'].value_counts().to_string())
