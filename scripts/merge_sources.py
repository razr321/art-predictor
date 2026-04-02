#!/usr/bin/env python3
"""Merge Christie's and Sotheby's lot data into a single lots.csv.

Reads:
  data/raw/lots_christies.csv  (renamed from lots.csv)
  data/raw/lots_sothebys.csv

Output:
  data/raw/lots.csv  (combined, deduplicated)
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import setup_logger, DATA_RAW

logger = setup_logger(__name__, "merge_sources.log")

CHRISTIES_FILE = DATA_RAW / "lots_christies.csv"
SOTHEBYS_FILE = DATA_RAW / "lots_sothebys.csv"
SAFFRONART_FILE = DATA_RAW / "lots_saffronart.csv"
PUNDOLES_FILE = DATA_RAW / "lots_pundoles.csv"
BONHAMS_FILE = DATA_RAW / "lots_bonhams.csv"
OUTPUT_FILE = DATA_RAW / "lots.csv"

# Also support the original lots.csv as Christie's source
CHRISTIES_FALLBACK = DATA_RAW / "lots.csv"

# Individual artist datasets (scraped separately)
ARTIST_FILES = [
    ("lots_tyeb_mehta_clean.csv", "artist_scrape"),
    ("lots_souza_clean.csv", "artist_scrape"),
    ("lots_raza_clean.csv", "artist_scrape"),
]


# ── Artist name normalization ─────────────────────────────────────────────────

# Canonical names for artists with known variants.
# Key = canonical (Title Case), Values = all known raw variants (case-insensitive match).
CANONICAL_NAMES = {
    "Francis Newton Souza": ["FRANCIS NEWTON SOUZA", "F N SOUZA", "F. N. SOUZA"],
    "Maqbool Fida Husain": ["MAQBOOL FIDA HUSAIN", "M F HUSAIN", "M. F. HUSAIN"],
    "Sayed Haider Raza": ["SAYED HAIDER RAZA", "S H RAZA", "S. H. RAZA"],
    "Tyeb Mehta": ["TYEB MEHTA"],
    "Vasudeo S. Gaitonde": ["VASUDEO S. GAITONDE", "V S GAITONDE", "V. S. GAITONDE"],
    "Ram Kumar": ["RAM KUMAR", "RAM KUMAR (B. 1924)", "RAM KUMAR (B.1924)"],
    "Akbar Padamsee": ["AKBAR PADAMSEE"],
    "Jamini Roy": ["JAMINI ROY"],
    "Krishen Khanna": ["KRISHEN KHANNA"],
    "Manjit Bawa": ["MANJIT BAWA"],
    "Bhupen Khakhar": ["BHUPEN KHAKHAR"],
    "Jagdish Swaminathan": ["JAGDISH SWAMINATHAN"],
    "Jehangir Sabavala": ["JEHANGIR SABAVALA"],
    "Ganesh Pyne": ["GANESH PYNE"],
    "K.G. Subramanyan": ["K G SUBRAMANYAN", "K. G. SUBRAMANYAN", "K.G. SUBRAMANYAN"],
    "B. Prabha": ["B PRABHA", "B. PRABHA"],
    "Satish Gujral": ["SATISH GUJRAL"],
    "Jogen Chowdhury": ["JOGEN CHOWDHURY"],
    "Sohan Qadri": ["SOHAN QADRI"],
    "Avinash Chandra": ["AVINASH CHANDRA"],
    "Badri Narayan": ["BADRI NARAYAN"],
    "George Keyt": ["GEORGE KEYT"],
    "Abdur Rahman Chughtai": ["ABDUR RAHMAN CHUGHTAI"],
    "Zarina": ["ZARINA"],
    "Gulam Rasool Santosh": ["GULAM RASOOL SANTOSH"],
    "Bikash Bhattacharjee": ["BIKASH BHATTACHARJEE"],
    "Sakti Burman": ["SAKTI BURMAN"],
    "Laxman Pai": ["LAXMAN PAI"],
    "Thota Vaikuntam": ["THOTA VAIKUNTAM"],
    "Shanti Dave": ["SHANTI DAVE"],
    "Krishnaji Howlaji Ara": ["KRISHNAJI HOWLAJI ARA"],
    "Kattingeri Krishna Hebbar": ["KATTINGERI KRISHNA HEBBAR"],
    "Anwar Jalal Shemza": ["ANWAR JALAL SHEMZA"],
    "Nasreen Mohamedi": ["NASREEN MOHAMEDI"],
    "Bal Chhabda": ["BAL CHHABDA"],
    "Walter Langhammer": ["WALTER LANGHAMMER"],
    "Prabhakar Barwe": ["PRABHAKAR BARWE"],
    "Ganesh Haloi": ["GANESH HALOI"],
    "Narayan Shridhar Bendre": ["NARAYAN SHRIDHAR BENDRE"],
    "Laxman Shrestha": ["LAXMAN SHRESTHA"],
    "Rameshwar Broota": ["RAMESHWAR BROOTA"],
    "Himmat Shah": ["HIMMAT SHAH"],
    "K. Laxma Goud": ["K LAXMA GOUD", "K. LAXMA GOUD"],
    "Anjolie Ela Menon": ["ANJOLIE ELA MENON"],
    "Paresh Maity": ["PARESH MAITY"],
    "Senaka Senanayake": ["SENAKA SENANAYAKE"],
    "Hari Ambadas Gade": ["HARI AMBADAS GADE"],
    "Zainul Abedin": ["ZAINUL ABEDIN"],
    "Krishna Reddy": ["KRISHNA REDDY"],
    "Biren De": ["BIREN DE"],
    "Meera Mukherjee": ["MEERA MUKHERJEE"],
    "Somnath Hore": ["SOMNATH HORE"],
    "Sailoz Mookherjea": ["SAILOZ MOOKHERJEA"],
    "Nicholas Roerich": ["NICHOLAS ROERICH"],
    "Madhvi Parekh": ["MADHVI PAREKH"],
    "Subodh Gupta": ["SUBODH GUPTA"],
    "Atul Dodiya": ["ATUL DODIYA"],
    "Gaganendranath Tagore": ["GAGANENDRANATH TAGORE"],
    "Rabindranath Tagore": ["RABINDRANATH TAGORE"],
    "Paritosh Sen": ["PARITOSH SEN"],
    "Mahadev Visvanath Dhurandhar": ["MAHADEV VISVANATH DHURANDHAR"],
    "Sankho Chaudhuri": ["SANKHO CHAUDHURI"],
    "Jagannath Panda": ["JAGANNATH PANDA"],
    "Lancelot Ribeiro": ["LANCELOT RIBEIRO"],
    "Haku Shah": ["HAKU SHAH"],
    "Rajendra Dhawan": ["RAJENDRA DHAWAN"],
    "Shyamal Dutta Ray": ["SHYAMAL DUTTA RAY"],
    "Mohan Samant": ["MOHAN SAMANT"],
    "Lalu Prasad Shaw": ["LALU PRASAD SHAW"],
    "J. Sultan Ali": ["J SULTAN ALI", "J. SULTAN ALI"],
    "B. Vithal": ["B VITHAL", "B. VITHAL"],
    "K.M. Adimoolam": ["K M ADIMOOLAM", "K.M. ADIMOOLAM"],
    "M. Sivanesan": ["M SIVANESAN", "M. SIVANESAN"],
    "K.C.S. Paniker": ["K C S PANIKER", "K.C.S. PANIKER", "K. C. S. PANIKER"],
    "K.S. Radhakrishnan": ["K S RADHAKRISHNAN", "K.S. RADHAKRISHNAN"],
    "A. Ramachandran": ["A RAMACHANDRAN", "A. RAMACHANDRAN"],
    "G. Ravinder Reddy": ["G RAVINDER REDDY", "G. RAVINDER REDDY"],
    "Kanwal Krishna": ["KANWAL KRISHNA", "KRISHNA KANWAL"],
    "Arpana Caur": ["ARPANA CAUR (B. 1954)", "ARPANA CAUR (B.1954)", "ARPANA CAUR"],
    "Jyoti Bhatt": ["JYOTI BHATT (B. 1934)", "JYOTI BHATT (B.1934)", "JYOTI BHATT"],
    "T.V. Santhosh": ["T.V. SANTHOSH (B. 1968)", "T. V. SANTHOSH (B. 1968)"],
    "N.N. Rimzon": ["N.N. RIMZON (B. 1957)", "N.N. Rimzon (B. 1957)"],
    "Prodosh Das Gupta": ["PRODOSH DAS GUPTA"],
    "Rashid Choudhury": ["RASHID CHOUDHURY"],
    "Mohammad Kibria": ["MOHAMMAD KIBRIA"],
    "Ahmed Parvez": ["AHMED PARVEZ"],
    "Abdulrahim Apabhai Almelkar": ["ABDULRAHIM APABHAI ALMELKAR"],
    "Ambadas Khobragade": ["AMBADAS KHOBRAGADE"],
    "Arup Das": ["ARUP DAS"],
    "Raja Ravi Varma": ["RAJA RAVI VARMA"],
    "Amrita Sher-Gil": ["AMRITA SHER-GIL"],
    "Gulam Mohammed Sheikh": ["GULAM MOHAMMED SHEIKH"],
    "Sheela Gowda": ["SHEELA GOWDA"],
}

# Build reverse lookup: uppercase variant → canonical name
_VARIANT_TO_CANONICAL = {}
for canonical, variants in CANONICAL_NAMES.items():
    _VARIANT_TO_CANONICAL[canonical.upper()] = canonical
    for v in variants:
        _VARIANT_TO_CANONICAL[v.upper()] = canonical


def _strip_birth_year(name: str) -> str:
    """Remove '(B. 1954)' or '(1913-2011)' suffixes."""
    return re.sub(r"\s*\((?:B\.?\s*)?\d{4}(?:\s*[-–]\s*\d{4})?\)\s*$", "", name, flags=re.I).strip()


def _to_title_case(name: str) -> str:
    """Convert 'FRANCIS NEWTON SOUZA' → 'Francis Newton Souza', preserving initials."""
    parts = name.split()
    result = []
    for p in parts:
        # Keep single letters or initials like "K.G." as-is but capitalize
        if len(p) <= 3 and p.replace(".", "").isalpha():
            result.append(p.upper() if len(p) == 1 else p[0].upper() + p[1:].lower())
        else:
            result.append(p.capitalize())
    return " ".join(result)


def _normalize_artist_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize artist_name to canonical Title Case names."""
    if "artist_name" not in df.columns:
        return df

    original_unique = df["artist_name"].nunique()

    def normalize(name):
        if pd.isna(name) or not str(name).strip():
            return name
        name = str(name).strip()

        # 1. Try exact match in variant lookup (case-insensitive)
        upper = name.upper()
        if upper in _VARIANT_TO_CANONICAL:
            return _VARIANT_TO_CANONICAL[upper]

        # 2. Try after stripping birth year suffix
        stripped = _strip_birth_year(name)
        upper_stripped = stripped.upper()
        if upper_stripped in _VARIANT_TO_CANONICAL:
            return _VARIANT_TO_CANONICAL[upper_stripped]

        # 3. Auto-normalize: if ALL CAPS, convert to Title Case
        if name == name.upper() and len(name) > 3:
            stripped = _strip_birth_year(name)
            return _to_title_case(stripped)

        return name

    df["artist_name"] = df["artist_name"].apply(normalize)
    new_unique = df["artist_name"].nunique()
    logger.info(f"Normalized artist names: {original_unique} → {new_unique} unique ({original_unique - new_unique} merged)")

    return df


def main():
    logger.info("=" * 60)
    logger.info("Merging auction house data")
    logger.info("=" * 60)

    frames = []

    # Christie's
    christies_path = CHRISTIES_FILE if CHRISTIES_FILE.exists() else CHRISTIES_FALLBACK
    if christies_path.exists():
        df_c = pd.read_csv(christies_path)
        # Ensure lot_ids don't collide — prefix if needed
        if not df_c["lot_id"].astype(str).str.startswith("christies_").any():
            # Only add prefix if not already prefixed and IDs are numeric
            if df_c["lot_id"].astype(str).str.isnumeric().all():
                df_c["lot_id"] = "christies_" + df_c["lot_id"].astype(str)
        df_c["source"] = "christies"
        frames.append(df_c)
        logger.info(f"Christie's: {len(df_c)} lots from {christies_path.name}")
    else:
        logger.warning("No Christie's data found")

    # Sotheby's
    if SOTHEBYS_FILE.exists():
        df_s = pd.read_csv(SOTHEBYS_FILE)
        df_s["source"] = "sothebys"
        frames.append(df_s)
        logger.info(f"Sotheby's: {len(df_s)} lots from {SOTHEBYS_FILE.name}")
    else:
        logger.warning("No Sotheby's data found")

    # Saffronart
    if SAFFRONART_FILE.exists():
        df_sa = pd.read_csv(SAFFRONART_FILE)
        df_sa["source"] = "saffronart"
        frames.append(df_sa)
        logger.info(f"Saffronart: {len(df_sa)} lots from {SAFFRONART_FILE.name}")
    else:
        logger.warning("No Saffronart data found")

    # Pundole's
    if PUNDOLES_FILE.exists():
        df_p = pd.read_csv(PUNDOLES_FILE)
        df_p["source"] = "pundoles"
        frames.append(df_p)
        logger.info(f"Pundole's: {len(df_p)} lots from {PUNDOLES_FILE.name}")
    else:
        logger.warning("No Pundole's data found")

    # Bonhams
    if BONHAMS_FILE.exists():
        df_b = pd.read_csv(BONHAMS_FILE)
        df_b["source"] = "bonhams"
        frames.append(df_b)
        logger.info(f"Bonhams: {len(df_b)} lots from {BONHAMS_FILE.name}")
    else:
        logger.warning("No Bonhams data found")

    # Individual artist datasets
    for fname, source_label in ARTIST_FILES:
        fpath = DATA_RAW / fname
        if fpath.exists():
            df_a = pd.read_csv(fpath)
            if "source" not in df_a.columns:
                df_a["source"] = source_label
            frames.append(df_a)
            logger.info(f"Artist file: {len(df_a)} lots from {fname}")
        else:
            logger.debug(f"Artist file not found: {fname}")

    if not frames:
        logger.error("No data to merge!")
        sys.exit(1)

    # Combine
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["lot_id"])
    df = df.sort_values(["auction_date", "lot_number"], ascending=[False, True])

    # Normalize artist names
    df = _normalize_artist_names(df)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"\nMerged: {len(df)} total lots → {OUTPUT_FILE}")
    logger.info(f"By source: {df['source'].value_counts().to_dict()}")
    logger.info(f"Sold: {df['is_sold'].sum()}")
    logger.info(f"Unique artists: {df['artist_name'].nunique()}")


if __name__ == "__main__":
    main()
