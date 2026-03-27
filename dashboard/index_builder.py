"""
Price index computation module for the Indian Art Market dashboard.

Builds median, top-tier, and hedonic (quality-adjusted) price indices
for different artist segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Artist name variants  (values are UPPERCASED to match artist_name_clean)
# ---------------------------------------------------------------------------
STAR_ARTISTS: dict[str, list[str]] = {
    "F.N. Souza": ["FRANCIS NEWTON SOUZA", "F N SOUZA"],
    "S.H. Raza": ["SAYED HAIDER RAZA", "S H RAZA"],
    "Tyeb Mehta": ["TYEB MEHTA"],
    "M.F. Husain": ["MAQBOOL FIDA HUSAIN", "M F HUSAIN"],
    "V.S. Gaitonde": ["VASUDEO S. GAITONDE", "V S GAITONDE"],
    "Akbar Padamsee": ["AKBAR PADAMSEE"],
    "Ram Kumar": ["RAM KUMAR"],
}

CORE_4: list[str] = ["F.N. Souza", "S.H. Raza", "Tyeb Mehta", "M.F. Husain"]

# Flat set of all uppercase variants for quick lookup
_ALL_STAR_VARIANTS: set[str] = {v for variants in STAR_ARTISTS.values() for v in variants}
_CORE4_VARIANTS: set[str] = {
    v for name in CORE_4 for v in STAR_ARTISTS[name]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def match_artists(df: pd.DataFrame, artist_set: set[str]) -> pd.Series:
    """Return a boolean mask where artist_name_clean matches any name in *artist_set*."""
    return df["artist_name_clean"].isin(artist_set)


def _rebase(series: pd.Series) -> pd.Series:
    """Rebase a series so the first non-null value equals 100."""
    first = series.dropna().iloc[0] if not series.dropna().empty else 1.0
    return (series / first) * 100


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def simple_index(df: pd.DataFrame) -> pd.Series:
    """
    Median hammer_price_usd per auction_year, rebased to 100 at the
    earliest year.  Only sold lots are included.
    """
    sold = df[df["is_sold"] == 1]
    median_by_year = sold.groupby("auction_year")["hammer_price_usd"].median()
    return _rebase(median_by_year).rename("median_index")


def top_tier_index(df: pd.DataFrame) -> pd.Series:
    """
    75th-percentile hammer_price_usd per auction_year, rebased to 100.
    Captures movement in the upper market segment.
    """
    sold = df[df["is_sold"] == 1]
    p75_by_year = sold.groupby("auction_year")["hammer_price_usd"].quantile(0.75)
    return _rebase(p75_by_year).rename("top_tier_index")


def hedonic_index(
    df: pd.DataFrame,
    predicted_usd: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Quality-adjusted hedonic index.

    For each sold lot the ratio actual / predicted is computed.  The yearly
    median of those ratios captures pure price-level changes net of
    compositional shifts in the lots offered.  Result is rebased to 100.

    Parameters
    ----------
    df : DataFrame
        Must include auction_year, is_sold, hammer_price_usd.
    predicted_usd : array-like
        Model-predicted prices aligned to *df* index.
    """
    tmp = df.copy()
    tmp["predicted_usd"] = np.asarray(predicted_usd)
    sold = tmp[(tmp["is_sold"] == 1) & (tmp["predicted_usd"] > 0)]
    sold = sold.assign(ratio=sold["hammer_price_usd"] / sold["predicted_usd"])
    ratio_by_year = sold.groupby("auction_year")["ratio"].median()
    return _rebase(ratio_by_year).rename("hedonic_index")


def compute_cagr(index_series: pd.Series) -> float:
    """
    Compound Annual Growth Rate from the first to last entry in an index
    series whose index is auction_year.

    Returns the CAGR as a decimal (e.g. 0.07 for 7 %).  Returns 0.0 when
    the series has fewer than two data points or the starting value is zero.
    """
    s = index_series.dropna().sort_index()
    if len(s) < 2 or s.iloc[0] == 0:
        return 0.0
    n_years = s.index[-1] - s.index[0]
    if n_years <= 0:
        return 0.0
    return (s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def _identify_full_star_set(df: pd.DataFrame, top_n_years: int = 10) -> set[str]:
    """
    'Full Star' set: artists whose maximum single-lot hammer price in the
    most recent *top_n_years* years is >= $500,000.
    """
    max_year = df["auction_year"].max()
    recent = df[
        (df["auction_year"] >= max_year - top_n_years + 1) & (df["is_sold"] == 1)
    ]
    max_prices = recent.groupby("artist_name_clean")["hammer_price_usd"].max()
    qualifying = set(max_prices[max_prices >= 500_000].index)
    return qualifying


def build_all_indices(
    master_df: pd.DataFrame,
    ml_df: pd.DataFrame | None = None,
    predicted_usd: pd.Series | np.ndarray | None = None,
) -> dict:
    """
    Build price indices for three segments:

    * **core4** -- the four marquee Modern masters
    * **full_star** -- any artist with a $500K+ lot in the last 10 years
    * **full_market** -- all sold lots

    Returns a dict keyed by segment name.  Each value is a dict with keys
    ``median``, ``top_tier``, optionally ``hedonic``, and ``cagr_median``.

    Parameters
    ----------
    master_df : DataFrame
        The master dataset.
    ml_df : DataFrame, optional
        If provided together with *predicted_usd*, used for hedonic index.
        Must have the same rows as *predicted_usd*.
    predicted_usd : array-like, optional
        ML model predictions aligned to *ml_df*.
    """
    results: dict = {}

    segments: dict[str, pd.DataFrame] = {
        "core4": master_df[match_artists(master_df, _CORE4_VARIANTS)].copy(),
        "full_star": master_df[
            match_artists(master_df, _identify_full_star_set(master_df))
        ].copy(),
        "full_market": master_df.copy(),
    }

    for seg_name, seg_df in segments.items():
        entry: dict = {}
        entry["median"] = simple_index(seg_df)
        entry["top_tier"] = top_tier_index(seg_df)
        entry["cagr_median"] = compute_cagr(entry["median"])
        entry["cagr_top_tier"] = compute_cagr(entry["top_tier"])

        # Hedonic index (only if ML predictions are available)
        if ml_df is not None and predicted_usd is not None:
            # Intersect ml_df rows that belong to this segment
            seg_mask = match_artists(ml_df, set(seg_df["artist_name_clean"].unique()))
            if seg_mask.sum() > 0:
                entry["hedonic"] = hedonic_index(
                    ml_df[seg_mask], np.asarray(predicted_usd)[seg_mask.values]
                )
                entry["cagr_hedonic"] = compute_cagr(entry["hedonic"])

        results[seg_name] = entry

    return results
