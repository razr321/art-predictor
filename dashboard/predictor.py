"""Price prediction logic: load models, build feature vectors, predict."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MANIFEST_FILE = MODELS_DIR / "model_manifest.json"


class ArtPredictor:
    def __init__(self):
        self.manifest = None
        self.cb_models = []
        self.xgb_models = []
        self.master_df = None
        self.ml_df = None
        self._label_encoders = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        # Load manifest
        with MANIFEST_FILE.open() as f:
            self.manifest = json.load(f)

        # Load CatBoost models
        for fname in self.manifest["catboost_models"]:
            m = CatBoostRegressor()
            m.load_model(str(MODELS_DIR / fname))
            self.cb_models.append(m)

        # Load XGBoost models
        for fname in self.manifest["xgboost_models"]:
            m = XGBRegressor()
            m.load_model(str(MODELS_DIR / fname))
            self.xgb_models.append(m)

        # Load data for artist lookup
        master_path = DATA_PROCESSED / "master.csv"
        if master_path.exists():
            self.master_df = pd.read_csv(master_path)
            self.master_df["auction_date"] = pd.to_datetime(self.master_df["auction_date"], errors="coerce")

        ml_path = DATA_PROCESSED / "ml_ready.csv"
        if ml_path.exists():
            self.ml_df = pd.read_csv(ml_path)
            # Build label encoders for XGBoost
            for col in self.manifest.get("cat_cols", []):
                if col in self.ml_df.columns:
                    vals = self.ml_df[col].fillna("unknown").astype(str).unique()
                    self._label_encoders[col] = {v: i for i, v in enumerate(vals)}

        self._loaded = True

    def get_artists(self) -> list[str]:
        """Get list of all artists in the dataset."""
        self.load()
        if self.master_df is not None:
            return sorted(self.master_df["artist_name"].dropna().unique().tolist())
        return []

    def get_artist_stats(self, artist_name: str) -> dict:
        """Get summary stats for an artist."""
        self.load()
        if self.master_df is None:
            return {}
        mask = self.master_df["artist_name"].str.upper() == artist_name.upper()
        artist_df = self.master_df[mask]
        if artist_df.empty:
            return {}

        sold = artist_df[artist_df["is_sold"] == True]
        return {
            "total_lots": len(artist_df),
            "lots_sold": len(sold),
            "sell_through_rate": len(sold) / len(artist_df) if len(artist_df) > 0 else 0,
            "avg_price": sold["hammer_price_usd"].mean() if len(sold) > 0 else 0,
            "median_price": sold["hammer_price_usd"].median() if len(sold) > 0 else 0,
            "max_price": sold["hammer_price_usd"].max() if len(sold) > 0 else 0,
            "min_price": sold["hammer_price_usd"].min() if len(sold) > 0 else 0,
            "first_sale": str(artist_df["auction_date"].min().date()) if artist_df["auction_date"].notna().any() else "",
            "last_sale": str(artist_df["auction_date"].max().date()) if artist_df["auction_date"].notna().any() else "",
        }

    def get_artist_history(self, artist_name: str) -> pd.DataFrame:
        """Get all lots for an artist (for time series)."""
        self.load()
        if self.master_df is None:
            return pd.DataFrame()
        mask = self.master_df["artist_name"].str.upper() == artist_name.upper()
        return self.master_df[mask].sort_values("auction_date")

    def build_feature_vector(
        self,
        artist_name: str,
        medium_category: str = "oil_on_canvas",
        height_cm: float = 60.0,
        width_cm: float = 45.0,
        year_created: int = 1970,
        is_signed: bool = True,
        is_dated: bool = False,
        provenance_count: int = 1,
        literature_count: int = 0,
        exhibition_count: int = 0,
        estimate_low: float = 10000,
        estimate_high: float = 20000,
        is_live_auction: bool = True,
        auction_year: int = 2025,
        auction_month: int = 3,
    ) -> pd.DataFrame:
        """Build a single-row feature vector for prediction."""
        self.load()
        feature_cols = self.manifest["feature_cols"]

        # Start with NaN for everything
        row = {col: np.nan for col in feature_cols}

        # Get latest artist rolling features from master
        if self.master_df is not None:
            artist_mask = self.master_df["artist_name_clean"] == artist_name.strip().upper()
            artist_lots = self.master_df[artist_mask].sort_values("auction_date")
            if not artist_lots.empty:
                last = artist_lots.iloc[-1]
                rolling_cols = [c for c in feature_cols if c.startswith("artist_")]
                for col in rolling_cols:
                    if col in last.index:
                        row[col] = last[col]

        # Lot-level features
        row["height_cm"] = height_cm
        row["width_cm"] = width_cm
        row["surface_area_cm2"] = height_cm * width_cm
        row["year_created"] = year_created
        row["artwork_age"] = auction_year - year_created
        row["is_signed"] = int(is_signed)
        row["is_dated"] = int(is_dated)
        row["provenance_count"] = provenance_count
        row["literature_count"] = literature_count
        row["exhibition_count"] = exhibition_count
        row["has_provenance"] = int(provenance_count > 0)
        row["has_literature"] = int(literature_count > 0)
        row["has_exhibitions"] = int(exhibition_count > 0)
        row["estimate_midpoint"] = (estimate_low + estimate_high) / 2
        row["estimate_spread"] = (estimate_high - estimate_low) / max(estimate_low, 1)
        row["log_estimate_mid"] = np.log1p((estimate_low + estimate_high) / 2)
        row["is_live_auction"] = int(is_live_auction)
        row["auction_year"] = auction_year
        row["is_spring_sale"] = int(auction_month in [3, 4, 5])
        row["is_fall_sale"] = int(auction_month in [9, 10, 11])

        # Categoricals
        row["medium_category"] = medium_category
        row["artist_name_clean"] = artist_name.strip().upper()

        # Market index — use latest available
        if self.master_df is not None and "market_index" in self.master_df.columns:
            valid = self.master_df["market_index"].dropna()
            if not valid.empty:
                row["market_index"] = valid.iloc[-1]

        # Artist birth/death
        if self.master_df is not None:
            artist_mask = self.master_df["artist_name_clean"] == artist_name.strip().upper()
            artist_lots = self.master_df[artist_mask]
            if not artist_lots.empty:
                birth = artist_lots["artist_birth_year"].dropna()
                if not birth.empty:
                    row["artist_age_at_creation"] = year_created - birth.iloc[0]
                death = artist_lots["artist_death_year"].dropna()
                row["artist_deceased"] = int(not death.empty)

        return pd.DataFrame([row])[feature_cols]

    def predict(self, features: pd.DataFrame) -> dict:
        """Run ensemble prediction. Returns dict with price estimates."""
        self.load()
        cat_cols = self.manifest.get("cat_cols", [])
        cat_indices = self.manifest.get("cat_indices", [])

        # CatBoost prediction
        X_cb = features.copy()
        for col in cat_cols:
            if col in X_cb.columns:
                X_cb[col] = X_cb[col].fillna("unknown").astype(str)
        pool = Pool(X_cb, cat_features=cat_indices)
        preds_cb = [m.predict(pool)[0] for m in self.cb_models]

        # XGBoost prediction
        X_xgb = features.copy()
        for col in cat_cols:
            if col in X_xgb.columns:
                enc = self._label_encoders.get(col, {})
                val = str(X_xgb[col].iloc[0])
                X_xgb[col] = enc.get(val, -1)
        preds_xgb = [m.predict(X_xgb)[0] for m in self.xgb_models]

        all_preds = preds_cb + preds_xgb
        mean_log = np.mean(all_preds)
        std_log = np.std(all_preds)

        predicted_price = np.exp(mean_log)
        low_ci = np.exp(mean_log - 1.96 * std_log)
        high_ci = np.exp(mean_log + 1.96 * std_log)

        return {
            "predicted_price": round(predicted_price, 2),
            "low_ci": round(low_ci, 2),
            "high_ci": round(high_ci, 2),
            "log_prediction": round(mean_log, 4),
            "ensemble_std": round(std_log, 4),
            "individual_preds": [round(np.exp(p), 2) for p in all_preds],
        }
