"""
Indian Art Market ML Price Predictor -- Main Streamlit Dashboard.
Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.styles import inject_css, page_header, section_header, kpi_card, sidebar_brand
from dashboard.visualizations import (
    plot_yearly_market,
    plot_source_comparison,
    plot_source_donut,
    plot_medium_breakdown,
    plot_artist_yoy,
    plot_artist_medium_mix,
    plot_artist_size_analysis,
    plot_provenance_impact,
    plot_index_comparison,
    plot_pred_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_error_by_price_range,
    plot_price_gauge,
)
from dashboard.index_builder import (
    build_all_indices,
    compute_cagr,
    CORE_4,
    STAR_ARTISTS,
    match_artists,
)
from dashboard.predictor import ArtPredictor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Art Market Predictor",
    page_icon="\U0001f3a8",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_currency(value: float, decimals: int = 0) -> str:
    """Format a USD value with appropriate suffix (K / M)."""
    if pd.isna(value) or value == 0:
        return "$0"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.{decimals}f}K"
    return f"${value:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def fmt_number(value, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading master dataset...")
def load_master() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "master.csv"
    df = pd.read_csv(path)
    df["auction_date"] = (
        pd.to_datetime(df["auction_date"], errors="coerce", format="mixed", utc=True)
        .dt.tz_localize(None)
    )
    return df


@st.cache_data(show_spinner="Loading ML-ready dataset...")
def load_ml_ready() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "ml_ready.csv"
    df = pd.read_csv(path)
    df["auction_date"] = (
        pd.to_datetime(df["auction_date"], errors="coerce", format="mixed", utc=True)
        .dt.tz_localize(None)
    )
    return df


@st.cache_data(show_spinner="Loading model manifest...")
def load_manifest() -> dict:
    path = PROJECT_ROOT / "models" / "saved" / "model_manifest.json"
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading prediction models...")
def load_predictor() -> ArtPredictor:
    predictor = ArtPredictor()
    predictor.load()
    return predictor


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Market Overview",
    "Price Predictor",
    "Artist Deep Dive",
    "Index Performance",
    "Model Performance",
    "Backtest",
]

with st.sidebar:
    sidebar_brand()
    st.markdown("---")
    page = st.selectbox("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Indian Art Market ML Predictor")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    master_df = load_master()
    ml_df = load_ml_ready()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

try:
    manifest = load_manifest()
    predictor = load_predictor()
except Exception as e:
    manifest = None
    predictor = None
    if page in ("Price Predictor", "Model Performance"):
        st.sidebar.warning(f"Models unavailable: {e}")


# ===========================================================================
# PAGE 1 -- Market Overview
# ===========================================================================

def page_market_overview():
    page_header("Market Overview", "Aggregate view of the Indian art auction market")

    sold_df = master_df[master_df["is_sold"] == True]

    # ---- KPI row ----
    total_lots = len(master_df)
    total_sold = len(sold_df)
    total_value = sold_df["hammer_price_usd"].sum()
    median_price = sold_df["hammer_price_usd"].median()
    unique_artists = master_df["artist_name"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Total Lots", fmt_number(total_lots))
    with k2:
        kpi_card("Sold", fmt_number(total_sold))
    with k3:
        kpi_card("Total Value", fmt_currency(total_value, 1))
    with k4:
        kpi_card("Median Price", fmt_currency(median_price))
    with k5:
        kpi_card("Unique Artists", fmt_number(unique_artists))

    st.markdown("")

    # ---- Yearly volume / value ----
    section_header("Yearly Volume & Value")
    fig_yearly = plot_yearly_market(master_df)
    st.plotly_chart(fig_yearly, use_container_width=True)

    # ---- Source comparison ----
    section_header("Auction House Comparison")
    col_bar, col_donut = st.columns([3, 2])
    with col_bar:
        fig_src = plot_source_comparison(master_df)
        st.plotly_chart(fig_src, use_container_width=True)
    with col_donut:
        fig_donut = plot_source_donut(master_df)
        st.plotly_chart(fig_donut, use_container_width=True)

    # ---- Medium breakdown ----
    section_header("Medium Breakdown")
    fig_med = plot_medium_breakdown(master_df)
    st.plotly_chart(fig_med, use_container_width=True)

    # ---- Top 20 sales ----
    section_header("Top 20 Sales")
    top20_cols = [
        "artist_name", "title", "auction_date", "hammer_price_usd",
        "estimate_low_usd", "estimate_high_usd", "medium_category", "source",
    ]
    available = [c for c in top20_cols if c in sold_df.columns]
    top20 = sold_df.nlargest(20, "hammer_price_usd")[available].reset_index(drop=True)
    top20.index = top20.index + 1

    rename = {
        "artist_name": "Artist",
        "title": "Title",
        "auction_date": "Date",
        "hammer_price_usd": "Hammer Price (USD)",
        "estimate_low_usd": "Est. Low",
        "estimate_high_usd": "Est. High",
        "medium_category": "Medium",
        "source": "Source",
    }
    top20 = top20.rename(columns=rename)

    fmt = {}
    for c in ["Hammer Price (USD)", "Est. Low", "Est. High"]:
        if c in top20.columns:
            fmt[c] = "${:,.0f}"
    if "Date" in top20.columns:
        fmt["Date"] = lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""

    st.dataframe(
        top20.style.format(fmt, na_rep="--"),
        use_container_width=True,
        height=600,
    )


# ===========================================================================
# PAGE 2 -- Price Predictor
# ===========================================================================

def page_price_predictor():
    page_header("Price Predictor", "ML-powered auction price estimation")

    if predictor is None:
        st.error("Prediction models are not available. Please ensure models are trained.")
        return

    artists = sorted(master_df["artist_name"].dropna().unique().tolist())
    mediums = sorted(master_df["medium_category"].dropna().unique().tolist())

    # ---- Input form ----
    col_left, col_right = st.columns(2)

    with col_left:
        section_header("Artwork Details")
        artist = st.selectbox("Artist", artists, key="pred_artist")
        medium = st.selectbox("Medium", mediums, key="pred_medium")

        dim1, dim2 = st.columns(2)
        with dim1:
            height = st.number_input(
                "Height (cm)", min_value=1.0, max_value=500.0,
                value=60.0, step=1.0, key="pred_h",
            )
        with dim2:
            width = st.number_input(
                "Width (cm)", min_value=1.0, max_value=500.0,
                value=45.0, step=1.0, key="pred_w",
            )
        year_created = st.number_input(
            "Year Created", min_value=1850, max_value=2025,
            value=1970, step=1, key="pred_yr",
        )

    with col_right:
        section_header("Provenance & Estimates")
        ch1, ch2 = st.columns(2)
        with ch1:
            is_signed = st.checkbox("Signed", value=True, key="pred_signed")
        with ch2:
            is_dated = st.checkbox("Dated", value=False, key="pred_dated")

        provenance_count = st.slider("Provenance Count", 0, 20, 1, key="pred_prov")
        literature_count = st.slider("Literature Count", 0, 20, 0, key="pred_lit")
        exhibition_count = st.slider("Exhibition Count", 0, 20, 0, key="pred_exh")

        est1, est2 = st.columns(2)
        with est1:
            est_low = st.number_input(
                "Estimate Low (USD)", min_value=0,
                value=10000, step=1000, key="pred_el",
            )
        with est2:
            est_high = st.number_input(
                "Estimate High (USD)", min_value=0,
                value=20000, step=1000, key="pred_eh",
            )

    st.markdown("")

    # ---- Predict ----
    if st.button("Predict Price", type="primary", use_container_width=True):
        with st.spinner("Running ensemble prediction..."):
            features = predictor.build_feature_vector(
                artist_name=artist,
                medium_category=medium,
                height_cm=height,
                width_cm=width,
                year_created=year_created,
                is_signed=is_signed,
                is_dated=is_dated,
                provenance_count=provenance_count,
                literature_count=literature_count,
                exhibition_count=exhibition_count,
                estimate_low=float(est_low),
                estimate_high=float(est_high),
            )
            result = predictor.predict(features)

        st.markdown("")
        section_header("Prediction Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            kpi_card("Predicted Price", fmt_currency(result["predicted_price"]))
        with r2:
            kpi_card("95% CI Low", fmt_currency(result["low_ci"]))
        with r3:
            kpi_card("95% CI High", fmt_currency(result["high_ci"]))

        st.markdown("")

        # Gauge chart
        fig_gauge = plot_price_gauge(
            predicted=result["predicted_price"],
            low_ci=result["low_ci"],
            high_ci=result["high_ci"],
            est_low=float(est_low),
            est_high=float(est_high),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Compare to artist history
        stats = predictor.get_artist_stats(artist)
        if stats.get("avg_price"):
            vs_avg = (result["predicted_price"] - stats["avg_price"]) / stats["avg_price"] * 100
            st.info(
                f"This prediction is **{vs_avg:+.1f}%** vs "
                f"{artist}'s historical average of {fmt_currency(stats['avg_price'])}"
            )

        # Detailed breakdown
        with st.expander("Prediction Details"):
            st.markdown(f"**Log-space prediction:** {result['log_prediction']}")
            st.markdown(f"**Ensemble std:** {result['ensemble_std']}")
            st.markdown("**Individual model predictions:**")
            pred_df = pd.DataFrame({
                "Model": [f"Model {i+1}" for i in range(len(result["individual_preds"]))],
                "Predicted Price (USD)": result["individual_preds"],
            })
            st.dataframe(
                pred_df.style.format({"Predicted Price (USD)": "${:,.0f}"}),
                use_container_width=True,
            )


# ===========================================================================
# PAGE 3 -- Artist Deep Dive
# ===========================================================================

def page_artist_deep_dive():
    page_header("Artist Deep Dive", "Individual artist analysis and lot history")

    artists = sorted(master_df["artist_name"].dropna().unique().tolist())
    artist = st.selectbox("Select Artist", artists, key="dd_artist")

    if not artist:
        st.info("Please select an artist to continue.")
        return

    # Filter data
    artist_df = master_df[master_df["artist_name"] == artist].copy()
    artist_sold = artist_df[artist_df["is_sold"] == True]

    if artist_df.empty:
        st.warning(f"No data found for {artist}.")
        return

    # ---- KPI row ----
    total_lots = len(artist_df)
    total_sold = len(artist_sold)
    str_pct = (total_sold / total_lots * 100) if total_lots > 0 else 0
    median_p = artist_sold["hammer_price_usd"].median() if len(artist_sold) > 0 else 0
    mean_p = artist_sold["hammer_price_usd"].mean() if len(artist_sold) > 0 else 0
    max_p = artist_sold["hammer_price_usd"].max() if len(artist_sold) > 0 else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("Total Lots", fmt_number(total_lots))
    with k2:
        kpi_card("Sold", fmt_number(total_sold))
    with k3:
        kpi_card("STR", fmt_pct(str_pct))
    with k4:
        kpi_card("Median Price", fmt_currency(median_p))
    with k5:
        kpi_card("Mean Price", fmt_currency(mean_p))
    with k6:
        kpi_card("Max Price", fmt_currency(max_p))

    st.markdown("")

    # ---- YoY price trends ----
    section_header("Year-over-Year Price Trends")
    fig_yoy = plot_artist_yoy(artist_df, artist)
    st.plotly_chart(fig_yoy, use_container_width=True)

    # ---- Three analysis columns ----
    col_med, col_size, col_prov = st.columns(3)

    with col_med:
        section_header("By Medium")
        if not artist_sold.empty:
            fig_med_mix = plot_artist_medium_mix(artist_sold)
            st.plotly_chart(fig_med_mix, use_container_width=True)
        else:
            st.info("No sold lots to analyze.")

    with col_size:
        section_header("By Size")
        if not artist_sold.empty:
            fig_size = plot_artist_size_analysis(artist_sold)
            st.plotly_chart(fig_size, use_container_width=True)
        else:
            st.info("No sold lots to analyze.")

    with col_prov:
        section_header("Provenance & Literature")
        if not artist_sold.empty:
            fig_prov = plot_provenance_impact(artist_sold)
            st.plotly_chart(fig_prov, use_container_width=True)
        else:
            st.info("No sold lots to analyze.")

    # ---- Full lot history ----
    section_header("Lot History")
    display_cols = [
        "auction_date", "title", "medium_category", "height_cm", "width_cm",
        "hammer_price_usd", "estimate_low_usd", "estimate_high_usd",
        "is_sold", "source",
    ]
    available = [c for c in display_cols if c in artist_df.columns]
    history = (
        artist_df[available]
        .sort_values("auction_date", ascending=False)
        .reset_index(drop=True)
    )
    history.index = history.index + 1

    rename_map = {
        "auction_date": "Date",
        "title": "Title",
        "medium_category": "Medium",
        "height_cm": "H (cm)",
        "width_cm": "W (cm)",
        "hammer_price_usd": "Hammer (USD)",
        "estimate_low_usd": "Est. Low",
        "estimate_high_usd": "Est. High",
        "is_sold": "Sold",
        "source": "Source",
    }
    history = history.rename(columns=rename_map)

    fmt = {}
    for c in ["Hammer (USD)", "Est. Low", "Est. High"]:
        if c in history.columns:
            fmt[c] = "${:,.0f}"
    if "Date" in history.columns:
        fmt["Date"] = lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""

    st.dataframe(
        history.style.format(fmt, na_rep="--"),
        use_container_width=True,
        height=500,
    )


# ===========================================================================
# PAGE 4 -- Index Performance
# ===========================================================================

def page_index_performance():
    page_header("Index Performance", "Artist and market price indices over time")

    sold_df = master_df[master_df["is_sold"] == True].copy()

    try:
        indices = build_all_indices(sold_df)
    except Exception as e:
        st.error(f"Could not build indices: {e}")
        return

    # Extract CAGRs (already computed inside build_all_indices)
    cagr_core4 = indices.get("core4", {}).get("cagr_top_tier", 0)
    cagr_star = indices.get("full_star", {}).get("cagr_top_tier", 0)
    cagr_market = indices.get("full_market", {}).get("cagr_top_tier", 0)

    # ---- KPI row ----
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card(
            "Core 4 CAGR",
            fmt_pct(cagr_core4 * 100, 1) if cagr_core4 else "N/A",
        )
    with k2:
        kpi_card(
            "Star Artists CAGR",
            fmt_pct(cagr_star * 100, 1) if cagr_star else "N/A",
        )
    with k3:
        kpi_card(
            "Market CAGR",
            fmt_pct(cagr_market * 100, 1) if cagr_market else "N/A",
        )

    st.markdown("")

    # ---- Index comparison chart ----
    section_header("Index Comparison")
    # Reshape indices dict into format expected by plot_index_comparison
    chart_indices = {}
    for seg_name, seg_data in indices.items():
        for metric in ["median", "top_tier", "hedonic"]:
            if metric in seg_data:
                series = seg_data[metric]
                label = f"{seg_name} ({metric})"
                chart_indices[label] = {
                    "years": series.index.tolist(),
                    "values": series.values.tolist(),
                }
    fig_idx = plot_index_comparison(chart_indices)
    st.plotly_chart(fig_idx, use_container_width=True)

    # ---- CAGR table ----
    with st.expander("CAGR Details"):
        cagr_rows = []
        for seg_name, seg_data in indices.items():
            row = {"Index": seg_name}
            for metric in ["cagr_median", "cagr_top_tier", "cagr_hedonic"]:
                val = seg_data.get(metric, None)
                row[metric.replace("cagr_", "").title() + " CAGR (%)"] = round(val * 100, 2) if val else None
            cagr_rows.append(row)
        cagr_table = pd.DataFrame(cagr_rows)
        st.dataframe(cagr_table, use_container_width=True, hide_index=True)

    # ---- Core 4 individual artist mini-charts (2x2 grid) ----
    section_header("Core 4 Artists")
    core4_names = ["F.N. Souza", "S.H. Raza", "Tyeb Mehta", "M.F. Husain"]
    core4_variants = {
        "F.N. Souza": ["FRANCIS NEWTON SOUZA", "F N SOUZA"],
        "S.H. Raza": ["SAYED HAIDER RAZA", "S H RAZA"],
        "Tyeb Mehta": ["TYEB MEHTA"],
        "M.F. Husain": ["MAQBOOL FIDA HUSAIN", "M F HUSAIN"],
    }

    for row_start in [0, 2]:
        row = core4_names[row_start:row_start + 2]
        cols = st.columns(len(row))
        for i, display_name in enumerate(row):
            with cols[i]:
                st.markdown(f"**{display_name}**")
                variants = core4_variants.get(display_name, [display_name.upper()])
                art_mask = sold_df["artist_name_clean"].isin(variants) if "artist_name_clean" in sold_df.columns else sold_df["artist_name"].str.upper().isin(variants)
                art_data = sold_df[art_mask]
                if not art_data.empty:
                    fig_mini = plot_artist_yoy(art_data, display_name)
                    st.plotly_chart(fig_mini, use_container_width=True, key=f"core4_{display_name}")
                else:
                    st.info("No sold lots found.")


# ===========================================================================
# PAGE 5 -- Model Performance
# ===========================================================================

def page_model_performance():
    page_header("Model Performance", "Evaluation metrics and diagnostics")

    if manifest is None:
        st.error("Model manifest not available. Please train models first.")
        return

    metrics = manifest.get("metrics", {})

    # ---- KPI row ----
    r2 = metrics.get("r2_log", 0)
    mape = metrics.get("mape_pct", 0)
    mae = metrics.get("mae_usd", 0)
    median_ae = metrics.get("median_ae_usd", 0)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("R-squared (log)", fmt_number(r2, 4))
    with k2:
        kpi_card("MAPE", fmt_pct(mape))
    with k3:
        kpi_card("MAE (USD)", fmt_currency(mae))
    with k4:
        kpi_card("Median AE (USD)", fmt_currency(median_ae))

    st.markdown("")

    # ---- Accuracy bands & charts (require predictor) ----
    if predictor is None or ml_df.empty:
        st.info("Load prediction models to see detailed performance charts.")
        _show_model_architecture(metrics)
        return

    with st.spinner("Computing accuracy metrics on test set..."):
        try:
            test_size = metrics.get("test_size", 527)

            # Sort by date; last N rows form the test set
            ml_sorted = ml_df.sort_values("auction_date").reset_index(drop=True)
            test_df = ml_sorted.tail(test_size).copy()

            feature_cols = manifest["feature_cols"]
            cat_cols = manifest.get("cat_cols", [])
            cat_indices = manifest.get("cat_indices", [])

            X_test = test_df[feature_cols].copy()
            y_actual = test_df["hammer_price_usd"].values
            y_log_actual = test_df["log_hammer_price"].values

            # --- CatBoost predictions ---
            from catboost import Pool as CbPool

            X_cb = X_test.copy()
            for col in cat_cols:
                if col in X_cb.columns:
                    X_cb[col] = X_cb[col].fillna("unknown").astype(str)
            pool = CbPool(X_cb, cat_features=cat_indices)

            # --- XGBoost predictions ---
            X_xgb = X_test.copy()
            for col in cat_cols:
                if col in X_xgb.columns:
                    enc = predictor._label_encoders.get(col, {})
                    X_xgb[col] = X_xgb[col].apply(lambda v: enc.get(str(v), -1))

            all_preds_log = []
            for m in predictor.cb_models:
                all_preds_log.append(m.predict(pool))
            for m in predictor.xgb_models:
                all_preds_log.append(m.predict(X_xgb))

            ensemble_log = np.mean(all_preds_log, axis=0)
            y_pred = np.exp(ensemble_log)

            # Accuracy bands
            pct_error = np.abs(y_pred - y_actual) / np.where(y_actual > 0, y_actual, 1)
            within_10 = np.mean(pct_error <= 0.10) * 100
            within_25 = np.mean(pct_error <= 0.25) * 100
            within_50 = np.mean(pct_error <= 0.50) * 100

        except Exception as e:
            st.warning(f"Could not compute test-set predictions: {e}")
            _show_model_architecture(metrics)
            return

    # ---- Accuracy band cards ----
    section_header("Accuracy Bands")
    a1, a2, a3 = st.columns(3)
    with a1:
        kpi_card("Within 10%", fmt_pct(within_10))
    with a2:
        kpi_card("Within 25%", fmt_pct(within_25))
    with a3:
        kpi_card("Within 50%", fmt_pct(within_50))

    st.markdown("")

    # ---- Diagnostic charts (tabs) ----
    tab_scatter, tab_resid, tab_feat, tab_err = st.tabs([
        "Predicted vs Actual",
        "Residual Distribution",
        "Feature Importance",
        "Error by Price Range",
    ])

    with tab_scatter:
        fig_pva = plot_pred_vs_actual(y_actual, y_pred)
        st.plotly_chart(fig_pva, use_container_width=True)

    with tab_resid:
        fig_res = plot_residuals(y_log_actual, ensemble_log)
        st.plotly_chart(fig_res, use_container_width=True)

    with tab_feat:
        importance = predictor.cb_models[0].get_feature_importance()
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
        fig_fi = plot_feature_importance(fi_df)
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab_err:
        fig_epr = plot_error_by_price_range(y_actual, y_pred)
        st.plotly_chart(fig_epr, use_container_width=True)

    # ---- Model architecture expander ----
    _show_model_architecture(metrics)


def _show_model_architecture(metrics: dict) -> None:
    """Render the model architecture expander."""
    with st.expander("Model Architecture"):
        st.markdown(
            f"**Ensemble:** {metrics.get('n_catboost', 0)} CatBoost + "
            f"{metrics.get('n_xgboost', 0)} XGBoost models"
        )
        st.markdown(f"**Train size:** {metrics.get('train_size', 'N/A'):,}")
        st.markdown(f"**Test size:** {metrics.get('test_size', 'N/A'):,}")
        st.markdown(f"**RMSE (log):** {metrics.get('rmse_log', 'N/A')}")
        st.markdown(f"**Target:** `{manifest.get('target', 'log_hammer_price') if manifest else 'log_hammer_price'}`")
        if manifest:
            st.markdown("**Feature columns:**")
            st.code(", ".join(manifest.get("feature_cols", [])))


# ===========================================================================
# PAGE 6 -- Backtest
# ===========================================================================

def page_backtest():
    page_header("Backtest", "Model predictions vs actual hammer prices across all auctions")

    backtest_path = PROJECT_ROOT / "data" / "processed" / "backtest_full.json"
    if not backtest_path.exists():
        st.warning("No backtest data found. Run `python scripts/run_backtest.py` first.")
        return

    import json as _json
    with open(backtest_path) as f:
        bt = _json.load(f)

    overall = bt.get("overall", {})
    test_set = bt.get("test_set", {})
    by_year = bt.get("by_year", [])
    by_source = bt.get("by_source", [])
    by_artist = bt.get("by_artist", [])

    # ---- KPI row: test set (out-of-sample) ----
    section_header("Out-of-Sample Performance (Test Set)")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kpi_card("Test Lots", f"{test_set.get('n', 0):,}"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Median Error", f"{test_set.get('median_err', 0):.1f}%", color="#ffb74d"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Within 10%", f"{test_set.get('within_10', 0)}/{test_set.get('n', 0)}"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Within 25%", f"{test_set.get('within_25', 0)}/{test_set.get('n', 0)}", color="#00bfa5"), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card("Within 50%", f"{test_set.get('within_50', 0)}/{test_set.get('n', 0)}", color="#66bb6a"), unsafe_allow_html=True)

    st.markdown("")

    # ---- Model vs Estimates comparison ----
    section_header("Model vs Auction House Estimates")
    col_m, col_vs, col_e = st.columns([2, 1, 2])
    with col_m:
        st.markdown(
            f'<div style="text-align:center;padding:1rem;">'
            f'<div style="font-size:2.5rem;font-weight:700;color:#00bfa5;">{test_set.get("within_50", 0)}/{test_set.get("n", 0)}</div>'
            f'<div style="color:#8899aa;font-size:0.85rem;margin-top:4px;">Model within 50%</div>'
            f'<div style="color:#ffb74d;font-size:1.1rem;margin-top:8px;">{test_set.get("median_err", 0):.1f}% median error</div>'
            f'</div>', unsafe_allow_html=True)
    with col_vs:
        st.markdown('<div style="text-align:center;padding:2rem;font-size:1.5rem;color:#8899aa;font-weight:700;">VS</div>', unsafe_allow_html=True)
    with col_e:
        est_w50 = test_set.get("est_within_50", 0)
        est_med = test_set.get("est_median_err", 0)
        st.markdown(
            f'<div style="text-align:center;padding:1rem;">'
            f'<div style="font-size:2.5rem;font-weight:700;color:#ffb74d;">{est_w50}/{test_set.get("n", 0)}</div>'
            f'<div style="color:#8899aa;font-size:0.85rem;margin-top:4px;">Estimates within 50%</div>'
            f'<div style="color:#ffb74d;font-size:1.1rem;margin-top:8px;">{est_med:.1f}% median error</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("")

    # ---- By Year ----
    section_header("Performance by Year")
    if by_year:
        yr_df = pd.DataFrame(by_year)
        yr_df["year"] = yr_df["year"].astype(int)
        yr_df["n"] = yr_df["n"].astype(int)
        yr_df = yr_df.sort_values("year")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yr_df["year"], y=yr_df["n"], name="# Lots", marker_color="#26c6da88"), secondary_y=False)
        fig.add_trace(go.Scatter(x=yr_df["year"], y=yr_df["median_err"], name="Model Median Error", line=dict(color="#00bfa5", width=2.5), mode="lines+markers"), secondary_y=True)
        if "est_median_err" in yr_df.columns:
            fig.add_trace(go.Scatter(x=yr_df["year"], y=yr_df["est_median_err"], name="Estimate Median Error", line=dict(color="#ffb74d", width=2, dash="dash"), mode="lines+markers"), secondary_y=True)
        fig.update_layout(plot_bgcolor="#0c0c1a", paper_bgcolor="#1e1e2e", font=dict(color="#8899aa"), height=400,
                         legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=60, r=60, t=30, b=40))
        fig.update_xaxes(gridcolor="#2a2a3e")
        fig.update_yaxes(title_text="# Lots", gridcolor="#2a2a3e", secondary_y=False)
        fig.update_yaxes(title_text="Median Error %", gridcolor="#2a2a3e", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(yr_df, use_container_width=True, hide_index=True)

    # ---- By Source ----
    section_header("Performance by Auction House")
    if by_source:
        src_df = pd.DataFrame(by_source)
        st.dataframe(src_df, use_container_width=True, hide_index=True)

    # ---- By Star Artist ----
    section_header("Star Artist Performance")
    if by_artist:
        art_df = pd.DataFrame(by_artist).sort_values("median_err")
        st.dataframe(art_df, use_container_width=True, hide_index=True)

    # ---- Full dataset summary ----
    with st.expander("Full Dataset (In-Sample + Test)"):
        o = bt.get("overall", {})
        st.markdown(f"""
        - **Total lots:** {o.get('n', 0):,}
        - **Median error:** {o.get('median_err', 0):.1f}%
        - **Within 10%:** {o.get('within_10', 0)} ({o.get('within_10', 0)/max(o.get('n', 1), 1)*100:.0f}%)
        - **Within 25%:** {o.get('within_25', 0)} ({o.get('within_25', 0)/max(o.get('n', 1), 1)*100:.0f}%)
        - **Within 50%:** {o.get('within_50', 0)} ({o.get('within_50', 0)/max(o.get('n', 1), 1)*100:.0f}%)
        - *Note: In-sample performance is optimistic — the model has seen these lots during training.*
        """)


# ===========================================================================
# Page dispatch
# ===========================================================================

if page == "Market Overview":
    page_market_overview()
elif page == "Price Predictor":
    page_price_predictor()
elif page == "Artist Deep Dive":
    page_artist_deep_dive()
elif page == "Index Performance":
    page_index_performance()
elif page == "Model Performance":
    page_model_performance()
elif page == "Backtest":
    page_backtest()
