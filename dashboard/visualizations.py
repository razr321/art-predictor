"""Plotly chart builders for the art market dashboard.

All charts use a dark theme with consistent styling.
Every public function returns a ``go.Figure``.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
TEAL = "#00bfa5"
CYAN = "#26c6da"
AMBER = "#ffb74d"
MAGENTA = "#e040fb"
RED = "#ef5350"
GREEN = "#66bb6a"
COLORS = [TEAL, CYAN, AMBER, "#7c4dff", "#ff7043", MAGENTA, GREEN]
BG_DARK = "#0c0c1a"
BG_CARD = "#1e1e2e"
GRID = "#2a2a3e"
TEXT = "#8899aa"
TEXT_LIGHT = "#e4eef5"

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

def _base_layout(fig: go.Figure, title: str, height: int = 500) -> go.Figure:
    """Apply consistent dark-theme layout to *fig* and return it."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=TEXT_LIGHT)),
        height=height,
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_CARD,
        font=dict(family="Inter, sans-serif", color=TEXT),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_LIGHT, size=11),
        ),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Yearly market overview (dual-axis)
# ---------------------------------------------------------------------------

def plot_yearly_market(df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: lots sold per year (bar) + total hammer value (line).

    Expected columns: ``auction_year``, ``hammer_price_usd``, ``is_sold``.
    """
    sold = df[df["is_sold"] == True].copy()
    yearly = sold.groupby("auction_year").agg(
        lots_sold=("hammer_price_usd", "count"),
        total_value=("hammer_price_usd", "sum"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=yearly["auction_year"],
            y=yearly["lots_sold"],
            name="Lots Sold",
            marker_color=TEAL,
            opacity=0.85,
            hovertemplate="Year %{x}<br>Lots Sold: %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=yearly["auction_year"],
            y=yearly["total_value"],
            name="Total Hammer Value",
            mode="lines+markers",
            line=dict(color=AMBER, width=3),
            marker=dict(size=7),
            hovertemplate="Year %{x}<br>Total: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Lots Sold", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Total Hammer Value (USD)", secondary_y=True, gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID, dtick=1)

    _base_layout(fig, "Yearly Market Overview", height=480)
    return fig


# ---------------------------------------------------------------------------
# 2. Source comparison (grouped bar)
# ---------------------------------------------------------------------------

def plot_source_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of average hammer price by source.

    Expected columns: ``source``, ``hammer_price_usd``.
    """
    stats = (
        df.groupby("source")["hammer_price_usd"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "avg", "median": "med"})
        .sort_values("avg", ascending=False)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats["source"],
        y=stats["avg"],
        name="Mean",
        marker_color=TEAL,
        hovertemplate="%{x}<br>Mean: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=stats["source"],
        y=stats["med"],
        name="Median",
        marker_color=CYAN,
        hovertemplate="%{x}<br>Median: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(barmode="group")
    _base_layout(fig, "Average Hammer Price by Source", height=440)
    return fig


# ---------------------------------------------------------------------------
# 3. Source donut
# ---------------------------------------------------------------------------

def plot_source_donut(df: pd.DataFrame) -> go.Figure:
    """Donut pie chart of lot count by source.

    Expected column: ``source``.
    """
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["source", "lots"]

    fig = go.Figure(go.Pie(
        labels=counts["source"],
        values=counts["lots"],
        hole=0.5,
        marker=dict(colors=COLORS[: len(counts)]),
        textinfo="label+percent",
        hovertemplate="%{label}<br>Lots: %{value:,}<br>%{percent}<extra></extra>",
    ))
    _base_layout(fig, "Lot Distribution by Source", height=420)
    return fig


# ---------------------------------------------------------------------------
# 4. Medium breakdown (horizontal bar, top 15)
# ---------------------------------------------------------------------------

def plot_medium_breakdown(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar of average price by ``medium_category`` (top 15, sold only).

    Expected columns: ``medium_category``, ``hammer_price_usd``, ``is_sold``.
    """
    sold = df[(df["is_sold"] == True) & df["hammer_price_usd"].notna()]
    if sold.empty:
        fig = go.Figure()
        _base_layout(fig, "Average Price by Medium", height=480)
        return fig

    stats = (
        sold.groupby("medium_category")["hammer_price_usd"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
        .tail(15)
        .reset_index()
    )

    fig = go.Figure(go.Bar(
        y=stats["medium_category"],
        x=stats["mean"],
        orientation="h",
        marker_color=CYAN,
        text=[f"${v:,.0f}  (n={c})" for v, c in zip(stats["mean"], stats["count"])],
        textposition="outside",
        hovertemplate="%{y}<br>Avg: $%{x:,.0f}<extra></extra>",
    ))

    _base_layout(fig, "Average Hammer Price by Medium (Top 15)", height=max(420, len(stats) * 32))
    fig.update_layout(xaxis_title="Avg Hammer Price (USD)")
    return fig


# ---------------------------------------------------------------------------
# 5. Seasonal bar
# ---------------------------------------------------------------------------

def plot_seasonal(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average hammer price by auction month (1-12).

    Expected columns: ``auction_month`` (or derived from ``auction_date``),
    ``hammer_price_usd``.
    """
    col = "auction_month"
    if col not in df.columns and "auction_date" in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df["auction_date"]).dt.month

    monthly = (
        df.dropna(subset=["hammer_price_usd"])
        .groupby(col)["hammer_price_usd"]
        .agg(["mean", "count"])
        .reindex(range(1, 13))
        .fillna(0)
    )

    labels = [MONTH_LABELS.get(m, str(m)) for m in monthly.index]

    fig = go.Figure(go.Bar(
        x=labels,
        y=monthly["mean"],
        marker_color=AMBER,
        text=[f"${v:,.0f}" for v in monthly["mean"]],
        textposition="outside",
        hovertemplate="%{x}<br>Avg: $%{y:,.0f}<br>Lots: %{customdata:,}<extra></extra>",
        customdata=monthly["count"].astype(int),
    ))

    _base_layout(fig, "Average Hammer Price by Month", height=440)
    fig.update_layout(xaxis_title="Month", yaxis_title="Avg Hammer Price (USD)")
    return fig


# ---------------------------------------------------------------------------
# 6. Price prediction gauge
# ---------------------------------------------------------------------------

def plot_price_gauge(
    predicted: float,
    low_ci: float,
    high_ci: float,
    est_low: float = None,
    est_high: float = None,
) -> go.Figure:
    """Plotly indicator gauge for a single predicted value with CI range."""
    max_val = max(high_ci, est_high or 0) * 1.3

    steps = [{"range": [low_ci, high_ci], "color": "rgba(0,191,165,0.2)"}]
    if est_low is not None and est_high is not None:
        steps.append({"range": [est_low, est_high], "color": "rgba(255,183,77,0.15)"})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted,
        number={"prefix": "$", "valueformat": ",.0f", "font": {"size": 38, "color": TEAL}},
        title={"text": "Predicted Hammer Price", "font": {"color": TEXT_LIGHT, "size": 16}},
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickprefix": "$",
                "tickformat": ",.0f",
                "tickcolor": TEXT,
            },
            "bar": {"color": TEAL, "thickness": 0.25},
            "bgcolor": BG_CARD,
            "bordercolor": GRID,
            "steps": steps,
            "threshold": {
                "line": {"color": TEXT_LIGHT, "width": 2},
                "value": predicted,
                "thickness": 0.8,
            },
        },
    ))

    if est_low is not None and est_high is not None:
        fig.add_annotation(
            text=f"Auction Estimate: ${est_low:,.0f} \u2013 ${est_high:,.0f}",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=13, color=AMBER),
        )

    fig.add_annotation(
        text=f"90 % CI: ${low_ci:,.0f} \u2013 ${high_ci:,.0f}",
        xref="paper", yref="paper", x=0.5, y=-0.02,
        showarrow=False, font=dict(size=12, color=TEXT),
    )

    _base_layout(fig, "", height=370)
    return fig


# ---------------------------------------------------------------------------
# 7. Artist year-over-year performance
# ---------------------------------------------------------------------------

def plot_artist_yoy(df: pd.DataFrame, artist_name: str) -> go.Figure:
    """Multi-line chart for one artist: high / low / median / mean price per year,
    with a bar overlay for lot count.

    *df* should already be filtered to the artist and contain
    ``auction_year`` and ``hammer_price_usd``.
    """
    sold = df[df["hammer_price_usd"].notna()].copy()
    if sold.empty:
        fig = go.Figure()
        _base_layout(fig, f"{artist_name} \u2014 Year-over-Year", height=500)
        return fig

    yearly = sold.groupby("auction_year")["hammer_price_usd"].agg(
        ["max", "min", "median", "mean", "count"]
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Lot-count bars (background)
    fig.add_trace(
        go.Bar(
            x=yearly["auction_year"], y=yearly["count"],
            name="Lot Count", marker_color="rgba(0,191,165,0.2)",
            hovertemplate="Year %{x}<br>Lots: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Price lines
    for col, label, color, dash in [
        ("max", "High", RED, None),
        ("min", "Low", CYAN, None),
        ("median", "Median", AMBER, "dash"),
        ("mean", "Mean", MAGENTA, "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=yearly["auction_year"], y=yearly[col],
                name=label, mode="lines+markers",
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=6),
                hovertemplate=f"{label}: $%{{y:,.0f}}<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.update_yaxes(title_text="Hammer Price (USD)", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Lot Count", secondary_y=True, gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID, dtick=1)

    _base_layout(fig, f"{artist_name} \u2014 Year-over-Year Performance", height=520)
    return fig


# ---------------------------------------------------------------------------
# 8. Artist medium mix
# ---------------------------------------------------------------------------

def plot_artist_medium_mix(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar of average price by medium for one artist.

    Expected columns: ``medium_category``, ``hammer_price_usd``.
    """
    data = df.dropna(subset=["hammer_price_usd"])
    if data.empty:
        fig = go.Figure()
        _base_layout(fig, "Average Price by Medium", height=400)
        return fig

    stats = (
        data.groupby("medium_category")["hammer_price_usd"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
        .reset_index()
    )

    fig = go.Figure(go.Bar(
        y=stats["medium_category"],
        x=stats["mean"],
        orientation="h",
        marker_color=COLORS[: len(stats)] if len(stats) <= len(COLORS) else TEAL,
        text=[f"${v:,.0f}  (n={c})" for v, c in zip(stats["mean"], stats["count"])],
        textposition="outside",
        hovertemplate="%{y}<br>Avg: $%{x:,.0f}<extra></extra>",
    ))

    _base_layout(fig, "Average Price by Medium", height=max(350, len(stats) * 35))
    fig.update_layout(xaxis_title="Avg Hammer Price (USD)")
    return fig


# ---------------------------------------------------------------------------
# 9. Artist size analysis (scatter)
# ---------------------------------------------------------------------------

def plot_artist_size_analysis(df: pd.DataFrame) -> go.Figure:
    """Scatter of area (height_cm * width_cm) vs hammer_price_usd.

    Points colored by ``medium_category``. Y-axis on log scale.
    """
    data = df.dropna(subset=["height_cm", "width_cm", "hammer_price_usd"]).copy()
    data["area_cm2"] = data["height_cm"] * data["width_cm"]
    data = data[data["area_cm2"] > 0]

    if data.empty:
        fig = go.Figure()
        _base_layout(fig, "Size vs Price", height=480)
        return fig

    mediums = data["medium_category"].unique()
    color_map = {m: COLORS[i % len(COLORS)] for i, m in enumerate(mediums)}

    fig = go.Figure()
    for medium in mediums:
        subset = data[data["medium_category"] == medium]
        fig.add_trace(go.Scatter(
            x=subset["area_cm2"],
            y=subset["hammer_price_usd"],
            mode="markers",
            name=medium,
            marker=dict(
                color=color_map[medium],
                size=8,
                opacity=0.7,
                line=dict(width=0.5, color=BG_DARK),
            ),
            hovertemplate=(
                f"<b>{medium}</b><br>"
                "Area: %{x:,.0f} cm\u00b2<br>"
                "Price: $%{y:,.0f}<extra></extra>"
            ),
        ))

    _base_layout(fig, "Size vs Hammer Price", height=500)
    fig.update_layout(
        xaxis_title="Area (cm\u00b2)",
        yaxis_title="Hammer Price (USD)",
        yaxis_type="log",
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Provenance / literature impact (box plots)
# ---------------------------------------------------------------------------

def plot_provenance_impact(df: pd.DataFrame) -> go.Figure:
    """Side-by-side box plots: price with/without provenance, with/without literature.

    Expected columns: ``provenance_count``, ``literature_count``, ``hammer_price_usd``.
    """
    data = df.dropna(subset=["hammer_price_usd"]).copy()
    data["has_provenance"] = (data["provenance_count"] > 0).map({True: "With Provenance", False: "Without"})
    data["has_literature"] = (data["literature_count"] > 0).map({True: "With Literature", False: "Without"})

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Provenance", "Literature"])

    for val, color in [("With Provenance", TEAL), ("Without", RED)]:
        subset = data[data["has_provenance"] == val]
        fig.add_trace(
            go.Box(
                y=subset["hammer_price_usd"],
                name=val,
                marker_color=color,
                boxmean="sd",
                hovertemplate="$%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )

    for val, color in [("With Literature", CYAN), ("Without", AMBER)]:
        subset = data[data["has_literature"] == val]
        fig.add_trace(
            go.Box(
                y=subset["hammer_price_usd"],
                name=val,
                marker_color=color,
                boxmean="sd",
                hovertemplate="$%{y:,.0f}<extra></extra>",
            ),
            row=1, col=2,
        )

    _base_layout(fig, "Impact of Provenance & Literature on Price", height=500)
    fig.update_yaxes(title_text="Hammer Price (USD)", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# 11. Index comparison (multi-line, rebased)
# ---------------------------------------------------------------------------

def plot_index_comparison(indices_dict: dict) -> go.Figure:
    """Multi-line chart of rebased price indices.

    *indices_dict* maps ``label -> {"years": [...], "values": [...]}``.
    """
    fig = go.Figure()

    for i, (label, series) in enumerate(indices_dict.items()):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=series["years"],
            y=series["values"],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Index: %{{y:.1f}}<extra></extra>",
        ))

    _base_layout(fig, "Price Index Comparison (Rebased)", height=480)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Index Value",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 12. Predicted vs Actual (log-log scatter)
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(y_actual, y_pred) -> go.Figure:
    """Log-log scatter of predicted vs actual prices with identity line.

    Points colored by absolute error magnitude.
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_actual)

    fig = go.Figure()

    # Identity line
    lo = max(min(y_actual.min(), y_pred.min()), 1)
    hi = max(y_actual.max(), y_pred.max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", name="Perfect Prediction",
        line=dict(color=TEXT, dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(
            color=abs_err,
            colorscale=[[0, TEAL], [0.5, AMBER], [1, RED]],
            size=5,
            opacity=0.6,
            colorbar=dict(title="Abs Error", tickprefix="$", tickformat=",.0f"),
            line=dict(width=0),
        ),
        hovertemplate=(
            "Actual: $%{x:,.0f}<br>"
            "Predicted: $%{y:,.0f}<br>"
            "Error: $%{marker.color:,.0f}<extra></extra>"
        ),
    ))

    _base_layout(fig, "Predicted vs Actual Prices", height=520)
    fig.update_layout(
        xaxis_title="Actual Price (USD)",
        yaxis_title="Predicted Price (USD)",
        xaxis_type="log",
        yaxis_type="log",
    )
    return fig


# ---------------------------------------------------------------------------
# 13. Residuals histogram
# ---------------------------------------------------------------------------

def plot_residuals(y_actual, y_pred) -> go.Figure:
    """Histogram of percentage errors ``(pred - actual) / actual * 100``.

    Vertical reference line at 0.
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_actual != 0
    pct_err = (y_pred[mask] - y_actual[mask]) / y_actual[mask] * 100

    fig = go.Figure(go.Histogram(
        x=pct_err,
        nbinsx=60,
        marker_color=CYAN,
        opacity=0.85,
        hovertemplate="Bin: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color=RED, width=2, dash="dash"))

    median_err = float(np.median(pct_err))
    fig.add_annotation(
        text=f"Median: {median_err:+.1f}%",
        x=median_err, y=1, yref="paper",
        showarrow=True, arrowhead=2, arrowcolor=AMBER,
        font=dict(color=AMBER, size=12),
    )

    _base_layout(fig, "Distribution of Prediction Errors", height=440)
    fig.update_layout(
        xaxis_title="Percentage Error (%)",
        yaxis_title="Count",
    )
    return fig


# ---------------------------------------------------------------------------
# 14. Feature importance (horizontal bar, top 25)
# ---------------------------------------------------------------------------

def plot_feature_importance(fi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar of top 25 features by importance.

    *fi_df* must have ``feature`` and ``importance`` columns.
    """
    top = fi_df.nlargest(25, "importance").sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        y=top["feature"],
        x=top["importance"],
        orientation="h",
        marker=dict(
            color=top["importance"],
            colorscale=[[0, CYAN], [1, TEAL]],
        ),
        text=[f"{v:.4f}" for v in top["importance"]],
        textposition="outside",
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))

    _base_layout(fig, "Top 25 Feature Importances", height=max(480, len(top) * 24))
    fig.update_layout(xaxis_title="Importance")
    return fig


# ---------------------------------------------------------------------------
# 15. Error by price range
# ---------------------------------------------------------------------------

def plot_error_by_price_range(y_actual, y_pred) -> go.Figure:
    """Bar chart of MAPE by price bucket."""
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    buckets = [
        ("<$10K", 0, 10_000),
        ("$10\u201350K", 10_000, 50_000),
        ("$50\u2013200K", 50_000, 200_000),
        ("$200K\u20131M", 200_000, 1_000_000),
        (">$1M", 1_000_000, np.inf),
    ]

    labels, mapes, counts = [], [], []
    for label, lo, hi in buckets:
        mask = (y_actual >= lo) & (y_actual < hi) & (y_actual != 0)
        if mask.sum() == 0:
            continue
        ape = np.abs((y_pred[mask] - y_actual[mask]) / y_actual[mask]) * 100
        labels.append(label)
        mapes.append(float(np.mean(ape)))
        counts.append(int(mask.sum()))

    fig = go.Figure(go.Bar(
        x=labels,
        y=mapes,
        marker_color=COLORS[: len(labels)],
        text=[f"{m:.1f}%" for m in mapes],
        textposition="outside",
        hovertemplate="%{x}<br>MAPE: %{y:.1f}%<br>Lots: %{customdata:,}<extra></extra>",
        customdata=counts,
    ))

    _base_layout(fig, "MAPE by Price Range", height=440)
    fig.update_layout(
        xaxis_title="Price Bucket",
        yaxis_title="MAPE (%)",
    )
    return fig
