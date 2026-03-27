"""
Centralized CSS and HTML component helpers for the Indian Art Market
Streamlit dashboard. Dark-themed, inspired by Sentra UI with teal/cyan accents.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_DEEPEST = "#0c0c1a"
BG_CARD = "#1e1e2e"
BG_BORDER = "#2a2a3e"
ACCENT_TEAL = "#00bfa5"
ACCENT_CYAN = "#26c6da"
TEXT_PRIMARY = "#e4eef5"
TEXT_SECONDARY = "#8899aa"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Root / body ---- */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: %(bg_deepest)s !important;
    color: %(text_primary)s !important;
    font-family: 'Inter', sans-serif !important;
}

/* ---- Main content area ---- */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: %(bg_card)s !important;
    border-right: 1px solid %(bg_border)s !important;
}
[data-testid="stSidebar"] * {
    color: %(text_primary)s !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: %(text_secondary)s !important;
    font-weight: 500;
}

/* ---- Cards (generic containers) ---- */
div[data-testid="stExpander"],
div[data-testid="stMetric"],
div[data-testid="metric-container"] {
    background-color: %(bg_card)s !important;
    border: 1px solid %(bg_border)s !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

/* ---- Metrics ---- */
[data-testid="stMetricValue"] {
    color: %(accent_teal)s !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: %(text_secondary)s !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] svg {
    display: inline !important;
}

/* ---- DataFrames / tables ---- */
.stDataFrame, .stTable {
    background-color: %(bg_card)s !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
.stDataFrame [data-testid="StyledLinkCell"],
.stDataFrame thead th {
    background-color: %(bg_border)s !important;
    color: %(text_primary)s !important;
}
.stDataFrame tbody td {
    color: %(text_primary)s !important;
    border-color: %(bg_border)s !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent !important;
    border-bottom: 2px solid %(bg_border)s !important;
}
.stTabs [data-baseweb="tab"] {
    color: %(text_secondary)s !important;
    font-weight: 500;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 20px !important;
    background-color: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: %(accent_teal)s !important;
    border-bottom: 3px solid %(accent_teal)s !important;
    background-color: rgba(0, 191, 165, 0.08) !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background-color: %(accent_teal)s !important;
    color: %(bg_deepest)s !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ---- Selectboxes / Inputs ---- */
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"],
.stTextInput input,
.stNumberInput input {
    background-color: %(bg_card)s !important;
    border: 1px solid %(bg_border)s !important;
    border-radius: 10px !important;
    color: %(text_primary)s !important;
}
.stSelectbox [data-baseweb="select"]:focus-within,
.stMultiSelect [data-baseweb="select"]:focus-within,
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: %(accent_teal)s !important;
    box-shadow: 0 0 0 1px %(accent_teal)s !important;
}

/* Dropdown menu */
[data-baseweb="popover"] {
    background-color: %(bg_card)s !important;
    border: 1px solid %(bg_border)s !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"] li {
    color: %(text_primary)s !important;
}
[data-baseweb="popover"] li:hover {
    background-color: %(bg_border)s !important;
}

/* ---- Slider ---- */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: %(accent_teal)s !important;
}

/* ---- Markdown text ---- */
.stMarkdown p, .stMarkdown li {
    color: %(text_primary)s !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: %(text_primary)s !important;
    font-family: 'Inter', sans-serif !important;
}

/* ---- Custom scrollbar ---- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: %(bg_deepest)s;
}
::-webkit-scrollbar-thumb {
    background: %(bg_border)s;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: %(text_secondary)s;
}

/* ---- Plotly chart containers ---- */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ---- Hide Streamlit branding ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""" % {
    "bg_deepest": BG_DEEPEST,
    "bg_card": BG_CARD,
    "bg_border": BG_BORDER,
    "accent_teal": ACCENT_TEAL,
    "accent_cyan": ACCENT_CYAN,
    "text_primary": TEXT_PRIMARY,
    "text_secondary": TEXT_SECONDARY,
}

# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=13),
        title=dict(font=dict(size=18, color=TEXT_PRIMARY)),
        xaxis=dict(
            gridcolor=BG_BORDER,
            linecolor=BG_BORDER,
            zerolinecolor=BG_BORDER,
            tickfont=dict(color=TEXT_SECONDARY),
            title_font=dict(color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            gridcolor=BG_BORDER,
            linecolor=BG_BORDER,
            zerolinecolor=BG_BORDER,
            tickfont=dict(color=TEXT_SECONDARY),
            title_font=dict(color=TEXT_SECONDARY),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_PRIMARY, size=12),
            bordercolor=BG_BORDER,
            borderwidth=1,
        ),
        colorway=[
            ACCENT_TEAL, ACCENT_CYAN, "#ff7043", "#ab47bc",
            "#ffa726", "#66bb6a", "#ef5350", "#42a5f5",
        ],
        hoverlabel=dict(
            bgcolor=BG_CARD,
            font_size=13,
            font_color=TEXT_PRIMARY,
            bordercolor=BG_BORDER,
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    ),
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def inject_css():
    """Inject the global CSS into the Streamlit app."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = ""):
    """Render a styled page-level banner."""
    icon_html = f'<span style="font-size:1.6rem;margin-right:10px;">{icon}</span>' if icon else ""
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{BG_CARD} 0%,rgba(0,191,165,0.12) 100%);border:1px solid {BG_BORDER};border-radius:14px;padding:1.8rem 2rem;margin-bottom:1.5rem;">'
        f'<div style="display:flex;align-items:center;">{icon_html}'
        f'<h1 style="margin:0;font-size:1.8rem;font-weight:700;color:{TEXT_PRIMARY};font-family:Inter,sans-serif;">{title}</h1>'
        f'</div>'
        f'<p style="margin:0.4rem 0 0 0;color:{TEXT_SECONDARY};font-size:0.95rem;">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    """Render a section divider with a teal accent bar."""
    subtitle_html = (
        f'<p style="margin:0.2rem 0 0 0;color:{TEXT_SECONDARY};font-size:0.85rem;">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin:1.8rem 0 1rem 0;">'
        f'<div style="width:40px;height:4px;background:{ACCENT_TEAL};border-radius:2px;margin-bottom:0.5rem;"></div>'
        f'<h2 style="margin:0;font-size:1.25rem;font-weight:600;color:{TEXT_PRIMARY};font-family:Inter,sans-serif;">{title}</h2>'
        f'{subtitle_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str = None, color: str = ACCENT_TEAL) -> str:
    """Return HTML string for a single KPI metric card."""
    delta_html = ""
    if delta is not None:
        is_positive = not str(delta).startswith("-")
        arrow = "&#9650;" if is_positive else "&#9660;"
        delta_color = "#66bb6a" if is_positive else "#ef5350"
        delta_html = f'<div style="margin-top:6px;font-size:0.8rem;color:{delta_color};font-weight:500;">{arrow} {delta}</div>'
    return (
        f'<div style="background-color:{BG_CARD};border:1px solid {BG_BORDER};border-radius:14px;padding:1.2rem 1.4rem;text-align:center;">'
        f'<div style="color:{TEXT_SECONDARY};font-size:0.8rem;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
        f'<div style="color:{color};font-size:1.6rem;font-weight:700;margin-top:6px;">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def sidebar_brand():
    """Render sidebar branding block."""
    st.sidebar.markdown(
        f'<div style="padding:1rem 0 1.5rem 0;border-bottom:1px solid {BG_BORDER};margin-bottom:1rem;">'
        f'<div style="font-size:1.3rem;font-weight:700;color:{TEXT_PRIMARY};font-family:Inter,sans-serif;line-height:1.3;">Indian Art Market</div>'
        f'<div style="font-size:0.8rem;font-weight:500;color:{ACCENT_TEAL};margin-top:2px;letter-spacing:0.5px;">ML Price Predictor</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
