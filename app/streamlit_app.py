"""
NYC Taxi Trip Duration Predictor - Streamlit Portfolio App

Apple-elegance design system: white surfaces, refined amber accent,
custom SVG icon set, micro-interactions. Functional first.

Run:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR      = Path(__file__).parent

# Palette
INK         = "#1D1D1F"
GRAPHITE    = "#535359"
SLATE       = "#86868B"
MIST        = "#A1A1A6"

ACCENT      = "#B8943F"
ACCENT_DARK = "#7A6027"
ACCENT_SOFT = "#F7F1E4"

WHITE       = "#FFFFFF"
CREAM       = "#FAFAF8"
SOFT        = "#F4F3EF"
BORDER      = "#ECEAE5"
DIVIDER     = "#F0EEEA"

SUCCESS     = "#2A8A45"
WARN        = "#B8743F"
ERROR       = "#A04243"

PLOTLY_FONT = dict(
    family='-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif',
    size=12,
    color=GRAPHITE,
)

CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "RatecodeID",
                         "VendorID", "payment_type", "day_of_week"]

# Custom SVG icon set
def _icon(path: str, size: int = 18, color: str = "currentColor", stroke: float = 1.6) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align: -3px; display: inline-block; flex-shrink: 0;">'
        f'{path}</svg>'
    )

ICON_PATHS = {
    "overview":  '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "explore":   '<path d="M3 21V3"/><path d="M3 21h18"/><path d="M7 16l4-4 3 3 5-7"/>',
    "results":   '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/>',
    "build":     '<path d="M12 2L2 7l10 5 10-5L12 2z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
    "taxi":      '<path d="M5 17h2l1-6h8l1 6h2"/><path d="M6 11l1-4h10l1 4"/><circle cx="8" cy="17" r="1.5"/><circle cx="16" cy="17" r="1.5"/>',
    "clock":     '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
    "calendar":  '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18"/><path d="M8 3v4"/><path d="M16 3v4"/>',
    "pin":       '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="2.5"/>',
    "network":   '<circle cx="12" cy="12" r="2.2"/><circle cx="5" cy="6" r="1.8"/><circle cx="19" cy="6" r="1.8"/><circle cx="5" cy="18" r="1.8"/><circle cx="19" cy="18" r="1.8"/><path d="M6.5 7L10 11"/><path d="M17.5 7L14 11"/><path d="M6.5 17L10 13"/><path d="M17.5 17L14 13"/>',
    "alert":     '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><path d="M12 9v4"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/>',
    "moon":      '<path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>',
    "plane":     '<path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/>',
    "check":     '<path d="M20 6L9 17l-5-5"/>',
    "spark":     '<path d="M12 3l2.39 6.96L21.5 11l-5.55 4.5L17.39 22 12 18.4 6.61 22l1.44-6.5L2.5 11l7.11-1.04L12 3z"/>',
    "lightbulb": '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 00-4 12.74V17h8v-2.26A7 7 0 0012 2z"/>',
    "loop":      '<path d="M3 12a9 9 0 0115-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 01-15 6.7L3 16"/><path d="M3 21v-5h5"/>',
    "arrow":     '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
    "down":      '<path d="M19 9l-7 7-7-7"/>',
    "up":        '<path d="M5 15l7-7 7 7"/>',
    "code":      '<path d="M16 18l6-6-6-6"/><path d="M8 6l-6 6 6 6"/>',
    "play":      '<path d="M8 5v14l11-7z"/>',
    "layers":    '<path d="M3 17l9 5 9-5"/><path d="M3 12l9 5 9-5"/><path d="M12 2L3 7l9 5 9-5-9-5z"/>',
    "filter":    '<path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z"/>',
    "shield":    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "compass":   '<circle cx="12" cy="12" r="9"/><path d="M16.24 7.76l-2.12 6.36-6.36 2.12 2.12-6.36 6.36-2.12z"/>',
    "history":   '<path d="M3 12a9 9 0 1 0 9-9 9.74 9.74 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
}

def icon(key: str, size: int = 18, color: str = "currentColor") -> str:
    return _icon(ICON_PATHS.get(key, ""), size=size, color=color)

# Plotly chart styling
def style_chart(fig, height: int = 340, show_legend: bool = True):
    fig.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=PLOTLY_FONT,
        title=dict(font=dict(size=14, color=INK, family=PLOTLY_FONT["family"]), x=0.0, xanchor="left"),
        height=height,
        margin=dict(t=46, b=24, l=14, r=14),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=GRAPHITE, size=12),
            borderwidth=0,
        ),
        xaxis=dict(
            gridcolor=DIVIDER, linecolor=BORDER, zerolinecolor=DIVIDER,
            tickfont=dict(color=SLATE, size=11),
            title_font=dict(color=GRAPHITE, size=12),
        ),
        yaxis=dict(
            gridcolor=DIVIDER, linecolor=BORDER, zerolinecolor=DIVIDER,
            tickfont=dict(color=SLATE, size=11),
            title_font=dict(color=GRAPHITE, size=12),
        ),
    )
    return fig

# CSS - Apple Elegance design system
st.markdown(f"""
<style>
/* ---- Reset & base ------------------------------------------------------ */
.stApp {{
    background: {CREAM};
    color: {INK};
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                 "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: "kern", "liga";
}}
.main .block-container {{
    padding-top: 2.5rem;
    padding-bottom: 5rem;
    max-width: 1200px;
}}
.stApp [data-testid="stHeader"] {{ background: transparent; }}
.stApp [data-testid="stToolbar"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ---- Sidebar ----------------------------------------------------------- */
[data-testid="stSidebar"] {{ background: {WHITE}; }}
[data-testid="stSidebar"] > div:first-child {{
    background: {WHITE};
    border-right: 1px solid {BORDER};
}}

.sidebar-brand {{
    padding: 24px 18px 22px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 18px;
}}
.sidebar-mark {{
    width: 34px; height: 34px;
    background: {INK};
    color: {WHITE};
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
}}
.sidebar-name {{
    font-size: 14px; font-weight: 600;
    color: {INK}; letter-spacing: -0.005em;
    line-height: 1.2;
}}
.sidebar-tag {{
    font-size: 11px; color: {SLATE};
    margin-top: 2px; letter-spacing: 0.02em;
}}

[data-testid="stSidebar"] .stRadio > div {{ gap: 2px; padding: 0 6px; }}
[data-testid="stSidebar"] .stRadio label {{
    padding: 9px 12px;
    border-radius: 9px;
    transition: background 0.18s ease, color 0.18s ease;
    cursor: pointer;
}}
[data-testid="stSidebar"] .stRadio label:hover {{ background: {SOFT}; }}
[data-testid="stSidebar"] .stRadio label p {{
    color: {GRAPHITE} !important;
    font-size: 14px; font-weight: 500;
    margin: 0;
}}
[data-testid="stSidebar"] .stRadio label [data-baseweb="radio"] {{ display: none; }}
[data-testid="stSidebar"] hr {{ border-color: {BORDER}; margin: 18px 12px; }}

.sidebar-stats {{
    padding: 6px 18px;
    color: {SLATE};
    font-size: 11px;
    line-height: 1.9;
    letter-spacing: 0.01em;
}}
.sidebar-stats .row {{ display: flex; justify-content: space-between; }}
.sidebar-stats .row strong {{ color: {INK}; font-weight: 500; font-variant-numeric: tabular-nums; }}

/* ---- Typography -------------------------------------------------------- */
h1, h2, h3, h4 {{
    font-family: inherit;
    letter-spacing: -0.018em;
    color: {INK};
    font-weight: 600;
}}

/* Hero */
.hero {{
    padding: 24px 0 36px;
    animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}}
.hero-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: {ACCENT_SOFT};
    color: {ACCENT_DARK};
    padding: 6px 12px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 22px;
}}
.hero-title {{
    font-size: 56px;
    font-weight: 600;
    letter-spacing: -0.028em;
    color: {INK};
    line-height: 1.02;
    max-width: 760px;
    margin: 0;
}}
.hero-subtitle {{
    font-size: 19px;
    font-weight: 400;
    color: {GRAPHITE};
    margin: 18px 0 0;
    max-width: 640px;
    line-height: 1.5;
}}

/* Section heading */
.section-head {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SLATE};
    margin: 36px 0 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-head svg {{ color: {ACCENT}; }}

.page-head {{
    font-size: 34px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: {INK};
    margin: 8px 0 4px;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.page-head svg {{ color: {ACCENT}; }}
.page-caption {{
    color: {SLATE};
    font-size: 14px;
    margin-bottom: 24px;
}}

/* ---- KPI cards --------------------------------------------------------- */
.kpi {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px 22px 18px;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1),
                box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1),
                border-color 0.25s ease;
    height: 100%;
    animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) backwards;
}}
.kpi:nth-child(1) {{ animation-delay: 0.05s; }}
.kpi:nth-child(2) {{ animation-delay: 0.12s; }}
.kpi:nth-child(3) {{ animation-delay: 0.19s; }}
.kpi:nth-child(4) {{ animation-delay: 0.26s; }}
.kpi:hover {{
    border-color: #D7D2C8;
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(29, 29, 31, 0.05);
}}
.kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SLATE};
    display: flex;
    align-items: center;
    gap: 6px;
}}
.kpi-value {{
    font-size: 38px;
    font-weight: 600;
    letter-spacing: -0.025em;
    color: {INK};
    line-height: 1.05;
    margin: 10px 0 6px;
    font-variant-numeric: tabular-nums;
}}
.kpi-delta {{
    font-size: 13px;
    color: {GRAPHITE};
    line-height: 1.4;
}}
.kpi-delta-up {{ color: {SUCCESS}; font-weight: 500; }}

/* ---- Pills ------------------------------------------------------------- */
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {WHITE};
    border: 1px solid {BORDER};
    color: {INK};
    padding: 6px 12px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 500;
    margin: 3px 4px 3px 0;
    transition: border-color 0.18s ease, background 0.18s ease;
}}
.pill:hover {{
    border-color: #D7D2C8;
    background: {CREAM};
}}
.pill-group-label {{
    font-size: 11px;
    font-weight: 600;
    color: {SLATE};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 12px 0 6px;
}}

/* ---- Cards ------------------------------------------------------------- */
.card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 24px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}}
.card:hover {{
    border-color: #D7D2C8;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(29, 29, 31, 0.04);
}}

/* ---- Callouts ---------------------------------------------------------- */
.callout {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    margin: 12px 0;
    transition: border-color 0.2s ease;
}}
.callout:hover {{ border-color: #D7D2C8; }}
.callout-title {{
    font-size: 13px;
    font-weight: 600;
    color: {INK};
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 7px;
}}
.callout-title svg {{ color: {ACCENT}; }}
.callout-body {{
    color: {GRAPHITE};
    font-size: 14px;
    line-height: 1.55;
}}
.callout-body code {{
    background: {SOFT};
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 13px;
    color: {ACCENT_DARK};
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}}

/* ---- Tabs -------------------------------------------------------------- */
[data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {BORDER};
    gap: 0;
    margin-bottom: 4px;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    color: {SLATE} !important;
    font-weight: 500 !important;
    padding: 12px 18px !important;
    border-radius: 0 !important;
    transition: color 0.18s ease, border-color 0.18s ease;
    border-bottom: 2px solid transparent !important;
    font-size: 14px !important;
}}
[data-baseweb="tab"]:hover {{ color: {INK} !important; }}
[data-baseweb="tab"][aria-selected="true"] {{
    color: {INK} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 18px; }}

/* ---- Inputs ------------------------------------------------------------ */
.stSelectbox label, .stSlider label {{
    font-size: 12px !important;
    font-weight: 600 !important;
    color: {SLATE} !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px !important;
}}
.stCheckbox label {{
    color: {GRAPHITE} !important;
    font-size: 13px !important;
}}
.stSelectbox > div > div {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    transition: border-color 0.2s ease;
}}
.stSelectbox > div > div:hover {{ border-color: #D7D2C8; }}

.stSlider [data-baseweb="slider"] > div > div > div {{ background: {ACCENT}; }}
.stSlider [role="slider"] {{
    background: {WHITE} !important;
    border: 2px solid {ACCENT} !important;
    box-shadow: 0 2px 6px rgba(29,29,31,0.08) !important;
}}
.stSlider [role="slider"]:hover {{ transform: scale(1.08); transition: transform 0.18s ease; }}

/* ---- Buttons ----------------------------------------------------------- */
.stButton > button {{
    background: {INK};
    color: {WHITE};
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 500;
    font-size: 14px;
    transition: background 0.18s ease, transform 0.18s ease;
}}
.stButton > button:hover {{
    background: #2D2D2F;
    transform: translateY(-1px);
}}
.stButton > button:active {{ transform: translateY(0); }}

/* ---- Tables / DataFrame ------------------------------------------------ */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* ---- Model cards ------------------------------------------------------- */
.model-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}}
.model-card:hover {{
    border-color: #D7D2C8;
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(29, 29, 31, 0.04);
}}
.model-card-winner {{
    background: linear-gradient(180deg, {WHITE} 0%, #FBF7EC 100%);
    border-color: #E5D9B8;
}}
.winner-tag {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: {ACCENT};
    color: {WHITE};
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-left: 10px;
}}

/* ---- Prediction --------------------------------------------------------- */
.prediction {{
    background: linear-gradient(180deg, {WHITE} 0%, {CREAM} 100%);
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 38px 28px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}}
.prediction:hover {{
    border-color: #D7D2C8;
    box-shadow: 0 16px 40px rgba(29, 29, 31, 0.05);
}}
.prediction-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SLATE};
}}
.prediction-value {{
    font-size: 64px;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: {INK};
    line-height: 1;
    margin: 12px 0 4px;
    font-variant-numeric: tabular-nums;
    animation: numberReveal 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}}
.prediction-sub {{ font-size: 14px; color: {GRAPHITE}; }}
.prediction-context {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 18px;
    padding: 7px 14px;
    background: {ACCENT_SOFT};
    border-radius: 100px;
    font-size: 12px;
    font-weight: 500;
    color: {ACCENT_DARK};
}}
.prediction-context svg {{ color: {ACCENT_DARK}; }}

.factor-row {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid {DIVIDER};
}}
.factor-row:last-child {{ border-bottom: none; }}
.factor-icon {{
    width: 30px; height: 30px;
    background: {SOFT};
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    color: {ACCENT_DARK};
}}
.factor-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {SLATE};
}}
.factor-value {{
    font-size: 14px;
    color: {INK};
    margin-top: 1px;
}}

/* ---- Timeline ---------------------------------------------------------- */
.timeline-item {{
    position: relative;
    padding: 14px 0 14px 30px;
    border-left: 1.5px solid {BORDER};
    margin-left: 4px;
    transition: border-left-color 0.25s ease;
}}
.timeline-item:hover {{ border-left-color: {ACCENT}; }}
.timeline-item::before {{
    content: '';
    position: absolute;
    left: -6px;
    top: 20px;
    width: 11px; height: 11px;
    background: {WHITE};
    border: 2px solid #D7D2C8;
    border-radius: 50%;
    transition: all 0.25s ease;
}}
.timeline-item:hover::before {{
    background: {ACCENT};
    border-color: {ACCENT};
    transform: scale(1.15);
}}
.timeline-day {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
}}
.timeline-title {{
    font-size: 16px;
    font-weight: 600;
    color: {INK};
    margin-top: 3px;
}}
.timeline-desc {{
    font-size: 14px;
    color: {GRAPHITE};
    line-height: 1.55;
    margin-top: 4px;
}}

/* ---- Pipeline table --------------------------------------------------- */
.pipeline-row {{
    display: grid;
    grid-template-columns: 80px 1fr 1fr 1.2fr;
    padding: 14px 0;
    border-bottom: 1px solid {DIVIDER};
    align-items: center;
}}
.pipeline-row:last-child {{ border-bottom: none; }}
.pipeline-header {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SLATE};
    padding-bottom: 10px;
    border-bottom: 1px solid {BORDER};
}}
.pipeline-stage {{ font-weight: 600; color: {INK}; }}
.pipeline-cell {{ color: {GRAPHITE}; font-size: 14px; }}

/* ---- Footer ------------------------------------------------------------ */
.footer {{
    text-align: center;
    color: {SLATE};
    font-size: 12px;
    padding: 28px 0 12px;
    margin-top: 48px;
    border-top: 1px solid {BORDER};
    letter-spacing: 0.01em;
}}

/* ---- Animations -------------------------------------------------------- */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes numberReveal {{
    from {{ opacity: 0; transform: translateY(6px); letter-spacing: -0.02em; }}
    to   {{ opacity: 1; transform: translateY(0); letter-spacing: -0.03em; }}
}}

/* Make Streamlit-rendered links inherit our color */
a {{ color: {ACCENT_DARK}; text-decoration: none; }}
a:hover {{ color: {ACCENT}; text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)


# NYC zone map
AIRPORT_ZONES   = {132, 138, 1}
MANHATTAN_ZONES = set(range(4, 153)) | {161, 162, 163, 164, 166, 170, 186,
                                         194, 202, 209, 211, 224, 229, 230,
                                         231, 232, 233, 234, 236, 237, 238,
                                         239, 243, 244, 246, 249, 261, 262}

POPULAR_ZONES = {
    "Midtown Center":              161,
    "Times Square / Theatre Dist": 230,
    "JFK Airport":                 132,
    "LaGuardia Airport":           138,
    "Penn Station / Madison Sq W": 186,
    "Grand Central":               234,
    "Financial District North":     87,
    "Financial District South":     88,
    "Upper East Side North":       236,
    "Upper East Side South":       237,
    "Upper West Side North":       238,
    "Upper West Side South":       239,
    "Hell's Kitchen North":        113,
    "Midtown East":                162,
    "East Village":                 79,
    "Greenwich Village North":     103,
    "Soho":                        211,
    "Tribeca / Civic Center":      246,
    "Astoria (Queens)":             7,
    "Williamsburg (Brooklyn)":     261,
    "Crown Heights (Brooklyn)":     61,
    "Harlem":                       74,
    "Washington Heights":          152,
    "Lower East Side":             148,
    "Murray Hill":                 170,
}
ZONE_NAMES = list(POPULAR_ZONES.keys())
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# Data loading
@st.cache_data
def load_model_results():
    path = APP_DIR / "model_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _demo_model_results()

@st.cache_data
def load_predictions():
    path = APP_DIR / "predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return _demo_predictions()

@st.cache_data
def load_feature_sample(n=20_000):
    bundled = APP_DIR / "eda_sample.parquet"
    full    = PROJECT_ROOT / "data" / "features" / "yellow_tripdata_2026-01_features.parquet"
    EDA_COLS = ["duration_sec", "trip_distance", "hour", "day_of_week",
                "fare_per_mile", "is_rush_hour", "is_weekend",
                "is_pu_manhattan", "is_airport_trip"]
    for path in (bundled, full):
        if path.exists():
            df = pd.read_parquet(path, columns=EDA_COLS)
            return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    return _demo_feature_sample(n)

@st.cache_resource
def load_model():
    path = PROJECT_ROOT / "models" / "production_model.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def _demo_model_results():
    return {
        "models": [
            {"name": "Linear Regression", "r2": 0.761, "mae_sec": 4667.0, "mae_min": 77.8, "rmse_log": 0.347,  "description": "Baseline, straight-line relationships only.",   "winner": False},
            {"name": "Ridge Regression",  "r2": 0.761, "mae_sec": 4666.0, "mae_min": 77.8, "rmse_log": 0.347,  "description": "Linear + L2, near-identical to baseline.",      "winner": False},
            {"name": "LightGBM",          "r2": 0.983, "mae_sec": 55.0,   "mae_min": 0.9,  "rmse_log": 0.0935, "description": "Gradient boosted trees.",                       "winner": True},
        ],
        "feature_importances": {"DOLocationID": 12880, "PULocationID": 10014, "fare_per_mile": 8780, "trip_distance": 4824, "hour": 1934},
        "stats": {"total_trips": 2379881, "n_features": 34, "best_r2": 0.983, "best_mae_min": 0.9, "baseline_mae_min": 77.8},
    }

def _demo_predictions():
    rng = np.random.default_rng(42)
    n = 5000
    actual = rng.exponential(700, n).clip(60, 10800)
    noise  = rng.normal(0, 50, n)
    predicted = (actual + noise).clip(60, 10800)
    return pd.DataFrame({
        "actual_sec": actual, "predicted_sec": predicted,
        "actual_min": actual/60, "predicted_min": predicted/60,
        "error_sec": predicted - actual,
        "trip_distance": rng.exponential(2.5, n).clip(0.1, 30),
        "hour": rng.integers(0, 24, n),
        "is_rush_hour": rng.integers(0, 2, n),
    })

def _demo_feature_sample(n):
    rng = np.random.default_rng(42)
    duration = rng.exponential(700, n).clip(60, 10800)
    hour     = rng.integers(0, 24, n)
    return pd.DataFrame({
        "duration_sec": duration, "trip_distance": rng.exponential(2.5, n).clip(0.1,30),
        "hour": hour, "day_of_week": rng.integers(0, 7, n),
        "fare_per_mile": rng.uniform(2, 12, n),
        "is_rush_hour": (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype(int),
        "is_weekend": rng.integers(0, 2, n), "is_pu_manhattan": rng.integers(0, 2, n),
        "is_airport_trip": rng.integers(0, 2, n),
    })


# Prediction helper
def build_input_row(pu_id, do_id, distance, hour, dow, passengers, fare_est):
    is_weekday   = int(dow < 5)
    is_am_rush   = int(is_weekday and 7 <= hour <= 9)
    is_pm_rush   = int(is_weekday and 16 <= hour <= 19)
    is_rush      = int(is_am_rush or is_pm_rush)
    is_weekend   = int(dow >= 5)
    is_late_night= int(hour >= 22 or hour <= 5)
    is_pu_man    = int(pu_id in MANHATTAN_ZONES)
    is_do_man    = int(do_id in MANHATTAN_ZONES)
    is_airport   = int(pu_id in AIRPORT_ZONES or do_id in AIRPORT_ZONES)
    is_same      = int(pu_id == do_id)
    is_both_man  = int(is_pu_man and is_do_man)
    fare_per_mile= fare_est / max(distance, 0.1)
    cbd_fee      = 0.75 if (is_pu_man or is_do_man) else 0.0

    return {
        "VendorID": 2, "passenger_count": passengers, "trip_distance": distance,
        "RatecodeID": 1, "PULocationID": pu_id, "DOLocationID": do_id,
        "payment_type": 1,
        "fare_amount": fare_est, "extra": 1.0, "mta_tax": 0.5,
        "tip_amount": 2.85, "tolls_amount": 0.0,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0 if not is_airport else 1.75,
        "cbd_congestion_fee": cbd_fee,
        "hour": hour, "day_of_week": dow,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow / 7),
        "dow_cos":  np.cos(2 * np.pi * dow / 7),
        "is_am_rush": is_am_rush, "is_pm_rush": is_pm_rush, "is_rush_hour": is_rush,
        "is_weekend": is_weekend, "is_late_night": is_late_night,
        "is_pu_manhattan": is_pu_man, "is_do_manhattan": is_do_man,
        "is_airport_trip": is_airport, "is_same_zone": is_same,
        "is_both_manhattan": is_both_man,
        "fare_per_mile": fare_per_mile,
        "distance_x_rush": distance * is_rush,
        "distance_x_night": distance * is_late_night,
    }


# Helpers for inline section headers
def section(title: str, icon_key: str = None):
    icon_html = icon(icon_key, size=14, color=ACCENT) if icon_key else ""
    st.markdown(f'<div class="section-head">{icon_html}<span>{title}</span></div>', unsafe_allow_html=True)

def page_title(title: str, icon_key: str, caption: str = None):
    st.markdown(
        f'<div class="page-head">{icon(icon_key, size=28)}<span>{title}</span></div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="page-caption">{caption}</div>', unsafe_allow_html=True)


# Pages
def page_overview():
    results = load_model_results()
    stats   = results["stats"]

    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">{icon('spark', size=12, color=ACCENT_DARK)}<span>ML Portfolio Project</span></div>
        <h1 class="hero-title">NYC taxi trip duration, predicted before the meter starts.</h1>
        <p class="hero-subtitle">An end-to-end machine learning pipeline trained on 2.38 million cleaned January 2026 yellow taxi trips. Average error: about one minute.</p>
    </div>
    """, unsafe_allow_html=True)

    section("Key numbers", "compass")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{icon('layers', size=12, color=SLATE)}<span>Trips analyzed</span></div>
            <div class="kpi-value">2.38M</div>
            <div class="kpi-delta">January 2026 NYC TLC data</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{icon('filter', size=12, color=SLATE)}<span>Features engineered</span></div>
            <div class="kpi-value">{stats['n_features']}</div>
            <div class="kpi-delta">from 19 raw columns</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{icon('results', size=12, color=SLATE)}<span>Model R²</span></div>
            <div class="kpi-value">{stats['best_r2']:.3f}</div>
            <div class="kpi-delta">explains {stats['best_r2']*100:.1f}% of variation</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        improvement = (stats['baseline_mae_min'] - stats['best_mae_min']) / stats['baseline_mae_min'] * 100
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{icon('clock', size=12, color=SLATE)}<span>Average error</span></div>
            <div class="kpi-value">~{stats['best_mae_min']:.0f} min</div>
            <div class="kpi-delta"><span class="kpi-delta-up">↑ {improvement:.0f}%</span> vs baseline</div>
        </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        section("What this project does", "lightbulb")
        st.markdown("""
This project answers a question every NYC taxi passenger asks: **how long will this take?**

Starting from raw TLC trip records, the pipeline cleans 3.7 million trips down to a clean 2.38 million, engineers 34 domain-aware features (rush hour flags, borough signals, cyclic time encoding), and trains a LightGBM model that predicts trip duration before the taxi moves.

The result: an average error of about one minute, good enough to set accurate ETA expectations, optimize dispatch routing, and power real-time fare estimates.
        """)

        section("Pipeline overview", "loop")
        st.markdown('<div class="pipeline-row pipeline-header"><div>Stage</div><div>Input</div><div>Output</div><div>Key result</div></div>', unsafe_allow_html=True)
        pipeline = [
            ("Day 1", "Data cleaning",        "3.72M raw rows",   "2.38M clean rows",       "Removed 36% bad data"),
            ("Day 2", "Exploratory analysis", "2.38M clean rows", "Insights + decisions",   "Log transform + rush hour"),
            ("Day 3", "Feature engineering",  "19 raw columns",   "34 engineered features", "Cyclic time, borough flags"),
            ("Day 4", "Model training",       "34 features",      "3 trained models",       "R² 0.761 → 0.983"),
        ]
        for day, stage, inp, outp, result in pipeline:
            st.markdown(f"""
            <div class="pipeline-row">
                <div class="pipeline-cell" style="color:{ACCENT};font-weight:600;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;">{day}</div>
                <div class="pipeline-stage">{stage}<div class="pipeline-cell" style="font-weight:400;color:{SLATE};font-size:12px;">{inp}</div></div>
                <div class="pipeline-cell">{outp}</div>
                <div class="pipeline-cell" style="color:{INK};">{result}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        section("Tech stack", "code")
        tech = {
            "Data":     ["Pandas", "PyArrow", "Parquet"],
            "ML":       ["LightGBM", "Scikit-learn", "Optuna"],
            "Tracking": ["MLflow"],
            "App":      ["Streamlit", "Plotly"],
            "Quality":  ["Pytest", "Ruff", "GitHub Actions"],
            "Deploy":   ["Docker", "Docker Compose"],
        }
        for category, tools in tech.items():
            st.markdown(f'<div class="pill-group-label">{category}</div>', unsafe_allow_html=True)
            badges = "".join(f'<span class="pill">{t}</span>' for t in tools)
            st.markdown(badges, unsafe_allow_html=True)

        section("Model performance", "results")
        models = results["models"]
        names  = [m["name"].replace(" ", "<br>") for m in models]
        maes   = [m["mae_min"] for m in models]
        colors = [ACCENT if m.get("winner") else BORDER for m in models]
        text   = [f"{m['mae_min']:.1f} min" for m in models]

        fig = go.Figure(go.Bar(
            x=names, y=maes,
            marker=dict(color=colors, line=dict(width=0)),
            text=text, textposition="outside",
            textfont=dict(color=INK, size=12, family=PLOTLY_FONT["family"]),
        ))
        fig.update_layout(yaxis_title="MAE (minutes, log scale)", yaxis_type="log")
        style_chart(fig, height=290, show_legend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    _footer()


def page_explore():
    df = load_feature_sample()

    page_title("Explore the data", "explore",
               caption=f"Based on {len(df):,} randomly sampled trips from January 2026.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Duration", "Time patterns", "Location", "Correlations"
    ])

    with tab1:
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            log_toggle = st.checkbox(
                "Apply log transform (what the model trains on)",
                value=False,
                help="Right-skewed distribution. Log compresses the tail so the model focuses on typical trips."
            )
            if log_toggle:
                durations = np.log1p(df["duration_sec"])
                title = "Trip duration, log(1 + seconds)"
                xlabel = "log(1 + duration_sec)"
            else:
                durations = df["duration_sec"] / 60
                title = "Trip duration in minutes"
                xlabel = "Duration (minutes)"

            fig = px.histogram(x=durations, nbins=80,
                               color_discrete_sequence=[ACCENT],
                               labels={"x": xlabel}, title=title)
            fig.update_traces(marker_line_width=0)
            style_chart(fig, height=380, show_legend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col2:
            section("Key stats", "spark")
            dur_min = df["duration_sec"] / 60
            for label, value in [
                ("Median trip", f"{dur_min.median():.1f} min"),
                ("Mean trip",   f"{dur_min.mean():.1f} min"),
                ("Shortest",    f"{dur_min.min():.1f} min"),
                ("Longest",     f"{dur_min.max():.0f} min"),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:11px 0;border-bottom:1px solid {DIVIDER};">
                    <span style="color:{SLATE};font-size:13px;">{label}</span>
                    <span style="color:{INK};font-weight:600;font-variant-numeric:tabular-nums;">{value}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="callout" style="margin-top:18px;">
                <div class="callout-title">{icon('lightbulb', 14)}Why log transform?</div>
                <div class="callout-body">
                    The raw distribution is right-skewed. A few 3-hour trips would dominate training.
                    Log compresses the tail so the model learns from typical trips, not rare outliers.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            avg_by_hour = df.groupby("hour")["duration_sec"].mean().div(60).reset_index()
            avg_by_hour.columns = ["Hour", "Avg duration (min)"]

            fig = px.line(avg_by_hour, x="Hour", y="Avg duration (min)",
                          title="Average duration by hour",
                          markers=True,
                          color_discrete_sequence=[INK])
            fig.update_traces(line=dict(width=2),
                              marker=dict(size=7, color=ACCENT, line=dict(color=INK, width=1.5)))
            fig.add_vrect(x0=7,  x1=9,  fillcolor=ACCENT, opacity=0.10, line_width=0)
            fig.add_vrect(x0=16, x1=19, fillcolor=ACCENT, opacity=0.10, line_width=0)
            style_chart(fig, height=340, show_legend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            st.markdown(f"""
            <div class="callout">
                <div class="callout-title">{icon('clock', 14)}Rush hour shifts the curve</div>
                <div class="callout-body">
                    Average duration rises noticeably from early morning through mid-afternoon.
                    The PM rush peak is taller than the AM peak.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            avg_by_dow = df.groupby("day_of_week")["duration_sec"].mean().div(60).reset_index()
            avg_by_dow["Day"] = avg_by_dow["day_of_week"].map(dict(enumerate(DAYS)))
            avg_by_dow = avg_by_dow.rename(columns={"duration_sec": "Avg duration (min)"})

            fig = px.bar(avg_by_dow, x="Day", y="Avg duration (min)",
                         title="Average duration by day of week",
                         color="Avg duration (min)",
                         color_continuous_scale=[[0, ACCENT_SOFT], [1, ACCENT]])
            fig.update_traces(marker_line_width=0)
            style_chart(fig, height=340, show_legend=False)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            st.markdown(f"""
            <div class="callout">
                <div class="callout-title">{icon('calendar', 14)}Weekday vs weekend gap</div>
                <div class="callout-body">
                    Mid-week trips run noticeably longer than weekend trips. Commuter traffic stretches the curve.
                    Sunday is the shortest day on average.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            df_plot = df.copy()
            df_plot["duration_min"] = df_plot["duration_sec"] / 60
            df_plot["Route type"] = np.where(
                df_plot["is_airport_trip"] == 1, "Airport",
                np.where(df_plot["is_pu_manhattan"] == 1, "Manhattan pickup", "Other boroughs")
            )

            fig = px.box(
                df_plot.sample(min(5000, len(df_plot)), random_state=42),
                x="Route type", y="duration_min",
                color="Route type",
                title="Duration by route type",
                color_discrete_map={
                    "Airport": ACCENT,
                    "Manhattan pickup": INK,
                    "Other boroughs": SLATE,
                },
                labels={"duration_min": "Duration (minutes)"},
                category_orders={"Route type": ["Airport", "Manhattan pickup", "Other boroughs"]},
            )
            style_chart(fig, height=360, show_legend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col2:
            df_plot2 = df.sample(min(3000, len(df)), random_state=42).copy()
            df_plot2["duration_min"] = df_plot2["duration_sec"] / 60
            df_plot2["Time of day"] = np.where(df_plot2["is_rush_hour"] == 1, "Rush hour", "Off-peak")

            fig = px.scatter(
                df_plot2, x="trip_distance", y="duration_min",
                color="Time of day",
                color_discrete_map={"Rush hour": ACCENT, "Off-peak": INK},
                opacity=0.45,
                title="Distance vs duration",
                labels={"trip_distance": "Distance (miles)", "duration_min": "Duration (minutes)"},
                trendline="ols",
            )
            fig.update_traces(marker=dict(size=6, line=dict(width=0)),
                              selector=dict(mode="markers"))
            style_chart(fig, height=360)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(f"""
        <div class="callout">
            <div class="callout-title">{icon('spark', 14)}Same distance, different time</div>
            <div class="callout-body">
                The two trend lines diverge. At the same distance, rush hour trips consistently take longer than off-peak trips.
                That is exactly why the interaction feature <code>distance × rush_hour</code> made it into the model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        numeric_df = df[["duration_sec", "trip_distance", "fare_per_mile",
                          "hour", "is_rush_hour", "is_weekend",
                          "is_pu_manhattan", "is_airport_trip"]].copy()
        numeric_df.columns = ["Duration", "Distance", "Fare/Mile",
                               "Hour", "Rush hour", "Weekend",
                               "Manhattan PU", "Airport trip"]
        corr = numeric_df.corr().round(2)

        fig = px.imshow(corr, text_auto=True,
                        color_continuous_scale=["#7AA0C4", WHITE, ACCENT],
                        color_continuous_midpoint=0, zmin=-1, zmax=1,
                        title="Correlation matrix",
                        aspect="auto")
        fig.update_traces(textfont=dict(size=11, family=PLOTLY_FONT["family"], color=INK))
        style_chart(fig, height=440, show_legend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        corr_dist = corr.loc["Distance",  "Duration"]
        corr_fare = corr.loc["Fare/Mile", "Duration"]

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(f"""
            <div class="callout">
                <div class="callout-title">{icon('arrow', 14)}Strongest signal: Distance, {corr_dist:+.2f}</div>
                <div class="callout-body">
                    Longer trips take longer, but not perfectly linearly. Rush hour and zone effects break the straight-line relationship, which is why a tree model beats linear regression.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            direction = "negatively" if corr_fare < 0 else "positively"
            st.markdown(f"""
            <div class="callout">
                <div class="callout-title">{icon('alert', 14)}Surprising: Fare/Mile, {corr_fare:+.2f}</div>
                <div class="callout-body">
                    Fare-per-mile is {direction} correlated with duration. Long highway trips have low fare/mile (fast). Short city trips have high fare/mile (meter on time). The model uses this as a congestion proxy.
                </div>
            </div>
            """, unsafe_allow_html=True)

    _footer()


def page_models():
    results     = load_model_results()
    predictions = load_predictions()
    prod_bundle = load_model()

    page_title("Model results", "results")

    section("Model comparison", "history")
    models = results["models"]

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        for m in models:
            winner = m.get("winner", False)
            tag_html = f'<span class="winner-tag">{icon("check", 10, "#FFFFFF")}Winner</span>' if winner else ""
            klass = "model-card model-card-winner" if winner else "model-card"
            r2_color = ACCENT_DARK if winner else INK
            card_html = (
                f'<div class="{klass}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div style="display:flex;align-items:center;">'
                f'<span style="font-weight:600;font-size:15px;color:{INK};">{m["name"]}</span>'
                f'{tag_html}'
                f'</div>'
                f'<div style="font-size:18px;font-weight:600;color:{r2_color};font-variant-numeric:tabular-nums;">'
                f'R² = {m["r2"]:.3f}'
                f'</div>'
                f'</div>'
                f'<div style="color:{SLATE};font-size:13px;margin-top:6px;">{m["description"]}</div>'
                f'<div style="display:flex;gap:24px;margin-top:10px;font-size:12px;">'
                f'<span style="color:{SLATE};">MAE</span> '
                f'<span style="color:{INK};font-weight:500;font-variant-numeric:tabular-nums;">{m["mae_min"]:.1f} min</span>'
                f'<span style="color:{SLATE};margin-left:12px;">RMSE (log)</span> '
                f'<span style="color:{INK};font-weight:500;font-variant-numeric:tabular-nums;">{m["rmse_log"]:.3f}</span>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    with col2:
        colors = [ACCENT if m.get("winner") else BORDER for m in models]
        fig = go.Figure(go.Bar(
            x=[m["name"].replace(" ", "<br>") for m in models],
            y=[m["r2"] for m in models],
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{m['r2']:.3f}" for m in models],
            textposition="outside",
            textfont=dict(color=INK, size=12, family=PLOTLY_FONT["family"]),
        ))
        fig.update_layout(title="R² score by model", yaxis=dict(range=[0.5, 1.06]))
        style_chart(fig, height=320, show_legend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(f"""
        <div class="callout">
            <div class="callout-title">{icon('lightbulb', 14)}Why LightGBM wins</div>
            <div class="callout-body">
                Linear models assume straight-line relationships. Rush hour does not add time linearly, it compounds with distance and location. LightGBM's decision trees capture those interactions naturally. That is the gap from R² = 0.76 to R² = 0.98.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_imp, col_res = st.columns(2, gap="large")

    with col_imp:
        section("Top features", "filter")
        fi = results.get("feature_importances", {})
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=[[0, BORDER], [1, ACCENT]])
        fig.update_traces(marker_line_width=0)
        style_chart(fig, height=440, show_legend=False)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with col_res:
        section("Predicted vs actual", "spark")
        sample_pred = predictions.sample(min(2000, len(predictions)), random_state=42)
        fig = px.scatter(sample_pred, x="actual_min", y="predicted_min",
                         opacity=0.4,
                         color_discrete_sequence=[INK],
                         labels={"actual_min": "Actual duration (min)",
                                 "predicted_min": "Predicted duration (min)"},
                         title="LightGBM predictions on the test set")
        fig.update_traces(marker=dict(size=5, line=dict(width=0)))
        max_val = max(sample_pred["actual_min"].max(), sample_pred["predicted_min"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color=ACCENT, width=2, dash="dash"))
        fig.add_annotation(x=max_val*0.7, y=max_val*0.92, text="Perfect prediction",
                           showarrow=False,
                           font=dict(color=ACCENT_DARK, size=11))
        style_chart(fig, height=440, show_legend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)
    section("Try it yourself", "play")

    if prod_bundle is None:
        st.warning("Model file not found. Run `python src/models/run_training.py` first to enable live predictions.")
        _footer()
        return

    model        = prod_bundle["model"]
    feature_cols = prod_bundle["feature_cols"]

    col_form, col_result = st.columns([2, 1], gap="large")

    with col_form:
        c1, c2 = st.columns(2)
        with c1: pu_name = st.selectbox("Pickup location",  ZONE_NAMES, index=0)
        with c2: do_name = st.selectbox("Dropoff location", ZONE_NAMES, index=2)

        c3, c4 = st.columns(2)
        with c3: distance  = st.slider("Trip distance (miles)", 0.5, 25.0, 3.0, 0.5)
        with c4: passengers = st.slider("Passengers", 1, 6, 1)

        c5, c6 = st.columns(2)
        with c5: hour = st.slider("Pickup hour", 0, 23, 8)
        with c6:
            dow_label = st.selectbox("Day of week", DAYS)
            dow = DAYS.index(dow_label)

        fare_est = st.slider("Estimated fare ($)", 5.0, 100.0,
                             float(round(2.5 + distance * 2.5, 1)), 0.5,
                             help="Base rate is roughly $2.50 + $2.50/mile. Used to compute the fare-per-mile congestion signal.")

    with col_result:
        pu_id = POPULAR_ZONES[pu_name]
        do_id = POPULAR_ZONES[do_name]
        row   = build_input_row(pu_id, do_id, distance, hour, dow, passengers, fare_est)

        input_df = pd.DataFrame([row])[feature_cols]
        for col in CATEGORICAL_FEATURES:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype("category")

        pred_log = model.predict(input_df)[0]
        pred_sec = float(np.expm1(pred_log))
        pred_min = pred_sec / 60

        if pred_min >= 60:
            h = int(pred_min // 60)
            m = int(round(pred_min - h * 60))
            time_str = f"{h}h {m}m"
            sub_str  = f"{pred_min:.0f} minutes total"
        else:
            time_str = f"{pred_min:.1f}"
            sub_str  = "minutes"

        if 7 <= hour <= 9 and dow < 5:
            ctx_icon, ctx_text = "alert", "AM rush, expect delays"
        elif 16 <= hour <= 19 and dow < 5:
            ctx_icon, ctx_text = "alert", "PM rush, expect delays"
        elif hour >= 22 or hour <= 5:
            ctx_icon, ctx_text = "moon", "Late night, roads are clear"
        else:
            ctx_icon, ctx_text = "check", "Off-peak hours"

        airport_html = ""
        if pu_id in AIRPORT_ZONES or do_id in AIRPORT_ZONES:
            airport_html = f'<div class="prediction-context" style="margin-left:6px;background:{SOFT};color:{GRAPHITE};">{icon("plane", 12, GRAPHITE)}Airport trip</div>'

        st.markdown(f"""
        <div class="prediction">
            <div class="prediction-label">Estimated trip duration</div>
            <div class="prediction-value">{time_str}<span style="font-size:24px;color:{SLATE};font-weight:500;letter-spacing:-0.01em;"> {'min' if pred_min < 60 else ''}</span></div>
            <div class="prediction-sub">{sub_str}</div>
            <div class="prediction-context">{icon(ctx_icon, 12, ACCENT_DARK)}{ctx_text}</div>
            {airport_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if pu_id == do_id:
            route_str = "Same zone, very short trip"
        else:
            pu_loc = "Manhattan" if pu_id in MANHATTAN_ZONES else "Outer borough"
            do_loc = "Manhattan" if do_id in MANHATTAN_ZONES else "Outer borough"
            route_str = f"{pu_loc} to {do_loc}, {distance:.1f} miles"

        time_str_factor = f"{'Rush hour' if (7<=hour<=9 or 16<=hour<=19) and dow<5 else 'Off-peak'}, {dow_label} {hour:02d}:00"

        factors = [
            ("pin",      "Route", route_str),
            ("clock",    "Time",  time_str_factor),
        ]
        if pu_id in AIRPORT_ZONES or do_id in AIRPORT_ZONES:
            factors.append(("plane", "Airport leg", "Highway routing, faster per mile"))

        for icon_key, label, value in factors:
            st.markdown(f"""
            <div class="factor-row">
                <div class="factor-icon">{icon(icon_key, 14, ACCENT_DARK)}</div>
                <div>
                    <div class="factor-label">{label}</div>
                    <div class="factor-value">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    _footer()


def page_how_built():
    page_title("How I built this", "build")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        section("Architecture", "layers")
        st.graphviz_chart(f"""
        digraph pipeline {{
            rankdir=LR;
            bgcolor="transparent";
            node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, color="{BORDER}", fontcolor="{INK}"]
            edge [color="{SLATE}", fontname="Helvetica", fontsize=9, fontcolor="{SLATE}"]

            raw   [label="Raw TLC data\\n3.72M rows", fillcolor="{SOFT}"]
            clean [label="Cleaning\\ncleaner.py", fillcolor="{ACCENT_SOFT}", color="{ACCENT}"]
            feat  [label="Feature engineering\\nengineering.py", fillcolor="{ACCENT_SOFT}", color="{ACCENT}"]
            train [label="Training\\nrun_training.py", fillcolor="{ACCENT_SOFT}", color="{ACCENT}"]
            tune  [label="Optuna tuning\\ntuning.py", fillcolor="{ACCENT_SOFT}", color="{ACCENT}"]
            mlf   [label="MLflow tracking", fillcolor="{SOFT}"]
            prod  [label="Production model\\nLightGBM", fillcolor="{INK}", fontcolor="{WHITE}", color="{INK}"]
            app   [label="Streamlit app", fillcolor="{INK}", fontcolor="{WHITE}", color="{INK}"]

            raw   -> clean [label="36% removed"]
            clean -> feat  [label="34 features"]
            feat  -> train
            train -> mlf   [style=dashed]
            train -> tune
            tune  -> prod  [label="best params"]
            prod  -> app
        }}
        """)

        section("Build timeline", "history")
        timeline = [
            ("Day 1", "Data exploration and quality gate",
             "Downloaded 3.7 million NYC taxi trips. Built automated quality checks for schema, null rates, and value ranges. Cleaned data to 2.4 million rows by removing impossible values like negative fares, 300k-mile trips, and sub-60-second rides."),
            ("Day 2", "Exploratory data analysis",
             "Built a 7-section EDA notebook. Key discovery: trip duration is right-skewed, requiring log transform. Rush hour adds 3 to 4 minutes for the same route. These findings shaped the feature engineering strategy."),
            ("Day 3", "Feature engineering",
             "Engineered 34 features across 3 categories: temporal (cyclic hour and day encoding, rush hour flags), geospatial (Manhattan and airport zones, same-zone trips), and interaction (distance times rush hour). Feature selection dropped 5 redundant columns."),
            ("Day 4", "Model training and tuning",
             "Compared LinearRegression (MAE 77.8 min), Ridge (similar), and LightGBM (MAE 0.9 min) with 5-fold cross-validation. MLflow logged all runs. Optuna searched 30 hyperparameter combinations using Bayesian optimization."),
            ("Day 5", "Portfolio app",
             "Built this 4-page Streamlit app. Live prediction form lets users input any NYC route and get an instant duration estimate using the production LightGBM model."),
            ("Day 6", "Production hardening",
             "Containerized with Docker, wrote 8 pytest tests, set up GitHub Actions CI/CD for automated tests and lint on every push."),
        ]
        for day, title, desc in timeline:
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-day">{day}</div>
                <div class="timeline-title">{title}</div>
                <div class="timeline-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        section("Key decisions", "lightbulb")
        decisions = [
            ("Log-transform the target",
             "Trip duration is right-skewed. Training on raw seconds means a few 3-hour outliers dominate the loss function. log1p(duration) fixes this, discovered in Day 2 EDA."),
            ("300k sample for tuning",
             "Tuning on all 1.9M training rows × 30 trials × 5 folds would take 8+ hours. Hyperparameters generalize from a representative sample. Tune fast, train final model on full data."),
            ("Cyclic encoding for time",
             "Hour 23 and hour 0 are 1 hour apart but numerically 23 apart. Sin/cos encoding wraps the clock into a circle so the model sees midnight and 11pm as neighbors."),
            ("Trees over linear for log targets",
             "Linear regression on a log-transformed target can produce huge predictions after expm1 back-transform. Trees stay bounded by leaf values. This is the real reason production duration models use trees."),
        ]
        for title, body in decisions:
            st.markdown(f"""
            <div class="callout">
                <div class="callout-title">{icon('lightbulb', 14)}{title}</div>
                <div class="callout-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

        section("Lessons learned", "spark")
        lessons = [
            "EDA before modeling. Log transform came from the data, not a textbook.",
            "Feature engineering matters more than hyperparameter tuning.",
            "Always benchmark against a simple baseline before complex models.",
            "Nested parallelism (joblib × LightGBM) crashes on macOS, set n_jobs=1 for cross_val_score.",
            "When 36% of taxi data is unusable, quality gates aren't optional.",
        ]
        for l in lessons:
            st.markdown(
                f"<div style='display:flex;gap:8px;padding:8px 0;color:{GRAPHITE};font-size:14px;line-height:1.5;'>"
                f"<span style='color:{ACCENT};flex-shrink:0;'>{icon('arrow', 14, ACCENT)}</span>"
                f"<span>{l}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        section("Data source", "shield")
        st.markdown(f"""
        <div style="font-size:14px;color:{GRAPHITE};line-height:1.6;">
        <strong style="color:{INK};">NYC TLC trip record data</strong><br>
        New York City Taxi &amp; Limousine Commission.<br><br>
        January 2026 Yellow Taxi trips, 3,724,889 raw records.<br>
        Available at <a href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page">nyc.gov/tlc</a>.
        </div>
        """, unsafe_allow_html=True)

    _footer()


def _footer():
    st.markdown("""
    <div class="footer">
        NYC Taxi Trip Duration Predictor &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; January 2026 NYC TLC data
    </div>
    """, unsafe_allow_html=True)


# Navigation
def main():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="sidebar-mark">{icon('taxi', 18, WHITE)}</div>
            <div>
                <div class="sidebar-name">NYC Taxi Predictor</div>
                <div class="sidebar-tag">ML Portfolio</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["Overview", "Explore the data", "Model results", "How I built this"],
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-stats">
            <div class="row"><span>Model</span><strong>LightGBM</strong></div>
            <div class="row"><span>R²</span><strong>0.983</strong></div>
            <div class="row"><span>MAE</span><strong>~1 min</strong></div>
            <div class="row"><span>Data</span><strong>2.38M trips</strong></div>
        </div>
        """, unsafe_allow_html=True)

    if   page == "Overview":          page_overview()
    elif page == "Explore the data":  page_explore()
    elif page == "Model results":     page_models()
    elif page == "How I built this":  page_how_built()


if __name__ == "__main__":
    main()
