
import json
import html
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from predict import MODEL_PATH, predict_email_details, validate_email_text


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "model_metrics.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="ScripSift",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []
st.session_state.setdefault("active_page", "Overview")


# Theme tokens follow Streamlit's selected color scheme.
C = {
    "background": "var(--sg-background)",
    "surface": "var(--sg-surface)",
    "surface2": "var(--sg-surface-soft)",
    "border": "var(--sg-border)",
    "text": "var(--sg-text)",
    "muted": "var(--sg-muted)",
    "accent": "var(--sg-accent)",
    "success": "var(--sg-success)",
    "danger": "var(--sg-danger)",
    "grid": "var(--sg-grid)",
}

# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .stApp {{
        --sg-background: light-dark(#EEF1F5, #0B0B0D);
        --sg-surface: light-dark(#FFFFFF, #131316);
        --sg-surface-soft: light-dark(#E2E7EE, #1A1A1F);
        --sg-card: var(--sg-surface);
        --sg-surface-alt: var(--sg-surface-soft);
        --sg-text: light-dark(#0F172A, #F4F4F6);
        --sg-muted: light-dark(#475569, #8E8E98);
        --sg-border: light-dark(rgba(15, 23, 42, 0.12), rgba(255, 255, 255, 0.08));
        --sg-border-strong: color-mix(in srgb, var(--sg-text) 22%, var(--sg-border));
        --sg-grid: light-dark(rgba(0, 0, 0, 0.08), rgba(255, 255, 255, 0.08));
        --sg-card-shadow-color: light-dark(rgba(15, 23, 42, 0.05), transparent);
        --sg-accent: light-dark(#52525B, #A1A1AA);
        --sg-success: light-dark(#16A34A, #22C55E);
        --sg-danger: light-dark(#DC2626, #EF4444);
        background: {C["background"]};
        color: {C["text"]};
    }}

    ::selection,
    ::-moz-selection {{
        background: light-dark(rgba(0, 0, 0, 0.12), rgba(255, 255, 255, 0.16));
        color: inherit;
    }}

    .block-container {{
        width: 100%;
        max-width: 1800px;
        padding-top: 3.25rem;
        padding-bottom: 3rem;
    }}

    [data-testid="stToolbarActions"],
    .stAppToolbar [data-testid="stActionButton"] {{
        display: none !important;
    }}

    [data-testid="stAppDeployButton"],
    .stDeployButton {{
        display: none !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        display: none;
    }}

    [data-testid="stAppViewContainer"] {{
        background: {C["background"]};
    }}

    h1, h2, h3, h4, p, label {{
        color: {C["text"]};
    }}

    .brand-lockup {{
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
    }}

    .brand-mark {{
        display: flex;
        flex: 0 0 38px;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border: 1px solid {C["border"]};
        border-radius: 10px;
        background: {C["surface"]};
        color: {C["muted"]};
    }}

    .brand {{
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.15;
        color: {C["text"]};
        white-space: nowrap;
    }}

    .brand-mark svg {{
        display: block;
        width: 32px;
        height: 32px;
    }}

    .brand-caption {{
        margin-top: 4px;
        font-size: 0.72rem;
        line-height: 1.3;
        letter-spacing: 0.12em;
        color: {C["muted"]};
        font-weight: 600;
        white-space: nowrap;
    }}

    .hero {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 10px;
        padding: 28px 30px;
        margin: 16px 0 25px 0;
    }}

    .hero h1 {{
        color: {C["text"]};
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.8px;
        margin-bottom: 8px;
    }}

    .hero p {{
        color: {C["muted"]};
        font-size: 14px;
        margin-bottom: 0;
    }}

    .metric-card {{
        box-sizing: border-box;
        width: 100%;
        min-width: 0;
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 9px;
        padding: 16px;
        min-height: 125px;
    }}

    .metric-label {{
        color: {C["muted"]};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
    }}

    .metric-value {{
        color: {C["text"]};
        font-size: 29px;
        font-weight: 700;
        letter-spacing: -0.6px;
        margin-top: 13px;
    }}

    .metric-note {{
        color: {C["muted"]};
        font-size: 11px;
        margin-top: 5px;
    }}

    .section-heading {{
        font-size: 21px;
        font-weight: 650;
        color: {C["text"]};
        margin: 28px 0 15px 0;
        letter-spacing: -0.3px;
    }}

    .info-card {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 9px;
        padding: 20px;
    }}

    .stTextArea textarea {{
        background: {C["surface2"]} !important;
        color: {C["text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 8px !important;
        transition: border-color 160ms ease;
    }}

    .stButton > button {{
        border-radius: 8px;
        min-height: 43px;
        font-weight: 600;
        border: 1px solid {C["border"]};
        transition: border-color 160ms ease, opacity 160ms ease, background 160ms ease;
    }}

    div[data-testid="stMetric"] {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 9px;
        padding: 15px;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {C["border"]};
        border-radius: 9px;
        overflow: hidden;
    }}

    hr {{
        border-color: {C["border"]};
    }}

    footer {{
        visibility: hidden;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 9px;
        overflow: hidden;
    }}

    .metric-card,
    .info-card,
    div[data-testid="stMetric"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stDataFrame"] {{
        transition: border-color 160ms ease;
    }}

    .hero,
    .metric-card,
    .info-card,
    div[data-testid="stMetric"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    .result-card,
    .xai-panel {{
        box-shadow: 0 1px 3px var(--sg-card-shadow-color);
    }}

    .metric-card:hover,
    .info-card:hover,
    div[data-testid="stMetric"]:hover,
    div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    div[data-testid="stDataFrame"]:hover {{
        border-color: color-mix(in srgb, {C["text"]} 22%, {C["border"]});
    }}

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        color: {C["muted"]};
    }}

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {C["text"]};
    }}

    .stTextArea textarea:focus {{
        border-color: {C["accent"]} !important;
        outline: 2px solid color-mix(in srgb, {C["accent"]} 45%, transparent);
        outline-offset: 1px;
    }}

    .stButton > button:hover {{
        border-color: {C["accent"]};
    }}

    .stButton > button[kind="primary"] {{
        color: light-dark(#FFFFFF, #0B0B0D) !important;
        border: 1px solid light-dark(#111113, #F4F4F6);
        background: light-dark(#111113, #F4F4F6);
    }}

    .stButton > button[kind="primary"] * {{
        color: light-dark(#FFFFFF, #0B0B0D) !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        opacity: 0.84;
    }}

    .st-key-sg-nav-bar [data-testid="stButton"] > button {{
        min-height: 46px !important;
        height: 46px !important;
        border-radius: 12px !important;
        font-size: 0.96rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid var(--sg-border) !important;
        transition: all 0.18s ease !important;
    }}

    .st-key-sg-nav-bar [data-testid="stButton"] > button[kind="secondary"] {{
        background: rgba(128, 138, 156, 0.10) !important;
        color: var(--sg-text) !important;
    }}

    .st-key-sg-nav-bar [data-testid="stButton"] > button[kind="secondary"]:hover {{
        background: rgba(128, 138, 156, 0.18) !important;
        border-color: var(--sg-border-strong) !important;
        transform: translateY(-1px);
    }}

    .st-key-sg-nav-bar [data-testid="stButton"] > button[kind="primary"] {{
        background: var(--sg-card) !important;
        color: var(--sg-text) !important;
        border: 1px solid var(--sg-border-strong) !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.10) !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }}

    .st-key-sg-nav-bar [data-testid="stButton"] > button[kind="primary"] * {{
        color: var(--sg-text) !important;
    }}

    div[data-testid="stSelectbox"] div[role="group"] {{
        box-sizing: border-box;
        min-height: 46px;
        background: {C["surface2"]};
        border: 1px solid {C["border"]};
        border-radius: 8px;
        cursor: pointer;
        transition: border-color 160ms ease;
    }}

    div[data-testid="stSelectbox"] div[role="group"]:hover {{
        border-color: color-mix(in srgb, {C["text"]} 22%, {C["border"]});
    }}

    div[data-testid="stSelectbox"] div[role="group"]:focus-within {{
        border-color: {C["accent"]};
        outline: 2px solid color-mix(in srgb, {C["accent"]} 45%, transparent);
        outline-offset: 1px;
    }}

    div[data-testid="stSelectbox"] input[role="combobox"] {{
        background: transparent !important;
        color: {C["text"]} !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        caret-color: transparent !important;
        cursor: pointer !important;
    }}

    div[data-testid="stSelectbox"] button[aria-label="Open"] {{
        color: {C["muted"]};
        min-width: 40px;
        background: transparent !important;
        border: 0 !important;
        cursor: pointer;
    }}

    div[data-testid="stAlert"] {{
        border-radius: 8px;
        border: 1px solid {C["border"]};
    }}

    div[data-testid="stPlotlyChart"] {{
        border-radius: 8px;
        overflow: hidden;
    }}

    .result-card {{
        padding: 18px 20px;
        margin: 8px 0 18px;
        border: 1px solid {C["border"]};
        border-radius: 9px;
        background: {C["surface"]};
    }}

    .result-card.is-spam {{
        background: color-mix(in srgb, {C["surface"]} 96%, {C["danger"]} 4%);
        border-color: color-mix(in srgb, {C["danger"]} 38%, {C["border"]});
    }}

    .result-card.is-safe {{
        background: color-mix(in srgb, {C["surface"]} 96%, {C["success"]} 4%);
        border-color: color-mix(in srgb, {C["success"]} 38%, {C["border"]});
    }}

    .result-label {{
        color: {C["text"]};
        font-size: 19px;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 4px;
    }}

    .result-card.is-spam .result-label {{
        color: {C["danger"]};
    }}

    .result-card.is-safe .result-label {{
        color: {C["success"]};
    }}

    .result-description {{
        color: {C["muted"]};
        font-size: 13px;
        margin: 0;
    }}

    .xai-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 12px;
    }}

    .xai-panel {{
        min-width: 0;
        padding: 14px;
        border: 1px solid {C["border"]};
        border-radius: 9px;
        background: {C["surface"]};
    }}

    .xai-panel h4 {{
        margin: 0 0 12px;
        color: {C["text"]};
        font-size: 14px;
        font-weight: 700;
    }}

    .xai-feature {{
        margin-top: 12px;
    }}

    .xai-feature-label {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        color: {C["text"]};
        font-size: 0.95rem;
        font-weight: 500;
        overflow-wrap: anywhere;
    }}

    .xai-feature-value {{
        flex: 0 0 auto;
        color: {C["muted"]};
        font-size: 0.95rem;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
    }}

    .xai-track {{
        height: 6px;
        margin-top: 5px;
        overflow: hidden;
        border-radius: 999px;
        background: {C["surface2"]};
        border: 1px solid {C["border"]};
    }}

    .xai-bar {{
        height: 100%;
        border-radius: inherit;
    }}

    .xai-bar.spam {{
        background: {C["danger"]};
    }}

    .xai-bar.not-spam {{
        background: {C["success"]};
    }}

    .st-key-spam-prediction-metric [data-testid="stMetricValue"] {{
        color: #F87171 !important;
    }}

    .st-key-safe-prediction-metric [data-testid="stMetricValue"] {{
        color: #4ADE80 !important;
    }}

    .st-key-spam-probability-progress [data-testid="stProgressBarTrack"] > div {{
        background-color: #EF4444 !important;
    }}

    .st-key-not-spam-probability-progress [data-testid="stProgressBarTrack"] > div {{
        background-color: #22C55E !important;
    }}

    .xai-empty {{
        margin: 0;
        color: {C["muted"]};
        font-size: 12px;
    }}

    @media (max-width: 900px) {{
        .metric-card {{
            padding: 14px;
        }}

        .metric-label {{
            font-size: 10px;
        }}

        .metric-value {{
            font-size: 25px;
        }}

        .hero {{
            padding: 24px;
        }}
    }}

    @media (max-width: 600px) {{
        .xai-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# LOAD MODEL METRICS
# =====================================================

@st.cache_data
def load_project_data():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return metrics, metadata


try:
    metrics, metadata = load_project_data()
except Exception as error:
    st.error(f"Could not load project analytics: {error}")
    st.stop()

INDEPENDENT_TEST_SPLIT = "grouped_independent_test"
TEST_METRIC_FIELDS = (
    ("accuracy", "Accuracy", "Overall correct classifications"),
    ("precision_spam", "Spam Precision", "Precision for the spam class"),
    ("recall_spam", "Spam Recall", "Recall for the spam class"),
    ("f1_spam", "Spam F1-score", "F1-score for the spam class"),
    ("roc_auc", "ROC-AUC", "Area under the ROC curve"),
    ("average_precision", "Average Precision", "Area under the precision-recall curve"),
)


def is_metric_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def resolve_independent_test_metrics(metrics_data, metadata_data):
    selected_name = metadata_data.get("selected_model") or metadata_data.get(
        "best_model"
    )
    if not isinstance(selected_name, str) or not selected_name:
        return None, None, [
            "Model metadata is missing both 'selected_model' and 'best_model'."
        ]

    selected_results = metrics_data.get(selected_name)
    if not isinstance(selected_results, dict):
        return selected_name, None, [
            f"Metrics for selected model '{selected_name}' are missing."
        ]

    test_results = selected_results.get("final_test")
    if not isinstance(test_results, dict):
        return selected_name, None, [
            f"The selected model '{selected_name}' has no 'final_test' metrics."
        ]
    if test_results.get("evaluation_split") != INDEPENDENT_TEST_SPLIT:
        return selected_name, None, [
            "The selected model's final_test entry is not marked as "
            f"'{INDEPENDENT_TEST_SPLIT}'; it will not be displayed as test data."
        ]

    missing_fields = [
        key
        for key, _, _ in TEST_METRIC_FIELDS
        if not is_metric_number(test_results.get(key))
    ]
    if "confusion_matrix" not in test_results:
        missing_fields.append("confusion_matrix")

    warnings = []
    if missing_fields:
        warnings.append(
            "Independent test results are incomplete or invalid for: "
            + ", ".join(missing_fields)
            + ". Available valid results are shown."
        )
    return selected_name, test_results, warnings


if not isinstance(metrics, dict):
    metrics = {}
    metrics_root_warning = "model_metrics.json must contain a model-to-metrics object."
else:
    metrics_root_warning = None

if not isinstance(metadata, dict):
    metadata = {}
    metadata_root_warning = "model_metadata.json must contain a JSON object."
else:
    metadata_root_warning = None

selected_model_name, independent_test_metrics, analytics_warnings = (
    resolve_independent_test_metrics(metrics, metadata)
)
if metrics_root_warning:
    analytics_warnings.insert(0, metrics_root_warning)
if metadata_root_warning:
    analytics_warnings.insert(0, metadata_root_warning)

best_model_name = selected_model_name or "Unavailable"


# =====================================================
# HEADER AND HORIZONTAL NAVIGATION
# =====================================================

header_left, header_nav = st.columns(
    [2.4, 8.0],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark" aria-hidden="true">
                <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 3.5 26 7v7.1c0 6.3-4.1 11.2-10 14.4-5.9-3.2-10-8.1-10-14.4V7l10-3.5Z"
                        stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M10 10.5h12l-4.5 5v4.1L14.5 22v-6.5l-4.5-5Z"
                        stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M13.5 12.5h5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="brand-copy">
                <div class="brand">ScripSift</div>
                <div class="brand-caption">INTELLIGENT EMAIL SECURITY</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def set_active_page(page_name: str) -> None:
    st.session_state.active_page = page_name


with header_nav:
    with st.container(key="sg-nav-bar"):
        page_names = ("Overview", "Email Scanner", "Analytics", "History", "About")
        nav_columns = st.columns(5, gap="small")
        for column, page_name in zip(nav_columns, page_names):
            with column:
                st.button(
                    page_name,
                    key=f"nav_{page_name.lower().replace(' ', '_')}",
                    type=(
                        "primary"
                        if page_name == st.session_state.active_page
                        else "secondary"
                    ),
                    use_container_width=True,
                    on_click=set_active_page,
                    args=(page_name,),
                )

page = st.session_state.active_page

st.divider()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def hero(title, description):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Inter, sans-serif", size=12, color=C["text"]),
        colorway=[C["accent"], C["muted"], C["success"], C["danger"]],
        title_font=dict(size=16, color=C["text"]),
        margin=dict(l=24, r=24, t=58, b=24),
    )
    fig.update_xaxes(
        gridcolor=C["grid"],
        linecolor=C["border"],
        zerolinecolor=C["grid"],
        tickfont_color=C["muted"],
        title_font_color=C["text"],
    )
    fig.update_yaxes(
        gridcolor=C["grid"],
        linecolor=C["border"],
        zerolinecolor=C["grid"],
        tickfont_color=C["muted"],
        title_font_color=C["text"],
    )
    return fig


def render_feature_panel(title, features, direction):
    if not features:
        return (
            f'<section class="xai-panel"><h4>{html.escape(title)}</h4>'
            '<p class="xai-empty">No non-zero feature contributions in this direction.</p>'
            "</section>"
        )

    max_magnitude = max(abs(item["contribution"]) for item in features)
    rows = []
    for item in features:
        feature_name = html.escape(str(item["feature"]))
        contribution = float(item["contribution"])
        width = max(0.0, min(100.0, abs(contribution) / max_magnitude * 100))
        signed_value = f"{contribution:+.4f}"
        rows.append(
            f'<div class="xai-feature"><div class="xai-feature-label">'
            f"<span>{feature_name}</span>"
            f'<span class="xai-feature-value">{signed_value}</span>'
            f'</div><div class="xai-track"><div class="xai-bar {direction}" '
            f'style="width:{width:.2f}%;"></div></div></div>'
        )

    return (
        f'<section class="xai-panel"><h4>{html.escape(title)}</h4>'
        + "".join(rows)
        + "</section>"
    )


# =====================================================
# OVERVIEW
# =====================================================

if page == "Overview":

    hero(
        "ScripSift Email Security Command Center",
        "Monitor email classification performance and analyze suspicious messages with machine learning.",
    )

    st.markdown(
        '<div class="section-heading">Model Performance</div>',
        unsafe_allow_html=True,
    )

    overview_fields = (
        ("TEST ACCURACY", "accuracy", "Independent grouped test set"),
        ("SPAM PRECISION", "precision_spam", "Precision for spam class"),
        ("SPAM RECALL", "recall_spam", "Recall for spam class"),
        ("SPAM F1-SCORE", "f1_spam", "F1-score for spam class"),
    )
    available_overview_fields = [
        (label, independent_test_metrics[key], note)
        for label, key, note in overview_fields
        if independent_test_metrics is not None
        and is_metric_number(independent_test_metrics.get(key))
    ]
    if available_overview_fields:
        columns = st.columns(len(available_overview_fields))
        for column, (label, value, note) in zip(
            columns, available_overview_fields
        ):
            with column:
                metric_card(label, f"{value * 100:.2f}%", note)
    else:
        st.warning(
            "Independent test metrics are unavailable. Validation metrics are "
            "not shown as test results. "
            + " ".join(analytics_warnings)
        )
    if available_overview_fields and analytics_warnings:
        st.warning(" ".join(analytics_warnings))

    st.markdown(
        '<div class="section-heading">System Overview</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### Selected Machine Learning Model")
            st.info(best_model_name)

            st.markdown("#### Dataset Statistics")

            a, b, d = st.columns(3)
            a.metric("Total samples", f"{metadata['rows_used']:,}")
            b.metric("Training", f"{metadata['train_rows']:,}")
            d.metric("Testing", f"{metadata['test_rows']:,}")

    with right:
        with st.container(border=True):
            distribution = pd.DataFrame({
                "Category": ["Not Spam", "Spam"],
                "Messages": [
                    metadata["class_distribution"]["not_spam"],
                    metadata["class_distribution"]["spam"],
                ],
            })

            fig = px.pie(
                distribution,
                names="Category",
                values="Messages",
                hole=0.68,
                title="Email Dataset Distribution",
                color="Category",
                color_discrete_map={
                    "Not Spam": C["success"],
                    "Spam": C["danger"],
                },
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
                theme="streamlit",
            )

    st.markdown(
        '<div class="section-heading">Recent Scans</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        if st.session_state.history:
            st.dataframe(
                pd.DataFrame(st.session_state.history),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No scans yet. Open Email Scanner to analyze your first message.")


# =====================================================
# EMAIL SCANNER
# =====================================================

elif page == "Email Scanner":

    hero(
        "Intelligent Email Scanner",
        "Analyze email content and identify messages that may contain spam.",
    )

    examples = {
        "Choose a sample": "",
        "Suspicious prize message": (
            "CONGRATULATIONS! You have won a cash prize. "
            "Click this link immediately to claim your reward. "
            "Send your bank details to verify your identity."
        ),
        "Normal business email": (
            "Hello team, please find attached the meeting notes "
            "and action items from today's project discussion."
        ),
    }

    with st.container(border=True):
        st.markdown("#### Enter Email Content")
        selected = st.selectbox(
            "Try a sample message",
            list(examples),
            accept_new_options=False,
        )

        email_text = st.text_area(
            "Email message",
            value=examples[selected],
            height=230,
            placeholder="Paste the complete email message here...",
        )

        scan_requested = st.button(
            "Scan Email",
            type="primary",
            use_container_width=True,
        )

    if scan_requested:
        validation_error = None
        try:
            validate_email_text(email_text)
        except ValueError as error:
            validation_error = str(error)

        if validation_error:
            st.warning(validation_error)
        elif not MODEL_PATH.exists():
            st.error("Trained model not found. Run python train_model.py first.")

        else:
            try:
                with st.spinner("Analyzing email..."):
                    details = predict_email_details(email_text)

                with st.container(border=True):
                    st.markdown(
                        '<div class="section-heading">Analysis Results</div>',
                        unsafe_allow_html=True,
                    )

                    is_spam = details["label"] == "Spam"
                    result_class = "is-spam" if is_spam else "is-safe"
                    result_message = (
                        "This message was classified as spam."
                        if is_spam
                        else "This message was classified as not spam."
                    )
                    st.markdown(
                        f"""
                        <div class="result-card {result_class}">
                            <div class="result-label">{details['label'].upper()}</div>
                            <p class="result-description">{result_message}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    a, b, d = st.columns(3)
                    prediction_metric_key = (
                        "spam-prediction-metric"
                        if is_spam
                        else "safe-prediction-metric"
                    )
                    with a.container(key=prediction_metric_key):
                        st.metric("Prediction", details["label"])
                    b.metric("Model Confidence", f"{details['confidence']:.2f}%")
                    d.metric("Spam Probability", f"{details['spam_probability']:.2f}%")

                    st.markdown("#### Class Probabilities")
                    st.write("Spam probability")
                    with st.container(key="spam-probability-progress"):
                        st.progress(details["spam_probability"] / 100)

                    st.write("Not Spam probability")
                    with st.container(key="not-spam-probability-progress"):
                        st.progress(details["not_spam_probability"] / 100)

                    st.caption(
                        "These are model-estimated probabilities, not guaranteed real-world risk scores."
                    )

                with st.container(border=True):
                    st.markdown("### Why was this email flagged?")
                    if details.get("explanation_available"):
                        spam_panel = render_feature_panel(
                            "Evidence toward Spam",
                            details.get("spam_features", []),
                            "spam",
                        )
                        not_spam_panel = render_feature_panel(
                            "Evidence toward Not Spam",
                            details.get("not_spam_features", []),
                            "not-spam",
                        )
                        st.markdown(
                            f'<div class="xai-grid">{spam_panel}{not_spam_panel}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "These are model-estimated feature contributions "
                            "from the email's vocabulary terms. They are not "
                            "proof that an email is malicious or safe."
                        )
                    else:
                        st.info(
                            details.get(
                                "explanation_message",
                                "An explanation is not available for this message.",
                            )
                        )

                st.session_state.history.insert(
                    0,
                    {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Prediction": details["label"],
                        "Confidence": f"{details['confidence']:.2f}%",
                        "Spam Probability": f"{details['spam_probability']:.2f}%",
                        "Message Preview": email_text[:75],
                    },
                )

                st.session_state.history = st.session_state.history[:20]

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# =====================================================
# ANALYTICS
# =====================================================

elif page == "Analytics":

    hero(
        "Machine Learning Analytics",
        "Review independent test performance and validation-only model comparisons.",
    )

    with st.container(border=True):
        st.markdown("### Evaluation Method")
        st.write(
            "Candidates are fitted on training data and selected using "
            "validation spam F1. The selected pipeline is refitted on training "
            "plus validation data, then evaluated on the grouped independent test set."
        )
        split_strategy = metadata.get("split_strategy")
        grouping_method = metadata.get("grouping_method")
        selection_metric = metadata.get(
            "model_selection_metric",
            metadata.get("selection_metric"),
        )
        if all(
            isinstance(value, str) and value
            for value in (split_strategy, grouping_method, selection_metric)
        ):
            st.caption(f"Split strategy: {split_strategy}")
            st.caption(f"Grouping: {grouping_method}")
            st.caption(f"Selection metric: {selection_metric}")
        else:
            st.warning(
                "Evaluation metadata is incomplete; showing the methodology "
                "summary without the missing metadata details."
            )

    st.markdown(
        '<div class="section-heading">Independent Test Results</div>',
        unsafe_allow_html=True,
    )
    if selected_model_name:
        st.caption(f"Selected model: {selected_model_name}")

    if independent_test_metrics is None:
        st.warning(
            "Independent test results are unavailable; validation metrics are "
            "not used as a fallback. "
            + " ".join(analytics_warnings)
        )
    else:
        if analytics_warnings:
            st.warning(" ".join(analytics_warnings))

        test_cards = [
            (label, independent_test_metrics[key], note)
            for key, label, note in TEST_METRIC_FIELDS
            if is_metric_number(independent_test_metrics.get(key))
        ]
        for start in range(0, len(test_cards), 3):
            columns = st.columns(3)
            for column, (label, value, note) in zip(
                columns, test_cards[start : start + 3]
            ):
                with column:
                    metric_card(label, f"{value * 100:.2f}%", note)

        matrix = independent_test_metrics.get("confusion_matrix")
        valid_matrix = (
            isinstance(matrix, list)
            and len(matrix) == 2
            and all(isinstance(row, list) and len(row) == 2 for row in matrix)
            and all(is_metric_number(value) for row in matrix for value in row)
        )
        if valid_matrix:
            with st.container(border=True):
                matrix_df = pd.DataFrame(
                    matrix,
                    index=["Actual Not Spam", "Actual Spam"],
                    columns=["Predicted Not Spam", "Predicted Spam"],
                )
                fig = px.imshow(
                    matrix_df,
                    text_auto=True,
                    color_continuous_scale=[
                        [0, C["surface2"]],
                        [1, C["text"]],
                    ],
                    title="Independent Test Confusion Matrix",
                    aspect="auto",
                    labels={
                        "x": "Predicted label",
                        "y": "Actual label",
                        "color": "Messages",
                    },
                )
                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True,
                    theme="streamlit",
                )
        else:
            st.warning(
                "The independent test confusion matrix is missing or invalid."
            )

        test_report = independent_test_metrics.get("classification_report")
        if isinstance(test_report, dict):
            with st.container(border=True):
                st.markdown("### Independent Test Classification Report")
                st.dataframe(
                    pd.DataFrame(test_report).T.round(4),
                    use_container_width=True,
                )
        else:
            st.warning(
                "The independent test classification report is missing or invalid."
            )

    st.markdown(
        '<div class="section-heading">Validation Model Comparison</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These scores are from the validation split and are used for model "
        "selection; they are not final test results."
    )

    comparison_rows = []
    validation_warnings = []
    for name, result in metrics.items():
        if not isinstance(result, dict):
            validation_warnings.append(
                f"Validation metrics for '{name}' are not a JSON object."
            )
            continue
        if result.get("evaluation_split") != "validation":
            validation_warnings.append(
                f"Skipped '{name}' because it is not explicitly marked as validation."
            )
            continue

        classification_report = result.get("classification_report")
        spam_report = (
            classification_report.get("Spam", {})
            if isinstance(classification_report, dict)
            else {}
        )
        values = {
            "accuracy": result.get("accuracy"),
            "precision_spam": result.get(
                "precision_spam",
                spam_report.get("precision"),
            ),
            "recall_spam": result.get(
                "recall_spam",
                spam_report.get("recall"),
            ),
            "f1_spam": result.get(
                "f1_spam",
                spam_report.get("f1-score"),
            ),
        }
        if not all(is_metric_number(value) for value in values.values()):
            validation_warnings.append(
                f"Skipped '{name}' because one or more validation metrics are missing."
            )
            continue

        comparison_rows.append(
            {
                "Model": name,
                "Accuracy (%)": round(values["accuracy"] * 100, 2),
                "Spam Precision (%)": round(values["precision_spam"] * 100, 2),
                "Spam Recall (%)": round(values["recall_spam"] * 100, 2),
                "Spam F1 (%)": round(values["f1_spam"] * 100, 2),
            }
        )

    if validation_warnings:
        st.warning(" ".join(validation_warnings))

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        with st.container(border=True):
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )
            fig = px.bar(
                comparison_df,
                x="Model",
                y="Accuracy (%)",
                color="Model",
                text="Accuracy (%)",
                title="Validation Accuracy by Model",
            )
            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                cliponaxis=False,
            )
            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
                theme="streamlit",
            )
    else:
        st.warning("No valid validation model comparisons are available.")


# =====================================================
# ABOUT
# =====================================================

elif page == "About":

    hero(
        "About ScripSift",
        "A machine learning project exploring natural language processing and email classification.",
    )

    st.markdown("### How the System Works")

    steps = [
        ("01", "Email Input", "The user provides an email message."),
        ("02", "Text Processing", "The message is processed by the model pipeline."),
        ("03", "Feature Extraction", "CountVectorizer converts text into numerical features."),
        ("04", "Classification", "Multinomial Naive Bayes predicts the email category."),
        ("05", "Prediction", "The application displays the predicted label and probabilities."),
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom:12px;">
                <span style="color:{C['accent']};font-weight:800;">{number}</span>
                <strong style="margin-left:14px;">{title}</strong>
                <p style="color:{C['muted']};margin:8px 0 0 36px;">{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown("### Technology Stack")
        st.write(
            "Python · Pandas · Scikit-learn · "
            "Multinomial Naive Bayes · Streamlit · Plotly · Joblib"
        )

    with st.container(border=True):
        st.markdown("### Dataset Summary")

        a, b, d = st.columns(3)
        a.metric("Total records", f"{metadata['rows_used']:,}")
        b.metric("Training records", f"{metadata['train_rows']:,}")
        d.metric("Testing records", f"{metadata['test_rows']:,}")

    with st.container(border=True):
        st.markdown("### Project Limitations")
        st.warning(
            "This educational project analyzes email text only. "
            "It does not inspect sender identity, attachments, URLs or email headers. "
            "It should not be used as the sole security control for real email accounts."
        )


# =====================================================
# PREDICTION HISTORY
# =====================================================

elif page == "History":

    hero(
        "Prediction History",
        "Review email scans completed during this application session.",
    )

    with st.container(border=True):
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            spam_scans = int((history_df["Prediction"] == "Spam").sum())
            not_spam_scans = int((history_df["Prediction"] == "Not Spam").sum())

            total_column, spam_column, safe_column = st.columns(3)
            total_column.metric("Scans this session", f"{len(history_df):,}")
            spam_column.metric("Spam flagged", f"{spam_scans:,}")
            safe_column.metric("Not spam", f"{not_spam_scans:,}")

            st.markdown("#### Recent email analyses")
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No scans in this session yet. Open Email Scanner to analyze a message."
            )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    f"""
    <div style="text-align:center;color:{C['muted']};font-size:12px;padding:12px;">
        <strong>ScripSift</strong> · Intelligent Email Security
    </div>
    """,
    unsafe_allow_html=True,
)
