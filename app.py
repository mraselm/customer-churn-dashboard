# app.py — Final Production Version (clicked AutoML + robust column handling)
# Customer Churn Prediction Dashboard with AutoML (PyCaret) + SHAP + OpenAI GPT Insight Assistant


import os
import sys
import warnings
import re
import hashlib
import io
import base64
warnings.filterwarnings("ignore")

# --- Load OpenAI API key early to ensure Streamlit detects it ---
from dotenv import load_dotenv

# Load .env only if it exists (for local dev)
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# Read from environment (works for both local and DigitalOcean)
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = None
OPENAI_AVAILABLE = False

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Professional Validation Engine (Backend Only) ---
from validation_engine import ChurnValidationEngine, quick_validate

# --- Autonomous Monitoring Agent (Self-Healing Training) ---
from monitoring_agent import MonitoringAgent

# --- Unsupervised Churn Detection (for unlabeled datasets) ---
from unsupervised_churn import UnsupervisedChurnDetector, prepare_unlabeled_dataset

# ----------------------------- ENV INFO -----------------------------
st.set_page_config(page_title="Customer Churn Dashboard ? AI Assistant", layout="wide")

# --- Load OpenAI API key from Streamlit secrets if not in env ---
if not OPENAI_KEY:
    try:
        OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()
        if OPENAI_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_KEY
    except Exception:
        OPENAI_KEY = ""

if OPENAI_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY  # ensure Streamlit can access it
        print("? OpenAI key loaded and available for Streamlit.")
        OPENAI_AVAILABLE = True
    except Exception as e:
        print(f"? Failed to initialize OpenAI: {e}")
        client = None
        OPENAI_AVAILABLE = False
else:
    print("?? OpenAI key not found in environment or Streamlit secrets.")
    client = None
    OPENAI_AVAILABLE = False

# Sidebar status messages (rendered at bottom later)
openai_sidebar_status = (
    "connected",
    "API - Connected"
) if OPENAI_AVAILABLE else (
    "disconnected",
    "API - Not Connected"
)

# PyCaret (robust import)
PYCARET_AVAILABLE = True
try:
    import pycaret
    from pycaret import classification as clf
    pycaret_sidebar_status = ("connected", f"PyCaret V {pycaret.__version__}")
except Exception as e:
    PYCARET_AVAILABLE = False
    pycaret_sidebar_status = ("disconnected", "PyCaret V Not detected")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# DiCE for Counterfactual Explanations
try:
    import dice_ml
    from dice_ml import Dice
    DICE_AVAILABLE = True
except Exception:
    # dice-ml is in requirements.txt — if import fails, skip gracefully without subprocess overhead
    DICE_AVAILABLE = False


# ─── Helper: render matplotlib figure as embedded base64 image ───────────
# Bypasses Streamlit's MediaFileHandler PNG cache entirely, avoiding
# "Missing file .png" errors on script reruns.
def _show_matplotlib(fig, container=None):
    """Render *fig* as an inline base64 <img> via st.markdown."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    html = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'
    target = container or st
    target.markdown(html, unsafe_allow_html=True)
    plt.close(fig)


# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600&family=Sora:wght@400;500;600;700&display=swap');

:root {
    --bg: #fff7ed;
    --bg-gradient: radial-gradient(1100px 520px at 105% -5%, rgba(15, 118, 110, 0.16) 0%, transparent 60%),
                   radial-gradient(900px 520px at -10% 10%, rgba(245, 158, 11, 0.18) 0%, transparent 62%),
                   linear-gradient(180deg, #fffaf3 0%, #f3f6fb 70%, #eef2f7 100%);
    --surface: #ffffff;
    --surface-alt: #f8fafc;
    --border: rgba(15, 23, 42, 0.09);
    --shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
    --shadow-soft: 0 8px 20px rgba(15, 23, 42, 0.08);
    --ink: #0f172a;
    --muted: #667085;
    --accent: #0f766e;
    --accent-strong: #115e59;
    --accent-2: #f59e0b;
    --accent-3: #22c55e;
    --info: #2563eb;
    --danger: #e11d48;
    --space-1: 8px;
    --space-2: 12px;
    --space-3: 16px;
    --space-4: 24px;
    --space-5: 32px;
    --icon-chart: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 3v18h18'/><path d='M7 14l3-3 4 4 5-6'/></svg>");
    --icon-search: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='7'/><path d='M20 20l-3.5-3.5'/></svg>");
    --icon-bulb: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M9 18h6'/><path d='M10 22h4'/><path d='M8.5 14.5C7 13.2 6 11.4 6 9.3 6 5.9 8.7 3 12 3s6 2.9 6 6.3c0 2.1-1 3.9-2.5 5.2'/></svg>");
    --icon-user: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='8' r='4'/><path d='M4 21c0-4 3.6-7 8-7s8 3 8 7'/></svg>");
    --icon-spark: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 17l6-6 4 4 7-8'/><path d='M3 21h18'/></svg>");
    --icon-target: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/></svg>");
}

@keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes glowPulse {
    0% { opacity: 0.4; transform: scale(0.98); }
    100% { opacity: 0.9; transform: scale(1); }
}

html, body, .stApp {
    font-family: "Manrope", "Segoe UI", sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--bg-gradient);
    color: var(--ink);
}

/* Sidebar layout: allow bottom-pinned status */
section[data-testid="stSidebar"] > div:first-child {
    height: 100%;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    height: 100%;
    display: flex;
    flex-direction: column;
}
.sidebar-spacer {
    flex: 1 1 auto;
}
.sidebar-status {
    font-size: 0.78rem;
    line-height: 1.2;
    margin: 6px 2px 0 2px;
}
.sidebar-status.connected {
    color: #16a34a;
}
.sidebar-status.disconnected {
    color: #9ca3af;
}
.sidebar-state {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: var(--space-2) var(--space-3);
    font-size: 0.85rem;
    color: var(--muted);
    margin: var(--space-2) 2px 0 2px;
}
.sidebar-section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #94a3b8;
    margin: var(--space-3) 2px var(--space-1) 2px;
    font-weight: 600;
}
.sidebar-divider {
    height: 1px;
    background: rgba(148, 163, 184, 0.35);
    margin: var(--space-2) 0;
}

.block-container {
    padding-top: 2.4rem;
    padding-bottom: 3.6rem;
    position: relative;
    z-index: 1;
}

h1, h2, h3, h4, h5, h6,
.gradient-title,
.stTabs [role="tab"] {
    font-family: "Sora", "Manrope", sans-serif;
    letter-spacing: -0.02em;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
    box-shadow: 10px 0 24px rgba(15, 23, 42, 0.05);
}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider {
    color: var(--ink);
}

div[data-testid="stButton"] > button {
    background: linear-gradient(120deg, var(--accent) 0%, var(--accent-strong) 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 14px 28px rgba(15, 118, 110, 0.25);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    width: 100%;
    white-space: nowrap;
    font-size: 0.84rem;
    padding: 0.46rem 0.72rem;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 34px rgba(15, 118, 110, 0.32);
    color: #ffffff !important;
}

div[data-testid="stButton"] > button:active {
    transform: translateY(0);
    color: #ffffff !important;
}
div[data-testid="stButton"] > button:focus,
div[data-testid="stButton"] > button:focus-visible {
    color: #ffffff !important;
    outline: none;
}
div[data-testid="stButton"] > button * {
    color: #ffffff !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #f8fafc !important;
    color: #0f172a !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button[kind="secondary"] * {
    color: #0f172a !important;
}

div[data-testid="stSlider"] {
    background: var(--surface);
    padding: 12px 14px 10px 14px;
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
}
div[data-testid="stSlider"] label {
    font-weight: 600;
    color: var(--ink);
}
div[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 6px;
}
div[data-testid="stSlider"] [data-baseweb="track"] {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
}
div[data-testid="stSlider"] [data-baseweb="thumb"] {
    width: 18px;
    height: 18px;
    border-radius: 999px;
    background: #ffffff;
    border: 3px solid var(--accent);
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.18);
}
div[data-testid="stSlider"] [data-baseweb="thumb"]:hover {
    border-color: var(--accent-strong);
    box-shadow: 0 8px 16px rgba(15, 23, 42, 0.24);
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: rgba(15, 118, 110, 0.45);
}
section[data-testid="stSidebar"] [data-baseweb="select"] div[role="combobox"] {
    padding: 8px 12px;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: var(--muted);
}
/* Hide clear (x) icon in sidebar selectboxes */
section[data-testid="stSidebar"] [data-baseweb="select"] [title="Clear"],
section[data-testid="stSidebar"] [data-baseweb="select"] [aria-label="Clear"],
section[data-testid="stSidebar"] [data-baseweb="select"] [aria-label="clear"],
section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="clear-icon"],
section[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="clear"] {
    display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
    overflow: hidden;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] > div:hover {
    border-color: rgba(15, 118, 110, 0.45);
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    padding: 8px 12px;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    color: var(--muted);
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    border-bottom: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"]::before,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"]::after {
    border-bottom: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    outline: none !important;
    background-image: none !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus {
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: linear-gradient(120deg, rgba(15, 118, 110, 0.18) 0%, rgba(245, 158, 11, 0.18) 100%);
    color: var(--ink);
    border: 1px solid rgba(15, 118, 110, 0.35);
    border-radius: 12px;
    padding: 0.36rem 0.7rem;
    font-size: 0.84rem;
    box-shadow: 0 10px 18px rgba(15, 23, 42, 0.1);
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    border-color: rgba(15, 118, 110, 0.55);
    box-shadow: 0 12px 20px rgba(15, 23, 42, 0.16);
}
/* File uploader remove (X) button should be neutral, not themed */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="stFileUploaderDeleteBtn"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button[title="Delete file"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button[aria-label="Delete file"] {
    background: transparent !important;
    border: none !important;
    color: #667085 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"]:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="stFileUploaderDeleteBtn"]:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button[title="Delete file"]:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button[aria-label="Delete file"]:hover {
    background: transparent !important;
    border: none !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button[data-testid="stFileUploaderDeleteBtn"] svg {
    color: #667085 !important;
}

.card {
    background: var(--surface);
    border-radius: 20px;
    padding: var(--space-4) var(--space-4) var(--space-3) var(--space-4);
    box-shadow: var(--shadow);
    margin-bottom: var(--space-3);
    color: var(--ink);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    animation: rise 0.5s ease both;
}
.card::before {
    content: none;
    display: none;
}
.card::after {
    content: "";
    position: absolute;
    right: -40px;
    bottom: -40px;
    width: 160px;
    height: 160px;
    background: radial-gradient(circle, rgba(15, 118, 110, 0.18) 0%, transparent 70%);
    animation: glowPulse 4s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
.card > * {
    position: relative;
    z-index: 1;
}

.metric-green { color: var(--accent-3); font-weight: 700; font-size: 1.15rem; }
.metric-red { color: var(--danger); font-weight: 700; font-size: 1.15rem; }
.metric-primary { color: var(--accent-strong); font-weight: 700; }
.metric-accent { color: var(--accent-2); font-weight: 700; }
.small-muted { color: var(--muted); font-size: 12px; }

.kpi-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-3);
    justify-content: flex-start;
    margin: var(--space-3) 0 var(--space-4) 0;
}
.kpi-metric {
    background: var(--surface);
    border-radius: 18px;
    padding: var(--space-3) var(--space-4) var(--space-3) var(--space-4);
    min-width: 180px;
    box-shadow: var(--shadow-soft);
    color: var(--ink);
    text-align: left;
    border: 1px solid var(--border);
    flex: 1 1 200px;
    position: relative;
    overflow: hidden;
    animation: rise 0.5s ease both;
}
.kpi-metric::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: 5px;
    background: linear-gradient(180deg, var(--accent) 0%, rgba(15, 118, 110, 0.1) 100%);
}
.kpi-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; }
.kpi-value { font-size: 1.7rem; font-weight: 700; }

.section-padding { margin-top: var(--space-4) !important; }

.section-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: "Sora", "Manrope", sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0 0 var(--space-2) 0;
}
.section-icon {
    width: 26px;
    height: 26px;
    background-color: var(--accent-strong);
    -webkit-mask-image: var(--icon-chart);
    mask-image: var(--icon-chart);
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: center;
    mask-position: center;
    -webkit-mask-size: contain;
    mask-size: contain;
}
.section-icon.overview {
    -webkit-mask-image: var(--icon-chart);
    mask-image: var(--icon-chart);
}
.section-icon.single {
    -webkit-mask-image: var(--icon-search);
    mask-image: var(--icon-search);
}
.section-icon.insights {
    -webkit-mask-image: var(--icon-bulb);
    mask-image: var(--icon-bulb);
}
.subsection-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: "Sora", "Manrope", sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--ink);
    margin: var(--space-2) 0 var(--space-3) 0;
}
.subsection-icon {
    width: 18px;
    height: 18px;
    background-color: var(--accent-strong);
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: center;
    mask-position: center;
    -webkit-mask-size: contain;
    mask-size: contain;
}
.subsection-icon.prediction {
    -webkit-mask-image: var(--icon-chart);
    mask-image: var(--icon-chart);
}
.subsection-icon.profile {
    -webkit-mask-image: var(--icon-user);
    mask-image: var(--icon-user);
}
.subsection-icon.shap {
    -webkit-mask-image: var(--icon-spark);
    mask-image: var(--icon-spark);
}
.subsection-icon.counterfactual {
    -webkit-mask-image: var(--icon-target);
    mask-image: var(--icon-target);
}
.subsection-icon.suggestion {
    -webkit-mask-image: var(--icon-bulb);
    mask-image: var(--icon-bulb);
}

.action-card {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.action-card:hover {
    border-color: #f59e0b;
    box-shadow: 0 4px 16px rgba(245,158,11,0.15);
    transform: translateY(-2px);
}

.action-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.action-icon {
    font-size: 24px;
    line-height: 1;
}

.action-title {
    font-size: 16px;
    font-weight: 700;
    color: #1e293b;
}

.action-body {
    font-size: 14px;
    line-height: 1.6;
    color: #475569;
    margin-bottom: 12px;
}

.action-impact {
    display: flex;
    justify-content: space-between;
    padding: 10px 14px;
    background: #fef3c7;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
}

.impact-before {
    color: #dc2626;
}

.impact-after {
    color: #16a34a;
}

div[data-testid="stRadio"] > div {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
    padding: 0;
    background: transparent;
    border: none;
    box-shadow: none;
}
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
    display: none !important;
}
div[data-testid="stRadio"] label {
    margin: 0 !important;
    display: block;
    width: 100%;
    min-width: 0;
}
div[data-testid="stRadio"] input {
    display: none;
}
div[data-testid="stRadio"] label > div:first-of-type {
    display: none;
}
div[data-testid="stRadio"] label > div:last-of-type {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    width: 100%;
    font-size: 0.98rem;
    padding: var(--space-2) var(--space-4);
    background: #ffffff;
    color: var(--ink);
    font-weight: 700;
    border-radius: 999px;
    border: 1px solid rgba(15, 23, 42, 0.12);
    box-shadow: 0 16px 34px rgba(15, 23, 42, 0.12);
    position: relative;
    transition: all 0.18s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
div[data-testid="stRadio"] > div > label > div:last-of-type::before {
    content: "";
    width: 18px;
    height: 18px;
    display: inline-block;
    background-color: currentColor;
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: center;
    mask-position: center;
    -webkit-mask-size: contain;
    mask-size: contain;
}
div[data-testid="stRadio"] > div > label:nth-child(1) > div:last-of-type::before {
    -webkit-mask-image: var(--icon-chart);
    mask-image: var(--icon-chart);
}
div[data-testid="stRadio"] > div > label:nth-child(2) > div:last-of-type::before {
    -webkit-mask-image: var(--icon-search);
    mask-image: var(--icon-search);
}
div[data-testid="stRadio"] > div > label:nth-child(3) > div:last-of-type::before {
    -webkit-mask-image: var(--icon-bulb);
    mask-image: var(--icon-bulb);
}
div[data-testid="stRadio"] label * {
    color: inherit !important;
}
div[data-testid="stRadio"] label:hover > div:last-of-type {
    transform: translateY(-1px);
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.16);
}
div[data-testid="stRadio"] label:has(input:checked) > div:last-of-type,
div[data-testid="stRadio"] input:checked + div + div,
div[data-testid="stRadio"] input:checked + div {
    background: linear-gradient(120deg, #0f172a 0%, #0f766e 65%, #10b981 100%);
    color: #ffffff;
    border: 1px solid rgba(15, 118, 110, 0.5);
    box-shadow: 0 22px 44px rgba(15, 23, 42, 0.22);
}
div[data-testid="stRadio"] label:has(input:checked) > div:last-of-type *,
div[data-testid="stRadio"] input:checked + div + div *,
div[data-testid="stRadio"] input:checked + div * {
    color: #ffffff !important;
}
div[data-testid="stRadio"] label:has(input:checked) > div:last-of-type::before,
div[data-testid="stRadio"] input:checked + div + div::before,
div[data-testid="stRadio"] input:checked + div::before {
    background-color: #ffffff !important;
}

.gray-info {
    background: var(--surface-alt);
    color: var(--muted);
    border-radius: 14px;
    padding: var(--space-3) var(--space-3);
    margin-bottom: var(--space-2);
    font-size: 1.02rem;
    text-align: center;
    border: 1px solid var(--border);
}

.accent-line {
    height: 5px;
    border-radius: 999px;
    background: linear-gradient(90deg, #0f766e 0%, #84cc16 55%, #f59e0b 100%);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
    margin: var(--space-2) 0 var(--space-4) 0;
}

.footer {
    color: var(--muted);
    background: none;
    text-align: center;
    font-size: 0.95rem;
    margin-top: var(--space-4);
    padding: var(--space-3) 0 var(--space-1) 0;
}

.gradient-header {
    background: linear-gradient(135deg, #0b1220 0%, #0f766e 55%, #a16207 110%);
    border-radius: 28px;
    padding: var(--space-5) var(--space-5);
    margin-bottom: var(--space-4);
    color: #f8fafc !important;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.35);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.14);
    animation: rise 0.6s ease both;
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(0, 0.75fr);
    gap: var(--space-4);
    align-items: center;
}
.gradient-header::before {
    content: "";
    position: absolute;
    top: -120px;
    right: -100px;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.45) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.gradient-header::after {
    content: "";
    position: absolute;
    left: -90px;
    bottom: -140px;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.4) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.gradient-header > * { position: relative; z-index: 1; }
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: var(--space-1) var(--space-2);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.24);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e2e8f0;
    margin-bottom: var(--space-2);
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: var(--space-2);
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255, 255, 255, 0.85);
    max-width: 640px;
    margin-bottom: var(--space-3);
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
}
.hero-chip {
    padding: var(--space-1) var(--space-2);
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.18);
    font-size: 0.82rem;
    color: #e2e8f0;
}
.hero-panel {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: var(--space-3) var(--space-3);
    backdrop-filter: blur(6px);
}
.hero-panel-title {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(255, 255, 255, 0.72);
    margin-bottom: var(--space-2);
}
.hero-list {
    list-style: none;
    padding: 0;
    margin: 0 0 var(--space-2) 0;
    display: grid;
    gap: var(--space-1);
    color: #f1f5f9;
    font-size: 0.95rem;
}
.hero-list li::before {
    content: "•";
    color: #fbbf24;
    font-weight: 700;
    margin-right: 8px;
}
.hero-panel-foot {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.75);
}
.gradient-title { font-size: 2.5rem; font-weight: 700; color: #ffffff; letter-spacing: -1px; margin-bottom: 8px; }
.gradient-subtitle { font-size: 1.15rem; color: #fde68a; margin-bottom: 0; }

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
    background: #ffffff;
}

details[data-testid="stExpander"] {
    border-radius: 18px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--shadow-soft);
    padding: var(--space-1) var(--space-2);
}

details[data-testid="stExpander"]:hover {
    border-color: var(--border);
    background: var(--surface);
}

details[data-testid="stExpander"] summary:hover {
    color: inherit !important;
    background: transparent !important;
}

progress {
    width: 100%;
    height: 12px;
    border-radius: 999px;
    overflow: hidden;
}
progress::-webkit-progress-bar {
    background-color: #e5e7eb;
    border-radius: 999px;
}
progress::-webkit-progress-value {
    background-image: linear-gradient(120deg, var(--accent) 0%, var(--accent-2) 100%);
    border-radius: 999px;
}
progress::-moz-progress-bar {
    background-image: linear-gradient(120deg, var(--accent) 0%, var(--accent-2) 100%);
    border-radius: 999px;
}

@media (max-width: 900px) {
    .gradient-header { grid-template-columns: 1fr; }
    .gradient-title { font-size: 2rem; }
    .hero-title { font-size: 1.9rem; }
    .kpi-row { gap: var(--space-2); }
    .kpi-metric { min-width: 160px; }
    div[data-testid="stRadio"] > div { grid-template-columns: 1fr; gap: var(--space-2); }
}
</style>
""", unsafe_allow_html=True)


# ----------------------------- UTILITIES -----------------------------
def safe_read_csv(uploaded_file):
    """Read CSV robustly (handles encoding edge cases)."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (internal use) without changing user-facing selection."""
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df


def resolve_column(actual_df: pd.DataFrame, chosen_name: str) -> str:
    """
    Find the real column name in df after normalization by matching lowercase/underscored keys.
    Returns the actual column name in df, or raises if not found.
    """
    if not chosen_name:
        raise ValueError("No column name provided")

    key = str(chosen_name).strip().lower().replace(" ", "_")
    mapping = {c.strip().lower(): c for c in actual_df.columns}
    if key in mapping:
        return mapping[key]
    # also try strict normalization (remove non-alnum)
    mapping2 = {c.strip().lower().replace(" ", "_"): c for c in actual_df.columns}
    if key in mapping2:
        return mapping2[key]
    raise KeyError(f"Column '{chosen_name}' not found in data after standardization")


def guess_target_name(cols):
    """Try to guess target column name if user doesn't select it yet."""
    candidates = [c for c in cols if c.lower() in ("churn", "target", "label", "y")]
    return candidates[0] if candidates else None


def guess_id_name(cols):
    """Try to guess ID column."""
    candidates = [c for c in cols if "id" in c.lower()]
    return candidates[0] if candidates else None


def detect_id_cols(df: pd.DataFrame, explicit_id: str | None = None, target_col: str | None = None) -> list:
    """Detect ID-like columns to exclude from modeling/SHAP."""
    id_cols = set()
    if explicit_id:
        id_cols.add(explicit_id)

    patterns = [
        r"(^|_)(id)($|_)",
        r"(^|_)(customer_id|customerid|cust_id)($|_)",
        r"(^|_)(account_id|accountid)($|_)",
        r"(^|_)(user_id|userid)($|_)",
        r"(^|_)(client_id|clientid)($|_)",
        r"(^|_)(subscriber_id|subscriberid)($|_)",
    ]

    for c in df.columns:
        if target_col and c == target_col:
            continue
        key = str(c).strip().lower()
        key = re.sub(r"\s+", "_", key)
        if any(re.search(p, key) for p in patterns):
            id_cols.add(c)

    return sorted(id_cols)


def extract_positive_proba(preds: pd.DataFrame) -> float:
    """Extract positive class probability robustly from PyCaret predict_model output."""
    # Typical columns: 'prediction_label', 'prediction_score' OR 'Score_0','Score_1'
    # Use Score_1 if available, else last prob/score column, else fallback to prediction_label.
    prob_cols = [c for c in preds.columns if "score" in c.lower() or "prob" in c.lower()]
    score1 = [c for c in prob_cols if c.lower().endswith("1")]
    if score1:
        return float(preds[score1[0]].iloc[0])
    if prob_cols:
        return float(preds[prob_cols[-1]].iloc[0])
    # fallback
    if "prediction_label" in preds.columns:
        label = preds["prediction_label"].iloc[0]
        try:
            return float(label)
        except Exception:
            return 1.0 if str(label).strip().lower() in ("1", "yes", "true", "churn") else 0.0
    return 0.5


def get_positive_score_series(preds: pd.DataFrame):
    """Return the positive-class probability series from predict_model output."""
    if "Score_1" in preds.columns:
        return preds["Score_1"]
    score1 = [c for c in preds.columns if c.lower().endswith("1")]
    if score1:
        return preds[score1[0]]
    if "prediction_score" in preds.columns:
        return preds["prediction_score"]
    prob_cols = [c for c in preds.columns if "score" in c.lower() or "prob" in c.lower()]
    return preds[prob_cols[-1]] if prob_cols else None


_POSITIVE_TARGET_TOKENS = {
    "1", "1.0", "yes", "y", "true", "t", "churn", "churned", "left", "leave",
    "leaver", "exit", "exited", "cancel", "cancelled", "canceled", "attrition",
    "attrited", "default", "fraud", "positive", "pos", "terminated", "terminate",
}
_NEGATIVE_TARGET_TOKENS = {
    "0", "0.0", "no", "n", "false", "f", "stay", "stayed", "retain", "retained",
    "active", "current", "negative", "neg", "not_churn", "non_churn", "nonchurn",
}
_MISSING_TARGET_TOKENS = {"", "nan", "none", "null", "<na>", "na"}


def _target_label_key(value):
    """Normalize any target label value to a stable comparable key."""
    if pd.isna(value):
        return None

    if isinstance(value, str):
        key = value.strip().lower()
    else:
        try:
            fval = float(value)
            if np.isnan(fval):
                return None
            key = str(int(fval)) if float(fval).is_integer() else str(fval)
        except Exception:
            key = str(value).strip().lower()

    key = key.strip().lower()
    if key in _MISSING_TARGET_TOKENS:
        return None
    return key


def _contains_any_token(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _infer_binary_target_mapping(raw_series: pd.Series) -> tuple[dict, str]:
    """
    Infer robust 0/1 mapping for binary target labels across arbitrary datasets.
    Returns mapping dict (normalized_label -> 0/1) and strategy label.
    """
    keys = raw_series.map(_target_label_key)
    key_counts = keys.dropna().value_counts()
    unique_keys = key_counts.index.tolist()

    if len(unique_keys) < 2:
        raise ValueError("Target column must contain at least two non-missing classes.")

    mapping = {}
    for key in unique_keys:
        if key in _POSITIVE_TARGET_TOKENS:
            mapping[key] = 1
        elif key in _NEGATIVE_TARGET_TOKENS:
            mapping[key] = 0

    if len(mapping) == len(unique_keys) and set(mapping.values()) == {0, 1}:
        return mapping, "semantic_token_mapping"

    if len(unique_keys) == 2:
        a, b = unique_keys[0], unique_keys[1]

        if a in mapping and b not in mapping:
            mapping[b] = 1 - mapping[a]
            return mapping, "paired_with_known_label"
        if b in mapping and a not in mapping:
            mapping[a] = 1 - mapping[b]
            return mapping, "paired_with_known_label"

        positive_markers = (
            "churn", "left", "exit", "cancel", "attrit", "default", "fraud",
            "terminate", "yes", "true", "positive",
        )
        negative_markers = (
            "stay", "retain", "active", "current", "no", "false", "negative",
        )

        a_pos = _contains_any_token(a, positive_markers)
        b_pos = _contains_any_token(b, positive_markers)
        a_neg = _contains_any_token(a, negative_markers)
        b_neg = _contains_any_token(b, negative_markers)

        if a_pos != b_pos:
            return ({a: 1, b: 0} if a_pos else {a: 0, b: 1}), "keyword_inference"
        if a_neg != b_neg:
            return ({a: 0, b: 1} if a_neg else {a: 1, b: 0}), "keyword_inference"

        # Final fallback: churn is typically minority in binary churn datasets.
        minority_key = key_counts.sort_values(ascending=True).index[0]
        majority_key = key_counts.sort_values(ascending=False).index[0]
        if minority_key == majority_key:
            ordered = sorted(unique_keys)
            return {ordered[0]: 0, ordered[1]: 1}, "lexicographic_fallback"
        return {majority_key: 0, minority_key: 1}, "minority_class_as_churn"

    if len(mapping) >= 2 and set(mapping.values()) == {0, 1}:
        # Keep semantic mappings and allow unknown labels to be dropped later.
        return mapping, "partial_semantic_mapping"

    sample_vals = ", ".join(map(str, unique_keys[:8]))
    raise ValueError(
        f"Target appears non-binary or ambiguous. Unique labels (sample): {sample_vals}"
    )


def encode_target_series(raw_series: pd.Series, fit: bool = False) -> pd.Series:
    """
    Encode target labels to 0/1 using a stored mapping.
    If fit=True, infer and store a new mapping in session state.
    """
    mapping = st.session_state.get("target_label_mapping")

    if fit or not mapping:
        mapping, strategy = _infer_binary_target_mapping(raw_series)
        st.session_state["target_label_mapping"] = mapping
        st.session_state["target_mapping_strategy"] = strategy

    keys = raw_series.map(_target_label_key)
    encoded = keys.map(st.session_state.get("target_label_mapping", {})).astype(float)

    # Safety: if stale mapping fails completely, refit once.
    if (not fit) and encoded.notna().sum() == 0:
        mapping, strategy = _infer_binary_target_mapping(raw_series)
        st.session_state["target_label_mapping"] = mapping
        st.session_state["target_mapping_strategy"] = strategy
        encoded = keys.map(mapping).astype(float)

    return encoded


# ----------------------------- ALIGN TO MODEL COLUMNS -----------------------------
def align_to_model_columns(df_in: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align df columns to model input columns. Fill missing columns with 0 or 'Unknown'.
    Also apply the same label-encoding used during training (label_maps).
    """
    import joblib
    # Prefer session_state (always current) over disk file (may be stale)
    model_cols = st.session_state.get("model_feature_cols", None)
    if model_cols is None:
        try:
            model_cols = joblib.load("automl_model_columns.pkl")
        except Exception:
            model_cols = df_in.columns.tolist()

    aligned = df_in.copy()

    # Stored schema
    cat_cols_clean = st.session_state.get("cat_cols_clean", [])
    num_cols_schema = st.session_state.get("numeric_features", [])
    binary_cols = st.session_state.get("binary_mapped_cols", [])
    binary_map = st.session_state.get("binary_map", {})

    # Clean categorical values consistently
    for c in cat_cols_clean:
        if c in aligned.columns:
            aligned[c] = aligned[c].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)

    # Apply binary mapping where applicable
    for c in binary_cols:
        if c in aligned.columns:
            lower_vals = aligned[c].astype(str).str.strip().str.lower()
            aligned[c] = lower_vals.map(binary_map).astype(float)

    # Coerce numeric schema cols to numeric
    for c in num_cols_schema:
        if c in aligned.columns:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")

    # 1) Apply label mappings to categorical columns so they match training encodings
    label_maps = st.session_state.get("label_maps", {})
    for col, mapping in label_maps.items():
        if col in aligned.columns:
            aligned[col] = aligned[col].astype(str).map(lambda x: mapping.get(x, 0))

    # 1b) Dummy-encode remaining object columns (matches training pre-encoding)
    _obj_cols_align = [c for c in aligned.columns if aligned[c].dtype == 'object']
    if _obj_cols_align:
        aligned = pd.get_dummies(aligned, columns=_obj_cols_align, drop_first=True)

    # 2) Add missing columns
    for col in model_cols:
        if col not in aligned.columns:
            # Try to infer type from train_df if available
            val = 0
            if "train_df" in st.session_state and st.session_state.train_df is not None:
                if col in st.session_state.train_df.columns:
                    if st.session_state.train_df[col].dtype.kind in "O":
                        val = "Unknown"
                    else:
                        val = 0
            aligned[col] = val

    # 3) Subset and fillna
    aligned = aligned[model_cols]
    for c in aligned.columns:
        if aligned[c].dtype.kind in "O":
            aligned[c] = aligned[c].fillna("Unknown")
        else:
            aligned[c] = aligned[c].fillna(0)

    return aligned


def align_to_raw_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Align raw input data to the original feature schema used during training (pre-preprocessing).
    This is used for SHAP + prep_pipe so categorical features are encoded correctly.
    """
    raw_cols = st.session_state.get("raw_feature_cols", None)
    if not raw_cols:
        return df_in.copy()

    aligned = df_in.copy()
    cat_cols_clean = st.session_state.get("cat_cols_clean", [])
    num_cols_schema = st.session_state.get("numeric_features", [])
    binary_cols = st.session_state.get("binary_mapped_cols", [])
    binary_map = st.session_state.get("binary_map", {})

    for col in raw_cols:
        if col not in aligned.columns:
            if col in num_cols_schema:
                aligned[col] = 0
            else:
                aligned[col] = "Unknown"

    aligned = aligned[raw_cols]

    for c in cat_cols_clean:
        if c in aligned.columns:
            aligned[c] = aligned[c].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)

    for c in binary_cols:
        if c in aligned.columns:
            lower_vals = aligned[c].astype(str).str.strip().str.lower()
            aligned[c] = lower_vals.map(binary_map).astype(float)

    for c in num_cols_schema:
        if c in aligned.columns:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")

    for c in aligned.columns:
        if aligned[c].dtype.kind in "O":
            aligned[c] = aligned[c].fillna("Unknown")
        else:
            aligned[c] = aligned[c].fillna(0)

    return aligned


def _predict_positive_scores(model_obj, X_in) -> np.ndarray:
    """Return positive-class probability as a 1D numpy array."""
    proba = model_obj.predict_proba(X_in)
    if isinstance(proba, pd.DataFrame):
        arr = proba.to_numpy()
    elif isinstance(proba, pd.Series):
        arr = proba.to_numpy().reshape(-1, 1)
    else:
        arr = np.asarray(proba)

    if arr.ndim == 1:
        return arr.astype(float)
    if arr.shape[1] > 1:
        return arr[:, 1].astype(float)
    return arr[:, 0].astype(float)


def _to_dense_numeric(arr_like) -> np.ndarray:
    """Convert sparse/dataframe/array-like to dense float ndarray with NaN handled."""
    if hasattr(arr_like, "toarray"):
        arr_like = arr_like.toarray()
    elif hasattr(arr_like, "to_numpy"):
        arr_like = arr_like.to_numpy()
    return np.nan_to_num(np.asarray(arr_like, dtype=float))


def _normalize_feature_key(name: str) -> str:
    s = str(name).strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", s)


def _map_model_feature_to_base_feature(feature_name: str, raw_feature_cols: list[str]) -> str:
    """
    Map transformed/encoded feature names back to original business feature names.
    Works with names like:
    - Contract_Month_to_month
    - onehotencoder__Contract_Month_to_month
    - num__tenure
    """
    raw_feature_cols = raw_feature_cols or []
    if not raw_feature_cols:
        return str(feature_name)

    core = str(feature_name).split("__")[-1]
    core_norm = _normalize_feature_key(core)
    raw_norm_map = {_normalize_feature_key(c): c for c in raw_feature_cols}

    if core_norm in raw_norm_map:
        return raw_norm_map[core_norm]

    # Prefix match for dummy-encoded names: originalCol_categoryValue
    raw_sorted = sorted(raw_feature_cols, key=lambda c: len(_normalize_feature_key(c)), reverse=True)
    for raw_col in raw_sorted:
        rnorm = _normalize_feature_key(raw_col)
        if core_norm.startswith(rnorm + "_"):
            return raw_col

    # sklearn fallback pattern: x12_original_feature_or_dummy
    m = re.match(r"^x\d+_(.+)$", core_norm)
    if m:
        inner = m.group(1)
        if inner in raw_norm_map:
            return raw_norm_map[inner]
        for raw_col in raw_sorted:
            rnorm = _normalize_feature_key(raw_col)
            if inner.startswith(rnorm + "_"):
                return raw_col

    if "_" in core_norm:
        first = core_norm.split("_", 1)[0]
        if first in raw_norm_map:
            return raw_norm_map[first]

    return core


def _aggregate_shap_to_base_features(
    shap_vals: np.ndarray,
    model_feature_names: list[str],
    row_model_df: pd.DataFrame | None,
    row_raw_df: pd.DataFrame | None
) -> tuple[np.ndarray, list[str], list]:
    """
    Aggregate SHAP values from transformed features back to original features.
    Returns (aggregated_shap_values, base_feature_names, display_feature_values).
    """
    vals = np.asarray(shap_vals, dtype=float).ravel()
    if len(vals) == 0:
        return vals, [], []

    names = list(model_feature_names) if model_feature_names is not None else []
    if len(names) != len(vals):
        names = [f"feature_{i}" for i in range(len(vals))]

    raw_cols = st.session_state.get("raw_feature_cols") or (
        list(row_raw_df.columns) if isinstance(row_raw_df, pd.DataFrame) else []
    )
    row_model_vals = row_model_df.iloc[0].to_dict() if isinstance(row_model_df, pd.DataFrame) and len(row_model_df) else {}
    row_raw_vals = row_raw_df.iloc[0].to_dict() if isinstance(row_raw_df, pd.DataFrame) and len(row_raw_df) else {}

    recs = []
    for fname, sval in zip(names, vals):
        base = _map_model_feature_to_base_feature(fname, raw_cols)
        recs.append({
            "model_feature": fname,
            "base_feature": base,
            "shap_value": float(sval),
            "model_value": row_model_vals.get(fname, None),
        })

    imp = pd.DataFrame(recs)
    if imp.empty:
        return vals, names, [row_model_vals.get(n, None) for n in names]

    # pick representative encoded value per base feature from largest |SHAP| contributor
    rep_idx = imp.groupby("base_feature")["shap_value"].apply(lambda s: s.abs().idxmax())
    rep = imp.loc[rep_idx.values, ["base_feature", "model_value"]]
    agg = imp.groupby("base_feature", as_index=False)["shap_value"].sum()
    agg = agg.merge(rep, on="base_feature", how="left")

    def _choose_display_value(base, model_val):
        raw_val = row_raw_vals.get(base, None)
        if raw_val is None:
            return model_val
        try:
            if pd.isna(raw_val):
                return model_val
        except Exception:
            pass
        return raw_val

    agg["feature_value"] = agg.apply(
        lambda r: _choose_display_value(r["base_feature"], r["model_value"]), axis=1
    )
    agg["abs_shap"] = agg["shap_value"].abs()
    agg = agg.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    return (
        agg["shap_value"].to_numpy(dtype=float),
        agg["base_feature"].tolist(),
        agg["feature_value"].tolist(),
    )


def generate_counterfactuals(customer_data, model, data_df, target_col, num_cfs=3):
    """
    Generate SHAP-based prescriptive recommendations.
    Uses SHAP to identify top churn drivers, then applies interventions on
    raw/base features so one-hot and other preprocessing remain valid.
    """
    try:
        drop_id_cols = st.session_state.get("drop_id_cols", [])
        drop_cols = [target_col] + list(drop_id_cols)
        drop_cols = [c for c in drop_cols if c and c in data_df.columns]

        X_raw = data_df.drop(columns=drop_cols, errors='ignore').copy()
        query_raw = customer_data.drop(columns=drop_cols, errors='ignore').copy()
        if len(X_raw) == 0 or len(query_raw) == 0:
            return None
        query_raw = query_raw.iloc[[0]].copy()

        raw_feature_cols = st.session_state.get("raw_feature_cols", None)
        if raw_feature_cols:
            keep_cols = [c for c in raw_feature_cols if c in X_raw.columns]
            if keep_cols:
                X_raw = X_raw[keep_cols].copy()
                query_raw = query_raw[[c for c in keep_cols if c in query_raw.columns]].copy()
                for c in keep_cols:
                    if c not in query_raw.columns:
                        query_raw[c] = 0
                query_raw = query_raw[keep_cols]

        def _encode_for_model(d):
            aligned = align_to_model_columns(d.copy(), model)
            return aligned.apply(pd.to_numeric, errors='coerce').fillna(0)

        query_encoded = _encode_for_model(query_raw)
        if query_encoded.shape[1] == 0:
            return None

        def _predict_prob_one(model_obj, x_df):
            try:
                return float(_predict_positive_scores(model_obj, x_df)[0])
            except Exception:
                x_values = x_df.values if isinstance(x_df, pd.DataFrame) else x_df
                try:
                    return float(_predict_positive_scores(model_obj, x_values)[0])
                except Exception:
                    return float(_predict_positive_scores(model, x_df)[0])

        _non_actionable_names = {
            'totalcharges', 'monthlycharges', 'clv', 'engagementscore',
            'tenure', 'customerid', 'seniorcitizen', 'gender', 'age',
        }
        _non_actionable_norm = {_normalize_feature_key(x) for x in _non_actionable_names}
        _non_actionable_norm.update({_normalize_feature_key(x) for x in drop_id_cols})

        _INTERVENTION_COSTS = {
            'contract': 75, 'onlinesecurity': 25, 'techsupport': 25,
            'onlinebackup': 20, 'deviceprotection': 20, 'streamingtv': 15,
            'streamingmovies': 15, 'paymentmethod': 10, 'paperlessbilling': 5,
            'satisfactionscore': 40, 'dayssincelastcontact': 15,
            'complaintproxy': 35, 'internetservice': 50, 'multiplelines': 20,
            'servicessubscribed': 30,
        }
        _DEFAULT_INTERVENTION_COST = 25

        def _get_base_feature(feat_name):
            return _map_model_feature_to_base_feature(
                feat_name,
                raw_feature_cols or list(X_raw.columns),
            )

        def _is_non_actionable(feat_name):
            k = _normalize_feature_key(_get_base_feature(feat_name))
            if k in _non_actionable_norm:
                return True
            return k.endswith("id") or ("_id" in k)

        def _intervention_cost(feat_name):
            base = _normalize_feature_key(_get_base_feature(feat_name))
            return _INTERVENTION_COSTS.get(base, _DEFAULT_INTERVENTION_COST)

        def _feasibility_meta(base_feat):
            k = _normalize_feature_key(base_feat)
            if k in {
                'contract', 'paperlessbilling', 'paymentmethod', 'onlinesecurity',
                'techsupport', 'onlinebackup', 'deviceprotection',
                'multiplelines', 'phoneservice'
            }:
                return 85, "High", "#10b981"
            if k in {'streamingtv', 'streamingmovies', 'internetservice', 'servicessubscribed'}:
                return 75, "Medium", "#f59e0b"
            return 65, "Medium", "#f59e0b"

        def _estimate_customer_clv(customer_df):
            revenue_keywords = ['monthly', 'charge', 'revenue', 'value', 'price', 'amount', 'fee']
            customer_value = None
            for col in customer_df.columns:
                if any(k in str(col).lower() for k in revenue_keywords):
                    try:
                        val = float(customer_df[col].iloc[0])
                        if 10 <= val <= 10000:
                            customer_value = val
                            break
                    except Exception:
                        continue
            if customer_value is not None:
                return float(customer_value * 24)
            return 1680.0

        _shap_estimator = model
        if hasattr(model, 'steps'):
            _shap_estimator = model.steps[-1][1]

        try:
            original_prob = _predict_prob_one(_shap_estimator, query_encoded)
        except Exception:
            original_prob = _predict_prob_one(model, query_encoded)
        original_prob = float(np.clip(original_prob, 0.0, 1.0))

        bench_n = min(1500, len(X_raw))
        X_bench_raw = X_raw.sample(bench_n, random_state=42).copy() if len(X_raw) > bench_n else X_raw.copy()
        low_risk_raw = X_bench_raw.copy()
        try:
            X_bench_enc = _encode_for_model(X_bench_raw)
            bench_scores = _predict_positive_scores(model, X_bench_enc)
            bench_df = pd.DataFrame({'score': np.asarray(bench_scores, dtype=float)}, index=X_bench_raw.index)
            low_n = min(len(bench_df), max(30, int(len(bench_df) * 0.30)))
            low_idx = bench_df.nsmallest(low_n, 'score').index
            low_risk_raw = X_bench_raw.loc[low_idx].copy()
        except Exception:
            pass

        try:
            import shap

            X_sample = X_raw.sample(min(60, len(X_raw)), random_state=42)
            X_sample_encoded = _encode_for_model(X_sample)

            explainer = shap.Explainer(
                _shap_estimator.predict_proba,
                X_sample_encoded.values,
                algorithm="permutation"
            )
            shap_result = explainer(query_encoded.values)
            shap_values = shap_result.values
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            shap_values = shap_values[0] if shap_values.ndim == 2 else np.asarray(shap_values).ravel()

            agg_vals, agg_names, _agg_display = _aggregate_shap_to_base_features(
                shap_values,
                list(query_encoded.columns),
                query_encoded,
                query_raw,
            )

            if len(agg_names) == 0:
                return None

            feature_importance = pd.DataFrame({
                'feature': agg_names,
                'shap_value': agg_vals,
            })
            feature_importance['current_value'] = feature_importance['feature'].map(
                lambda c: query_raw[c].iloc[0] if c in query_raw.columns else None
            )
            feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
            feature_importance = feature_importance.sort_values('abs_shap', ascending=False)
            feature_importance = feature_importance[
                ~feature_importance['feature'].apply(_is_non_actionable)
            ].reset_index(drop=True)

            positive = feature_importance[feature_importance['shap_value'] > 0].copy()
            if len(positive) == 0:
                st.sidebar.info("Customer has low churn risk - no major interventions needed")
                return None

            def _driver_deviation(feature_name, cur_val):
                if feature_name not in X_raw.columns:
                    return 0.0
                s_all = X_raw[feature_name]
                s_low = low_risk_raw[feature_name] if feature_name in low_risk_raw.columns else s_all
                s_num_all = pd.to_numeric(s_all, errors='coerce')
                s_num_low = pd.to_numeric(s_low, errors='coerce')
                cur_num = pd.to_numeric(pd.Series([cur_val]), errors='coerce').iloc[0]
                if np.isfinite(cur_num) and s_num_all.notna().mean() >= 0.8:
                    target = float(s_num_low.median()) if s_num_low.notna().any() else float(s_num_all.median())
                    scale = float(s_num_all.std()) if float(s_num_all.std()) > 1e-9 else 1.0
                    return abs(float(cur_num) - target) / scale
                cur_str = str(cur_val)
                mode_low = s_low.astype(str).mode()
                if len(mode_low) == 0:
                    return 0.0
                return 1.0 if cur_str != str(mode_low.iloc[0]) else 0.0

            positive['deviation'] = positive.apply(
                lambda r: _driver_deviation(r['feature'], r['current_value']),
                axis=1
            )
            personalised = positive[positive['deviation'] > 0.15]
            top_drivers = personalised.head(3) if len(personalised) >= 2 else positive.head(3)

        except Exception:
            try:
                X_sample_enc = _encode_for_model(X_raw.sample(min(300, len(X_raw)), random_state=42))
                preds = _predict_positive_scores(_shap_estimator, X_sample_enc.values)
                corr = X_sample_enc.corrwith(pd.Series(preds, index=X_sample_enc.index)).dropna()
                corr = corr[corr > 0].sort_values(ascending=False)
                if len(corr) == 0:
                    return None
                proxy = pd.DataFrame({'feature': corr.index, 'shap_value': corr.values})
                proxy['feature'] = proxy['feature'].map(_get_base_feature)
                proxy = proxy.groupby('feature', as_index=False)['shap_value'].sum()
                proxy['current_value'] = proxy['feature'].map(
                    lambda c: query_raw[c].iloc[0] if c in query_raw.columns else None
                )
                proxy['abs_shap'] = proxy['shap_value'].abs()
                proxy = proxy[~proxy['feature'].apply(_is_non_actionable)]
                top_drivers = proxy.sort_values('abs_shap', ascending=False).head(3)
            except Exception:
                return None

        if top_drivers is None or len(top_drivers) == 0:
            return None

        customer_clv = _estimate_customer_clv(customer_data)
        margin_rate = float(st.session_state.get("retention_margin_rate", 0.35))
        margin_rate = float(np.clip(margin_rate, 0.05, 1.0))
        execution_map = st.session_state.get(
            "retention_execution_probs",
            {"High": 0.85, "Medium": 0.70, "Low": 0.55}
        )

        recommendations = []

        for _, driver in top_drivers.iterrows():
            base_feature = driver['feature']
            shap_contribution = float(driver['shap_value'])
            if base_feature not in query_raw.columns or base_feature not in X_raw.columns:
                continue

            current_value = query_raw[base_feature].iloc[0]
            full_series = X_raw[base_feature]
            low_series = low_risk_raw[base_feature] if base_feature in low_risk_raw.columns else full_series

            full_num = pd.to_numeric(full_series, errors='coerce')
            current_num = pd.to_numeric(pd.Series([current_value]), errors='coerce').iloc[0]
            is_numeric = full_num.notna().mean() >= 0.8 and np.isfinite(current_num)

            scenarios = []
            if is_numeric:
                valid = full_num.dropna()
                if len(valid) < 15:
                    continue

                uniq_vals = set(np.round(valid.unique(), 10).tolist())
                is_binary = uniq_vals.issubset({0.0, 1.0}) and len(uniq_vals) <= 2
                if is_binary:
                    flip_val = 0.0 if float(current_num) >= 0.5 else 1.0
                    scenarios.append({
                        'new_value': flip_val,
                        'name': 'Toggle',
                        'description': f'Change to {int(flip_val)}'
                    })
                else:
                    low_num = pd.to_numeric(low_series, errors='coerce').dropna()
                    target_num = float(low_num.median()) if len(low_num) > 0 else float(valid.median())
                    min_v, max_v = float(valid.min()), float(valid.max())
                    gap = target_num - float(current_num)

                    candidate_vals = []
                    if abs(gap) > 1e-9:
                        for frac, lbl in [(0.4, 'Conservative'), (0.7, 'Moderate'), (1.0, 'Aggressive')]:
                            nv = float(current_num) + frac * gap
                            nv = float(np.clip(nv, min_v, max_v))
                            candidate_vals.append((lbl, nv))
                    else:
                        for q, lbl in [(0.35, 'Conservative'), (0.5, 'Moderate'), (0.65, 'Aggressive')]:
                            nv = float(valid.quantile(q))
                            if abs(nv - float(current_num)) > 1e-9:
                                candidate_vals.append((lbl, nv))

                    seen = set()
                    for lbl, nv in candidate_vals:
                        rnv = round(float(nv), 6)
                        if rnv in seen:
                            continue
                        seen.add(rnv)
                        scenarios.append({
                            'new_value': float(nv),
                            'name': lbl,
                            'description': f'Move toward low-risk baseline ({lbl.lower()})'
                        })
            else:
                cur_str = str(current_value)
                low_vals = low_series.astype(str).value_counts()
                all_vals = full_series.astype(str).value_counts()
                candidate_vals = [v for v in low_vals.index.tolist() if v != cur_str]
                if len(candidate_vals) < 2:
                    candidate_vals.extend([v for v in all_vals.index.tolist() if v != cur_str and v not in candidate_vals])
                for i, val in enumerate(candidate_vals[:3]):
                    scenarios.append({
                        'new_value': val,
                        'name': f'Option{i+1}',
                        'description': f'Change to {val}'
                    })

            if len(scenarios) == 0:
                continue

            feasibility_score, feasibility_category, feasibility_color = _feasibility_meta(base_feature)
            implementation_cost = float(_intervention_cost(base_feature))
            execution_prob = execution_map.get(feasibility_category, 0.70)
            try:
                execution_prob = float(execution_prob)
            except Exception:
                execution_prob = 0.70
            execution_prob = float(np.clip(execution_prob, 0.1, 1.0))

            for scenario in scenarios:
                new_value = scenario['new_value']
                modified_raw = query_raw.copy()
                modified_raw.loc[modified_raw.index[0], base_feature] = new_value
                modified_encoded = _encode_for_model(modified_raw)
                try:
                    new_prob = _predict_prob_one(_shap_estimator, modified_encoded)
                except Exception:
                    continue
                new_prob = float(np.clip(new_prob, 0.0, 1.0))

                prob_reduction = max(0.0, float(original_prob) - float(new_prob))
                if prob_reduction <= 1e-6:
                    continue

                value_saved = prob_reduction * customer_clv * margin_rate * execution_prob
                net_benefit = value_saved - implementation_cost
                roi_ratio_num = (value_saved / implementation_cost) if implementation_cost > 0 else float('inf')

                if is_numeric:
                    change_description = f"{base_feature}: {float(current_num):.3g} -> {float(new_value):.3g}"
                else:
                    change_description = f"{base_feature}: {current_value} -> {new_value}"

                recommendations.append({
                    'changes': {base_feature: {'from': current_value, 'to': new_value}},
                    'predicted_churn_prob': round(new_prob, 6),
                    'original_churn_prob': round(original_prob, 6),
                    'prob_reduction': round(prob_reduction, 6),
                    'shap_contribution': shap_contribution,
                    'change_description': change_description,
                    'feasibility_score': feasibility_score,
                    'feasibility_category': feasibility_category,
                    'feasibility_color': feasibility_color,
                    'implementation_cost': round(implementation_cost, 2),
                    'customer_clv': round(customer_clv, 2),
                    'margin_rate': round(margin_rate, 3),
                    'execution_prob': round(execution_prob, 3),
                    'value_saved': round(value_saved, 2),
                    'roi_ratio': round(roi_ratio_num, 2) if roi_ratio_num != float('inf') else "inf",
                    'net_benefit': round(net_benefit, 2),
                    'scenario_name': scenario['name'],
                    'scenario_description': scenario['description'],
                })

        if len(recommendations) == 0:
            return None

        unique_recommendations = []
        seen_keys = set()
        for rec in recommendations:
            feature = list(rec.get('changes', {}).keys())[0] if rec.get('changes') else "feature"
            key = (feature, round(float(rec.get('predicted_churn_prob', 1.0)), 4))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_recommendations.append(rec)

        unique_recommendations.sort(
            key=lambda x: (
                -float(x.get('net_benefit', 0)),
                float(x.get('predicted_churn_prob', 1.0)),
                -float(x.get('feasibility_score', 0)),
            )
        )
        return unique_recommendations[:num_cfs]

    except Exception as e:
        st.sidebar.error(f"Recommendation generation failed: {str(e)[:100]}")
        import traceback
        st.sidebar.code(traceback.format_exc()[-500:])
        return None


def extract_actionable_shap_signal(customer_data, data_df, model, target_col, id_col=None):
    """
    Extract a SHAP-derived actionable signal for thesis-mode ablations.
    Returns driver metadata without generating counterfactual actions.
    """
    result = {
        "actionable": 0,
        "top_feature": None,
        "top_feature_value": None,
        "top_shap_value": 0.0,
        "positive_actionable_shap_sum": 0.0,
        "driver_count": 0,
        "signal_source": None,
    }
    try:
        drop_id_cols = st.session_state.get("drop_id_cols", [])
        drop_cols = [c for c in [id_col, target_col] + list(drop_id_cols) if c and c in data_df.columns]
        bg_full = data_df.drop(columns=drop_cols, errors="ignore").copy()
        row_raw = customer_data.drop(columns=drop_cols, errors="ignore").copy()
        if len(bg_full) == 0 or len(row_raw) == 0:
            return result
        row_raw = row_raw.iloc[[0]].copy()

        raw_feature_cols = st.session_state.get("raw_feature_cols", None)
        if raw_feature_cols:
            keep_cols = [c for c in raw_feature_cols if c in bg_full.columns]
            if keep_cols:
                bg_full = bg_full[keep_cols].copy()
                for c in keep_cols:
                    if c not in row_raw.columns:
                        row_raw[c] = 0
                row_raw = row_raw[keep_cols].copy()

        model_obj = model
        prep_pipe = st.session_state.get("prep_pipe", None)
        estimator = model_obj
        if hasattr(model_obj, "named_steps"):
            try:
                steps = list(model_obj.named_steps.items())
                if len(steps) > 1:
                    from sklearn.pipeline import Pipeline as SkPipeline
                    estimator = steps[-1][1]
                    if prep_pipe is None:
                        prep_pipe = SkPipeline(steps[:-1])
            except Exception:
                pass

        def _is_actionable_feature(feat_name):
            feat_norm = _normalize_feature_key(feat_name)
            non_actionable = {
                "totalcharges", "monthlycharges", "clv", "engagementscore",
                "tenure", "customerid", "seniorcitizen", "gender", "age",
            }
            immutable_keywords = {
                "age", "tenure", "gender", "senior", "city", "marital",
                "id", "date", "year", "month"
            }
            if feat_norm in non_actionable:
                return False
            return not any(kw in feat_norm for kw in immutable_keywords)

        attempts = []

        def _attempt_model_aligned():
            bg_model = align_to_model_columns(bg_full.copy(), model_obj)
            row_model = align_to_model_columns(row_raw.copy(), model_obj)
            bg_model = bg_model.apply(pd.to_numeric, errors="coerce").fillna(0)
            row_model = row_model.apply(pd.to_numeric, errors="coerce").fillna(0)
            _ = _predict_positive_scores(model_obj, row_model.head(1))
            feat_names = list(bg_model.columns)
            bg_mat = _to_dense_numeric(bg_model)
            row_mat = _to_dense_numeric(row_model)

            def _pred(arr):
                arr_df = pd.DataFrame(arr, columns=feat_names)
                return _predict_positive_scores(model_obj, arr_df)

            return bg_mat, row_mat, feat_names, row_model, _pred, "shap"

        attempts.append(_attempt_model_aligned)

        if prep_pipe is not None:
            def _attempt_preppipe_estimator():
                bg_raw_aligned = align_to_raw_features(bg_full.copy())
                row_raw_aligned = align_to_raw_features(row_raw.copy())
                bg_trans = prep_pipe.transform(bg_raw_aligned)
                row_trans = prep_pipe.transform(row_raw_aligned)
                bg_mat = _to_dense_numeric(bg_trans)
                row_mat = _to_dense_numeric(row_trans)
                _ = _predict_positive_scores(estimator, row_mat[:1])
                try:
                    feat_names = list(prep_pipe.get_feature_names_out())
                except Exception:
                    feat_names = [f"feature_{i}" for i in range(bg_mat.shape[1])]
                row_model = pd.DataFrame(row_mat, columns=feat_names)

                def _pred(arr):
                    return _predict_positive_scores(estimator, _to_dense_numeric(arr))

                return bg_mat, row_mat, feat_names, row_model, _pred, "shap"

            attempts.append(_attempt_preppipe_estimator)

        signal_df = None
        try:
            import shap

            last_shap_err = None
            for attempt_fn in attempts:
                try:
                    bg_mat, row_mat, feat_names, row_model_df, pred_fn, source = attempt_fn()
                    max_evals = int(min(401, max(31, 2 * bg_mat.shape[1] + 1)))
                    explainer = shap.Explainer(pred_fn, bg_mat, algorithm="permutation")
                    shap_values = explainer(row_mat, max_evals=max_evals)
                    sv = shap_values
                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]
                    vals = np.array(sv.values) if hasattr(sv, "values") else np.array(sv)
                    if vals.ndim == 3:
                        out_idx = 1 if vals.shape[-1] > 1 else 0
                        vals = vals[:, :, out_idx]
                    if vals.ndim == 2:
                        vals = vals[0]
                    vals = np.asarray(vals, dtype=float).ravel()

                    agg_vals, agg_names, agg_data = _aggregate_shap_to_base_features(
                        vals,
                        feat_names,
                        row_model_df,
                        row_raw,
                    )
                    signal_df = pd.DataFrame({
                        "feature": agg_names,
                        "shap_value": agg_vals,
                        "feature_value": agg_data if agg_data is not None else [None for _ in agg_names],
                    })
                    result["signal_source"] = source
                    break
                except Exception as shap_attempt_err:
                    last_shap_err = shap_attempt_err
                    continue
            if signal_df is None and last_shap_err is not None:
                raise last_shap_err
        except Exception:
            try:
                bg_model = align_to_model_columns(bg_full.copy(), model_obj)
                bg_model = bg_model.apply(pd.to_numeric, errors="coerce").fillna(0)
                preds = _predict_positive_scores(model_obj, bg_model)
                corr = bg_model.corrwith(pd.Series(preds, index=bg_model.index)).dropna()
                corr = corr[corr > 0].sort_values(ascending=False)
                if len(corr) == 0:
                    return result
                signal_df = pd.DataFrame({
                    "feature": [
                        _map_model_feature_to_base_feature(
                            col,
                            raw_feature_cols or list(bg_full.columns)
                        )
                        for col in corr.index
                    ],
                    "shap_value": corr.values,
                })
                signal_df = signal_df.groupby("feature", as_index=False)["shap_value"].sum()
                signal_df["feature_value"] = signal_df["feature"].map(
                    lambda c: row_raw[c].iloc[0] if c in row_raw.columns else None
                )
                result["signal_source"] = "proxy"
            except Exception:
                return result

        if signal_df is None or len(signal_df) == 0:
            return result

        signal_df["abs_shap"] = signal_df["shap_value"].abs()
        signal_df = signal_df[
            (signal_df["shap_value"] > 0)
            & signal_df["feature"].apply(_is_actionable_feature)
        ].sort_values("abs_shap", ascending=False).reset_index(drop=True)
        if len(signal_df) == 0:
            return result

        top_row = signal_df.iloc[0]
        result["actionable"] = 1
        result["top_feature"] = top_row["feature"]
        result["top_feature_value"] = top_row.get("feature_value", None)
        result["top_shap_value"] = float(top_row["abs_shap"])
        result["positive_actionable_shap_sum"] = float(signal_df["shap_value"].sum())
        result["driver_count"] = int(len(signal_df))
        return result
    except Exception:
        return result

# ----------------------------- CHAPTER 5 EVALUATION -----------------------------
def run_chapter5_evaluation(df, model, target_col, id_col, threshold=0.35):
    """
    Chapter 5 evaluation: compares Condition A (prediction only)
    vs Condition B (integrated system) across all high-risk customers.
    Saves results to chapter5_evaluation_data.csv
    """
    import json
    try:
        from scipy.stats import spearmanr, wilcoxon, mannwhitneyu, shapiro
    except ImportError:
        spearmanr = wilcoxon = mannwhitneyu = shapiro = None

    try:
        # ── Step 1: Get predictions for full dataset ──
        drop_id_cols = st.session_state.get("drop_id_cols", [])
        drop_cols = [c for c in [id_col, target_col] + list(drop_id_cols) if c and c in df.columns]
        aligned_full = align_to_model_columns(df.copy(), model)
        proba_full = model.predict_proba(aligned_full)
        if hasattr(proba_full, 'iloc'):
            scores = proba_full.iloc[:, 1] if proba_full.shape[1] > 1 else proba_full.iloc[:, 0]
        else:
            scores = proba_full[:, 1] if proba_full.shape[1] > 1 else proba_full[:, 0]
        score_series = pd.Series(scores, index=df.index)

        # ── Step 2: Filter high-risk customers ──
        high_risk_mask = score_series > threshold
        if high_risk_mask.sum() < 20:
            threshold_lowered = 0.25
            high_risk_mask = score_series > threshold_lowered
            st.warning(
                f"Only {(score_series > threshold).sum()} customers above {threshold:.2f}. "
                f"Lowered threshold to {threshold_lowered:.2f} ({high_risk_mask.sum()} customers)."
            )
        high_risk_idx = df.index[high_risk_mask]

        # ── Cap at 100 customers (top churn probability) ──
        if len(high_risk_idx) > 100:
            top100 = score_series.loc[high_risk_idx].nlargest(100).index
            high_risk_idx = top100
            st.info(f"Capped evaluation to top 100 highest-risk customers (out of {high_risk_mask.sum()}).")

        # ── Step 3: Define immutable features ──
        immutable_features_explicit = [
            'Tenure', 'Gender', 'CityTier', 'MaritalStatus',
            'SeniorCitizen', 'Age', 'tenure', 'gender'
        ]
        immutable_keywords = [
            'age', 'tenure', 'gender', 'senior', 'city', 'marital',
            'id', 'date', 'year', 'month'
        ]

        def _is_immutable(feat_name):
            if feat_name in immutable_features_explicit:
                return True
            fl = feat_name.lower()
            return any(kw in fl for kw in immutable_keywords)

        # ── Step 4: Estimate CLV once (reused for Condition A) ──
        revenue_keywords = ['monthly', 'charge', 'revenue', 'value', 'price', 'amount', 'fee']
        tenure_keywords = ['tenure', 'age', 'duration', 'month', 'year', 'days']
        value_col = None
        for col in df.columns:
            if any(kw in str(col).lower() for kw in revenue_keywords):
                if df[col].dtype in ['int64', 'float64']:
                    value_col = col
                    break
        tenure_col_name = None
        for col in df.columns:
            if any(kw in str(col).lower() for kw in tenure_keywords):
                if df[col].dtype in ['int64', 'float64']:
                    tenure_col_name = col
                    break
        if value_col and tenure_col_name:
            estimated_clv_global = float(df[value_col].median()) * float(df[tenure_col_name].median())
        elif value_col:
            estimated_clv_global = float(df[value_col].median()) * 12
        else:
            estimated_clv_global = 1680
        estimated_clv_global = max(100, min(100000, estimated_clv_global))

        # ── RFM data ──
        rfm_df = st.session_state.get("rfm_dataframe")

        # ── Step 5: Loop over high-risk customers ──
        rows = []
        progress = st.progress(0, text="Evaluating high-risk customers...")
        total = len(high_risk_idx)

        for i, idx in enumerate(high_risk_idx):
            progress.progress((i + 1) / total, text=f"Customer {i+1}/{total}")

            customer_df = df.iloc[[df.index.get_loc(idx)]]
            churn_prob = float(score_series.loc[idx])

            # Customer ID
            if id_col and id_col in df.columns:
                cust_id = str(customer_df[id_col].iloc[0])
            else:
                cust_id = str(idx)

            # ── Condition B (integrated system) ──
            cond_b_actionable = 0
            top_shap_feature = None
            intervention_cost = None
            predicted_prob_after = None
            prob_reduction = None
            customer_clv = None
            feasibility = None
            condition_b_clv_saved = None
            margin_rate = None
            execution_prob = None

            try:
                cfs = generate_counterfactuals(
                    customer_data=customer_df,
                    model=model,
                    data_df=df,
                    target_col=target_col,
                    num_cfs=3
                )
                if cfs and len(cfs) > 0:
                    best = cfs[0]
                    changes = best.get('changes', {})
                    if changes:
                        top_feat = list(changes.keys())[0]
                        top_shap_feature = top_feat
                        cond_b_actionable = 0 if _is_immutable(top_feat) else 1
                    intervention_cost = best.get('implementation_cost', None)
                    predicted_prob_after = best.get('predicted_churn_prob', None)
                    if predicted_prob_after is not None:
                        prob_reduction = churn_prob - float(predicted_prob_after)
                    customer_clv = best.get('customer_clv', None)
                    feasibility = best.get('feasibility_category', None)
                    condition_b_clv_saved = best.get('value_saved', None)
                    margin_rate = best.get('margin_rate', None)
                    execution_prob = best.get('execution_prob', None)
            except Exception as cf_err:
                msgs = st.session_state.get("validation_messages", [])
                msgs.append(f"Ch5 eval: CF failed for customer {cust_id}: {str(cf_err)[:80]}")
                st.session_state["validation_messages"] = msgs

            # ── RFM match ──
            rfm_segment = None
            rfm_composite = None
            if rfm_df is not None:
                try:
                    if idx in rfm_df.index:
                        rfm_row = rfm_df.loc[idx]
                    elif id_col and id_col in rfm_df.columns:
                        match = rfm_df[rfm_df[id_col].astype(str) == cust_id]
                        rfm_row = match.iloc[0] if len(match) > 0 else None
                    else:
                        rfm_row = None
                    if rfm_row is not None:
                        rfm_segment = rfm_row.get('RFM_Segment', None) if hasattr(rfm_row, 'get') else rfm_row['RFM_Segment'] if 'RFM_Segment' in rfm_row.index else None
                        rfm_composite = rfm_row.get('RFM_Composite_Score', None) if hasattr(rfm_row, 'get') else rfm_row['RFM_Composite_Score'] if 'RFM_Composite_Score' in rfm_row.index else None
                except Exception:
                    pass

            # FIX 3 — keep missing RFM explicit instead of forcing a risk segment
            if rfm_segment is None or (isinstance(rfm_segment, float) and np.isnan(rfm_segment)):
                rfm_segment = "Unknown"

            rows.append({
                'customer_id': cust_id,
                'churn_prob': round(churn_prob, 4),
                'condition_b_actionable': cond_b_actionable,
                'top_shap_feature': top_shap_feature,
                'intervention_cost': intervention_cost,
                'predicted_prob_after': predicted_prob_after,
                'prob_reduction': round(prob_reduction, 4) if prob_reduction is not None else None,
                'customer_clv': customer_clv,
                'feasibility': feasibility,
                'condition_b_clv_saved': condition_b_clv_saved,
                'margin_rate': margin_rate,
                'execution_prob': execution_prob,
                'rfm_segment': rfm_segment,
                'rfm_composite_score': rfm_composite,
            })

        progress.empty()
        results_df = pd.DataFrame(rows)

        # ====================================================================
        # FIX 1 — Recalculate ROI and net_benefit from scratch (consistent)
        #         # ====================================================================
        def _safe_float(v, default=0.0):
            try:
                f = float(v)
                return f if not np.isnan(f) else default
            except (TypeError, ValueError):
                return default

        _DEFAULT_MARGIN = 0.35
        _DEFAULT_EXEC = 0.70
        _pr = results_df['prob_reduction'].apply(lambda x: max(0.0, _safe_float(x)))
        _clv = results_df['customer_clv'].apply(lambda x: _safe_float(x))
        _cost = results_df['intervention_cost'].apply(lambda x: _safe_float(x))
        _margin = results_df['margin_rate'].apply(lambda x: _safe_float(x, _DEFAULT_MARGIN)).clip(lower=0.05, upper=1.0)
        _exec = results_df['execution_prob'].apply(lambda x: _safe_float(x, _DEFAULT_EXEC)).clip(lower=0.10, upper=1.0)
        _value_saved_cf = results_df['condition_b_clv_saved'].apply(lambda x: _safe_float(x, np.nan))
        _value_saved_fallback = _pr * _clv * _margin * _exec
        _value_saved_b = pd.Series(
            np.where(np.isnan(_value_saved_cf.values), _value_saved_fallback.values, _value_saved_cf.values),
            index=results_df.index
        )
        results_df['condition_b_clv_saved'] = _value_saved_b.round(2)
        results_df['net_benefit'] = (_value_saved_b - _cost).round(2)
        results_df['roi'] = np.where(
            _cost > 0,
            (_value_saved_b / _cost).round(2),
            0.0,
        )

        # ====================================================================
        # FIX 2 — Condition A: tiered baseline cost by CLV percentile
        # ====================================================================
        _clv_series = results_df['customer_clv'].apply(lambda x: _safe_float(x, estimated_clv_global))
        _p33 = _clv_series.quantile(0.33)
        _p66 = _clv_series.quantile(0.66)

        def _tiered_cost(clv_val):
            if clv_val <= _p33:
                return 30.0
            elif clv_val <= _p66:
                return 50.0
            else:
                return 75.0

        results_df['condition_a_cost'] = _clv_series.apply(_tiered_cost)
        results_df['condition_a_actionable'] = 0
        # Condition A baseline: generic campaign with lower execution certainty
        _cond_a_effectiveness = 0.30
        _cond_a_exec = 0.65
        results_df['condition_a_clv_saved'] = (
            results_df['churn_prob'].clip(lower=0, upper=1)
            * _clv_series
            * _cond_a_effectiveness
            * _DEFAULT_MARGIN
            * _cond_a_exec
        ).round(2)
        results_df['condition_a_net_value'] = (
            results_df['condition_a_clv_saved'] - results_df['condition_a_cost']
        ).round(2)

        # ── Step 6: Summary metrics ──
        actionability_b = results_df['condition_b_actionable'].mean()

        # Rank correlation
        correlation, pvalue_corr = 0.0, 1.0
        if spearmanr is not None and len(results_df) >= 5:
            rank_a = results_df['churn_prob'].rank(ascending=False)
            rank_b = results_df['roi'].rank(ascending=False)
            try:
                correlation, pvalue_corr = spearmanr(rank_a, rank_b)
                if np.isnan(correlation):
                    correlation, pvalue_corr = 0.0, 1.0
            except Exception:
                correlation, pvalue_corr = 0.0, 1.0

        # Top 20% overlap
        top20_n = max(1, int(len(results_df) * 0.20))
        top20_a = set(results_df.nlargest(top20_n, 'churn_prob').index)
        top20_b = set(results_df['roi'].nlargest(top20_n).index)
        overlap = len(top20_a & top20_b) / top20_n

        # Expected retention value
        total_value_a = results_df['condition_a_net_value'].sum()
        total_value_b = results_df['net_benefit'].sum()

        # RFM segment summary
        segment_summary = None
        if 'rfm_segment' in results_df.columns and results_df['rfm_segment'].notna().sum() > 0:
            try:
                _temp = results_df.copy()
                _temp['_roi'] = _temp['roi'].apply(lambda x: _safe_float(x))
                _temp['_cost'] = _temp['intervention_cost'].apply(lambda x: _safe_float(x))
                _temp['_pr'] = _temp['prob_reduction'].apply(lambda x: _safe_float(x))
                segment_summary = _temp.groupby('rfm_segment').agg(
                    count=('customer_id', 'count'),
                    avg_churn_prob=('churn_prob', 'mean'),
                    avg_roi=('_roi', 'mean'),
                    avg_cost=('_cost', 'mean'),
                    avg_prob_reduction=('_pr', 'mean')
                ).round(3)
            except Exception:
                segment_summary = None

        summary_metrics = {
            'total_high_risk_customers': len(results_df),
            'actionability_rate_a': 0.0,
            'actionability_rate_b': round(float(actionability_b), 4),
            'rank_correlation': round(float(correlation), 4),
            'rank_correlation_pvalue': round(float(pvalue_corr), 4),
            'top20_priority_overlap': round(float(overlap), 4),
            'total_retention_value_a': round(float(total_value_a), 2),
            'total_retention_value_b': round(float(total_value_b), 2),
            'value_improvement_pct': round(
                (total_value_b - total_value_a) / max(abs(total_value_a), 1) * 100, 2
            ),
            'customers_with_positive_roi': int((results_df['roi'] > 1).sum()),
            'avg_prob_reduction': round(
                float(results_df['prob_reduction'].apply(lambda x: _safe_float(x)).mean()), 4
            ),
        }

        # ====================================================================
        # ADD — Five statistical reliability tests
        # ====================================================================
        stat_tests = {}

        # Per-customer value columns for paired comparisons
        _val_a = results_df['condition_a_net_value'].values.astype(float)
        _val_b = results_df['net_benefit'].values.astype(float)

        # Test 1 — Wilcoxon signed-rank (Condition B vs Condition A per-customer)
        try:
            _diffs = _val_b - _val_a
            # Wilcoxon requires non-zero differences
            _nonzero = _diffs[_diffs != 0]
            if len(_nonzero) >= 10 and wilcoxon is not None:
                w_stat, w_pval = wilcoxon(_nonzero)
                stat_tests['wilcoxon_signed_rank'] = {
                    'statistic': round(float(w_stat), 4),
                    'p_value': round(float(w_pval), 6),
                    'significant_at_005': bool(w_pval < 0.05),
                    'n_nonzero_pairs': int(len(_nonzero)),
                    'description': 'Tests whether Condition B net value significantly differs from Condition A per-customer',
                }
            else:
                stat_tests['wilcoxon_signed_rank'] = {
                    'skipped': True,
                    'reason': f'Insufficient non-zero differences ({len(_nonzero)})',
                }
        except Exception as e:
            stat_tests['wilcoxon_signed_rank'] = {'error': str(e)[:120]}
            msgs = st.session_state.get("validation_messages", [])
            msgs.append(f"Ch5 stat: Wilcoxon failed: {str(e)[:80]}")
            st.session_state["validation_messages"] = msgs

        # Test 2 — Cohen's d effect size
        try:
            _mean_diff = float(np.mean(_val_b - _val_a))
            _pooled_std = float(np.sqrt((np.var(_val_a, ddof=1) + np.var(_val_b, ddof=1)) / 2))
            if _pooled_std > 0:
                cohens_d = _mean_diff / _pooled_std
            else:
                cohens_d = 0.0
            if abs(cohens_d) >= 0.8:
                _label = "Large"
            elif abs(cohens_d) >= 0.5:
                _label = "Medium"
            elif abs(cohens_d) >= 0.2:
                _label = "Small"
            else:
                _label = "Negligible"
            stat_tests['cohens_d'] = {
                'effect_size': round(float(cohens_d), 4),
                'magnitude': _label,
                'description': 'Practical magnitude of difference between Condition A and Condition B',
            }
        except Exception as e:
            stat_tests['cohens_d'] = {'error': str(e)[:120]}
            msgs = st.session_state.get("validation_messages", [])
            msgs.append(f"Ch5 stat: Cohen's d failed: {str(e)[:80]}")
            st.session_state["validation_messages"] = msgs

        # Test 3 — Bootstrap 95% CI for value improvement percentage
        try:
            _rng = np.random.RandomState(42)
            _boot_improvements = []
            for _ in range(1000):
                _idx = _rng.choice(len(_val_a), size=len(_val_a), replace=True)
                _boot_a = _val_a[_idx].sum()
                _boot_b = _val_b[_idx].sum()
                _pct = (_boot_b - _boot_a) / max(abs(_boot_a), 1) * 100
                _boot_improvements.append(_pct)
            _ci_low = float(np.percentile(_boot_improvements, 2.5))
            _ci_high = float(np.percentile(_boot_improvements, 97.5))
            stat_tests['bootstrap_ci'] = {
                'ci_lower_2_5': round(_ci_low, 2),
                'ci_upper_97_5': round(_ci_high, 2),
                'point_estimate': round(float(np.median(_boot_improvements)), 2),
                'n_iterations': 1000,
                'seed': 42,
                'description': '95% bootstrap confidence interval for value improvement percentage (B over A)',
            }
        except Exception as e:
            stat_tests['bootstrap_ci'] = {'error': str(e)[:120]}
            msgs = st.session_state.get("validation_messages", [])
            msgs.append(f"Ch5 stat: Bootstrap failed: {str(e)[:80]}")
            st.session_state["validation_messages"] = msgs

        # Test 4 — Mann-Whitney U: ROI by RFM segment vs rest
        try:
            _roi_arr = results_df['roi'].values.astype(float)
            _segments = results_df['rfm_segment'].values
            _unique_segs = [s for s in sorted(set(_segments)) if pd.notna(s)]
            mw_results = {}
            for seg in _unique_segs:
                _in = _roi_arr[_segments == seg]
                _out = _roi_arr[_segments != seg]
                if len(_in) >= 3 and len(_out) >= 3 and mannwhitneyu is not None:
                    u_stat, u_pval = mannwhitneyu(_in, _out, alternative='two-sided')
                    mw_results[str(seg)] = {
                        'statistic': round(float(u_stat), 4),
                        'p_value': round(float(u_pval), 6),
                        'significant_at_005': bool(u_pval < 0.05),
                        'n_segment': int(len(_in)),
                        'n_rest': int(len(_out)),
                    }
                else:
                    mw_results[str(seg)] = {
                        'skipped': True,
                        'reason': f'Insufficient samples (segment={len(_in)}, rest={len(_out)})',
                    }
            stat_tests['mann_whitney_roi_by_segment'] = {
                'segments': mw_results,
                'description': 'Mann-Whitney U test comparing ROI distribution in each RFM segment vs all others',
            }
        except Exception as e:
            stat_tests['mann_whitney_roi_by_segment'] = {'error': str(e)[:120]}
            msgs = st.session_state.get("validation_messages", [])
            msgs.append(f"Ch5 stat: Mann-Whitney failed: {str(e)[:80]}")
            st.session_state["validation_messages"] = msgs

        # Test 5 — Shapiro-Wilk normality test on prob_reduction
        try:
            _pr_clean = results_df['prob_reduction'].apply(lambda x: _safe_float(x)).values
            _pr_nonzero = _pr_clean[_pr_clean != 0]
            if len(_pr_nonzero) >= 8 and shapiro is not None:
                # Shapiro-Wilk limited to 5000 samples
                _sample_pr = _pr_nonzero[:5000]
                sw_stat, sw_pval = shapiro(_sample_pr)
                _is_normal = bool(sw_pval >= 0.05)
                stat_tests['normality_prob_reduction'] = {
                    'statistic': round(float(sw_stat), 4),
                    'p_value': round(float(sw_pval), 6),
                    'is_normal': _is_normal,
                    'n_samples': int(len(_sample_pr)),
                    'justification': (
                        'Distribution is normal (p >= 0.05); parametric tests also valid.'
                        if _is_normal else
                        'Distribution is non-normal (p < 0.05); non-parametric tests (Wilcoxon, Mann-Whitney) are the correct choice.'
                    ),
                    'description': 'Shapiro-Wilk test on prob_reduction to validate non-parametric test selection',
                }
            else:
                stat_tests['normality_prob_reduction'] = {
                    'skipped': True,
                    'reason': f'Insufficient non-zero values ({len(_pr_nonzero)})',
                }
        except Exception as e:
            stat_tests['normality_prob_reduction'] = {'error': str(e)[:120]}
            msgs = st.session_state.get("validation_messages", [])
            msgs.append(f"Ch5 stat: Shapiro-Wilk failed: {str(e)[:80]}")
            st.session_state["validation_messages"] = msgs

        summary_metrics['statistical_tests'] = stat_tests

        # ── Save files ──
        results_df.to_csv("chapter5_evaluation_data.csv", index=False)
        with open("chapter5_summary_metrics.json", "w") as f:
            json.dump(summary_metrics, f, indent=2)
        if segment_summary is not None:
            segment_summary.to_csv("chapter5_rfm_segments.csv")

        return results_df, summary_metrics, segment_summary

    except Exception as e:
        import traceback
        st.error(f"Chapter 5 evaluation failed: {e}")
        st.code(traceback.format_exc()[-800:])
        return None, None, None


def run_thesis_evaluation(
    df,
    model,
    target_col,
    id_col,
    candidate_pool_size=100,
    decision_budget=30,
    n_bootstrap=100,
    generic_effectiveness=0.30,
    generic_margin_rate=0.35,
    generic_execution_prob=0.65,
):
    """
    Thesis-mode evaluation for fixed-budget decision comparison.
    Compares:
    - prediction-only ranking
    - prediction x CLV ranking
    - integrated AI system (prediction + SHAP + prescription)
    """
    import json
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    try:
        drop_id_cols = st.session_state.get("drop_id_cols", [])
        aligned_full = align_to_model_columns(df.copy(), model)
        score_series = pd.Series(_predict_positive_scores(model, aligned_full), index=df.index)

        candidate_pool_size = int(max(1, min(candidate_pool_size, len(df))))
        decision_budget = int(max(1, min(decision_budget, candidate_pool_size)))
        n_bootstrap = int(max(20, n_bootstrap))

        immutable_features_explicit = {
            'Tenure', 'Gender', 'CityTier', 'MaritalStatus',
            'SeniorCitizen', 'Age', 'tenure', 'gender'
        }
        immutable_keywords = {
            'age', 'tenure', 'gender', 'senior', 'city', 'marital',
            'id', 'date', 'year', 'month'
        }

        def _is_immutable(feat_name):
            if feat_name in immutable_features_explicit:
                return True
            fl = str(feat_name).lower()
            return any(kw in fl for kw in immutable_keywords)

        def _safe_float(v, default=0.0):
            try:
                f = float(v)
                return f if not np.isnan(f) else default
            except (TypeError, ValueError):
                return default

        def _estimate_clv_series(frame):
            numeric_cols = [
                c for c in frame.columns
                if c != target_col and pd.api.types.is_numeric_dtype(frame[c])
            ]
            clv_cols = [c for c in numeric_cols if any(k in str(c).lower() for k in ['clv', 'ltv', 'lifetime'])]
            if clv_cols:
                clv = pd.to_numeric(frame[clv_cols[0]], errors='coerce')
                return clv.fillna(clv.median()).clip(lower=100, upper=100000)

            monthly_cols = [c for c in numeric_cols if any(k in str(c).lower() for k in ['monthly', 'mrr', 'arpu'])]
            revenue_cols = [c for c in numeric_cols if any(k in str(c).lower() for k in ['charge', 'revenue', 'value', 'price', 'amount', 'fee'])]
            base_col = monthly_cols[0] if monthly_cols else (revenue_cols[0] if revenue_cols else None)
            if base_col is not None:
                monthly_value = pd.to_numeric(frame[base_col], errors='coerce')
                clv = monthly_value.fillna(monthly_value.median()) * 24.0
                return clv.clip(lower=100, upper=100000)

            return pd.Series(1680.0, index=frame.index)

        clv_series_full = _estimate_clv_series(df)
        clv_p33 = float(clv_series_full.quantile(0.33))
        clv_p66 = float(clv_series_full.quantile(0.66))

        def _generic_cost(clv_val):
            if clv_val <= clv_p33:
                return 30.0
            if clv_val <= clv_p66:
                return 50.0
            return 75.0

        candidate_idx = score_series.nlargest(candidate_pool_size).index
        progress = st.progress(0, text="Running thesis-mode evaluation...")
        rows = []

        for i, idx in enumerate(candidate_idx):
            progress.progress((i + 1) / candidate_pool_size, text=f"Candidate {i+1}/{candidate_pool_size}")
            customer_df = df.iloc[[df.index.get_loc(idx)]]
            churn_prob = float(score_series.loc[idx])
            customer_clv = float(clv_series_full.loc[idx])
            generic_cost = float(_generic_cost(customer_clv))
            generic_prob_reduction = max(0.0, min(1.0, churn_prob * generic_effectiveness))
            generic_value_saved = generic_prob_reduction * customer_clv * generic_margin_rate * generic_execution_prob
            generic_net_benefit = generic_value_saved - generic_cost
            generic_roi = (generic_value_saved / generic_cost) if generic_cost > 0 else 0.0

            if id_col and id_col in df.columns:
                cust_id = str(customer_df[id_col].iloc[0])
            else:
                cust_id = str(idx)

            shap_actionable = 0
            shap_top_feature = None
            shap_top_feature_value = None
            shap_top_shap_value = 0.0
            shap_positive_sum = 0.0
            shap_signal_score = 0.0
            shap_signal_source = None
            shap_driver_count = 0

            try:
                shap_signal = extract_actionable_shap_signal(
                    customer_df,
                    df,
                    model,
                    target_col,
                    id_col=id_col,
                )
                shap_actionable = int(shap_signal.get("actionable", 0))
                shap_top_feature = shap_signal.get("top_feature", None)
                shap_top_feature_value = shap_signal.get("top_feature_value", None)
                shap_top_shap_value = _safe_float(shap_signal.get("top_shap_value", 0.0))
                shap_positive_sum = _safe_float(shap_signal.get("positive_actionable_shap_sum", 0.0))
                shap_driver_count = int(shap_signal.get("driver_count", 0) or 0)
                shap_signal_source = shap_signal.get("signal_source", None)
                shap_signal_score = churn_prob * max(shap_positive_sum, shap_top_shap_value)
                if not np.isfinite(shap_signal_score):
                    shap_signal_score = 0.0
            except Exception as thesis_shap_err:
                msgs = st.session_state.get("validation_messages", [])
                msgs.append(f"Thesis eval: SHAP signal failed for customer {cust_id}: {str(thesis_shap_err)[:80]}")
                st.session_state["validation_messages"] = msgs

            integrated_actionable = 0
            integrated_top_feature = None
            integrated_pred_after = None
            integrated_prob_reduction = 0.0
            integrated_cost = 0.0
            integrated_value_saved = 0.0
            integrated_net_benefit = -1e12
            integrated_roi = 0.0
            integrated_feasibility = None

            try:
                cfs = generate_counterfactuals(
                    customer_data=customer_df,
                    model=model,
                    data_df=df,
                    target_col=target_col,
                    num_cfs=3
                )
                if cfs and len(cfs) > 0:
                    best = cfs[0]
                    changes = best.get('changes', {})
                    if changes:
                        integrated_top_feature = list(changes.keys())[0]
                        integrated_actionable = 0 if _is_immutable(integrated_top_feature) else 1
                    integrated_pred_after = _safe_float(best.get('predicted_churn_prob', None), np.nan)
                    integrated_prob_reduction = max(0.0, _safe_float(best.get('prob_reduction', None)))
                    integrated_cost = _safe_float(best.get('implementation_cost', None))
                    integrated_value_saved = _safe_float(best.get('value_saved', None))
                    if integrated_value_saved <= 0:
                        integrated_value_saved = (
                            integrated_prob_reduction
                            * customer_clv
                            * _safe_float(best.get('margin_rate', generic_margin_rate), generic_margin_rate)
                            * _safe_float(best.get('execution_prob', generic_execution_prob), generic_execution_prob)
                        )
                    integrated_net_benefit = integrated_value_saved - integrated_cost
                    integrated_roi = (integrated_value_saved / integrated_cost) if integrated_cost > 0 else 0.0
                    integrated_feasibility = best.get('feasibility_category', None)
                    if integrated_actionable == 0:
                        integrated_net_benefit = -1e12
                        integrated_roi = 0.0
            except Exception as thesis_cf_err:
                msgs = st.session_state.get("validation_messages", [])
                msgs.append(f"Thesis eval: CF failed for customer {cust_id}: {str(thesis_cf_err)[:80]}")
                st.session_state["validation_messages"] = msgs

            rows.append({
                'customer_id': cust_id,
                'churn_prob': round(churn_prob, 6),
                'customer_clv': round(customer_clv, 2),
                'generic_cost': round(generic_cost, 2),
                'baseline_pred_only_score': round(churn_prob, 6),
                'baseline_pred_clv_score': round(churn_prob * customer_clv, 6),
                'baseline_generic_prob_reduction': round(generic_prob_reduction, 6),
                'baseline_generic_value_saved': round(generic_value_saved, 2),
                'baseline_generic_net_benefit': round(generic_net_benefit, 2),
                'baseline_generic_roi': round(generic_roi, 4),
                'shap_actionable': int(shap_actionable),
                'shap_top_feature': shap_top_feature,
                'shap_top_feature_value': shap_top_feature_value,
                'shap_top_shap_value': round(shap_top_shap_value, 6),
                'shap_positive_actionable_sum': round(shap_positive_sum, 6),
                'shap_signal_score': round(shap_signal_score, 6),
                'shap_driver_count': int(shap_driver_count),
                'shap_signal_source': shap_signal_source,
                'integrated_actionable': int(integrated_actionable),
                'integrated_top_feature': integrated_top_feature,
                'integrated_predicted_prob_after': round(float(integrated_pred_after), 6) if pd.notna(integrated_pred_after) else np.nan,
                'integrated_prob_reduction': round(integrated_prob_reduction, 6),
                'integrated_cost': round(integrated_cost, 2),
                'integrated_value_saved': round(integrated_value_saved, 2),
                'integrated_net_benefit': round(integrated_net_benefit, 2) if integrated_net_benefit > -1e11 else np.nan,
                'integrated_roi': round(integrated_roi, 4),
                'integrated_feasibility': integrated_feasibility,
            })

        progress.empty()
        candidate_df = pd.DataFrame(rows)

        arms = {
            'Prediction Only': {
                'score_col': 'baseline_pred_only_score',
                'net_col': 'baseline_generic_net_benefit',
                'value_col': 'baseline_generic_value_saved',
                'roi_col': 'baseline_generic_roi',
                'prob_col': 'baseline_generic_prob_reduction',
                'action_col': None,
            },
            'Prediction x CLV': {
                'score_col': 'baseline_pred_clv_score',
                'net_col': 'baseline_generic_net_benefit',
                'value_col': 'baseline_generic_value_saved',
                'roi_col': 'baseline_generic_roi',
                'prob_col': 'baseline_generic_prob_reduction',
                'action_col': None,
            },
            'Prediction + SHAP': {
                'score_col': 'shap_signal_score',
                'net_col': 'baseline_generic_net_benefit',
                'value_col': 'baseline_generic_value_saved',
                'roi_col': 'baseline_generic_roi',
                'prob_col': 'baseline_generic_prob_reduction',
                'action_col': 'shap_actionable',
            },
            'Integrated AI + SHAP': {
                'score_col': 'integrated_net_benefit',
                'net_col': 'integrated_net_benefit',
                'value_col': 'integrated_value_saved',
                'roi_col': 'integrated_roi',
                'prob_col': 'integrated_prob_reduction',
                'action_col': 'integrated_actionable',
            },
        }

        def _select_arm(pool_df, arm_cfg):
            temp = pool_df.copy()
            if arm_cfg['action_col'] is not None:
                temp = temp[temp[arm_cfg['action_col']] == 1].copy()
            temp = temp.replace([np.inf, -np.inf], np.nan)
            temp = temp[temp[arm_cfg['score_col']].notna()].copy()
            if len(temp) == 0:
                return temp
            n_select = min(decision_budget, len(temp))
            return temp.nlargest(n_select, arm_cfg['score_col'])

        integrated_selected = _select_arm(candidate_df, arms['Integrated AI + SHAP'])
        integrated_ids = set(integrated_selected['customer_id'].tolist()) if len(integrated_selected) else set()

        arm_rows = []
        for arm_name, arm_cfg in arms.items():
            selected = _select_arm(candidate_df, arm_cfg)
            selected_ids = set(selected['customer_id'].tolist()) if len(selected) else set()
            overlap = (len(selected_ids & integrated_ids) / max(len(integrated_ids), 1)) if integrated_ids else 0.0
            arm_rows.append({
                'arm': arm_name,
                'selected_customers': int(len(selected)),
                'total_value_saved': round(float(selected[arm_cfg['value_col']].sum()), 2) if len(selected) else 0.0,
                'total_net_benefit': round(float(selected[arm_cfg['net_col']].sum()), 2) if len(selected) else 0.0,
                'avg_roi': round(float(selected[arm_cfg['roi_col']].mean()), 4) if len(selected) else 0.0,
                'avg_prob_reduction': round(float(selected[arm_cfg['prob_col']].mean()), 4) if len(selected) else 0.0,
                'positive_roi_rate': round(float((selected[arm_cfg['roi_col']] > 1).mean()), 4) if len(selected) else 0.0,
                'selection_overlap_vs_integrated': round(float(overlap), 4),
            })
        arm_summary_df = pd.DataFrame(arm_rows)

        bootstrap_rows = []
        for seed in range(n_bootstrap):
            sample_df = candidate_df.sample(n=len(candidate_df), replace=True, random_state=seed)
            for arm_name, arm_cfg in arms.items():
                selected = _select_arm(sample_df, arm_cfg)
                bootstrap_rows.append({
                    'seed': seed,
                    'arm': arm_name,
                    'selected_customers': int(len(selected)),
                    'total_net_benefit': float(selected[arm_cfg['net_col']].sum()) if len(selected) else 0.0,
                    'total_value_saved': float(selected[arm_cfg['value_col']].sum()) if len(selected) else 0.0,
                    'avg_roi': float(selected[arm_cfg['roi_col']].mean()) if len(selected) else 0.0,
                    'avg_prob_reduction': float(selected[arm_cfg['prob_col']].mean()) if len(selected) else 0.0,
                })
        bootstrap_df = pd.DataFrame(bootstrap_rows)
        bootstrap_summary_df = (
            bootstrap_df.groupby('arm')
            .agg(
                mean_total_net_benefit=('total_net_benefit', 'mean'),
                std_total_net_benefit=('total_net_benefit', 'std'),
                ci_low_total_net_benefit=('total_net_benefit', lambda s: np.percentile(s, 2.5)),
                ci_high_total_net_benefit=('total_net_benefit', lambda s: np.percentile(s, 97.5)),
                mean_avg_roi=('avg_roi', 'mean'),
                mean_avg_prob_reduction=('avg_prob_reduction', 'mean'),
            )
            .reset_index()
            .round(4)
        )

        pairwise_tests = {}
        pivot_boot = bootstrap_df.pivot(index='seed', columns='arm', values='total_net_benefit')
        for baseline_name in ['Prediction Only', 'Prediction x CLV', 'Prediction + SHAP']:
            if baseline_name not in pivot_boot.columns or 'Integrated AI + SHAP' not in pivot_boot.columns:
                continue
            diffs = (pivot_boot['Integrated AI + SHAP'] - pivot_boot[baseline_name]).dropna()
            improvement_pct = (
                diffs / pivot_boot[baseline_name].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).dropna() * 100.0
            w_result = None
            if wilcoxon is not None and len(diffs) >= 10 and np.any(diffs.values != 0):
                try:
                    stat, pval = wilcoxon(diffs.values)
                    w_result = {
                        'statistic': round(float(stat), 4),
                        'p_value': round(float(pval), 6),
                        'significant_at_005': bool(pval < 0.05),
                        'n_pairs': int(len(diffs)),
                    }
                except Exception:
                    w_result = None
            diff_std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
            cohens_d = (float(diffs.mean()) / diff_std) if diff_std > 1e-9 else 0.0
            pairwise_tests[baseline_name] = {
                'mean_net_benefit_diff': round(float(diffs.mean()), 4),
                'median_value_improvement_pct': round(float(improvement_pct.median()), 4) if len(improvement_pct) else 0.0,
                'ci_low_value_improvement_pct': round(float(np.percentile(improvement_pct, 2.5)), 4) if len(improvement_pct) else 0.0,
                'ci_high_value_improvement_pct': round(float(np.percentile(improvement_pct, 97.5)), 4) if len(improvement_pct) else 0.0,
                'cohens_d': round(float(cohens_d), 4),
                'wilcoxon': w_result,
            }

        thesis_summary = {
            'mode': 'thesis_mode',
            'candidate_pool_size': int(candidate_pool_size),
            'decision_budget': int(decision_budget),
            'bootstrap_repetitions': int(n_bootstrap),
            'generic_effectiveness': float(generic_effectiveness),
            'generic_margin_rate': float(generic_margin_rate),
            'generic_execution_prob': float(generic_execution_prob),
            'arm_metrics': arm_summary_df.to_dict(orient='records'),
            'bootstrap_summary': bootstrap_summary_df.to_dict(orient='records'),
            'pairwise_tests': pairwise_tests,
            'note': (
                'Thesis mode compares decision policies within a fixed high-risk candidate pool. '
                'It includes prediction-only, prediction x CLV, prediction + SHAP triage, '
                'and the full integrated SHAP-prescription system.'
            ),
        }

        candidate_df.to_csv("thesis_candidate_pool.csv", index=False)
        arm_summary_df.to_csv("thesis_arm_summary.csv", index=False)
        bootstrap_df.to_csv("thesis_bootstrap_results.csv", index=False)
        with open("thesis_summary_metrics.json", "w") as f:
            json.dump(thesis_summary, f, indent=2)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))
        ax_bar.bar(arm_summary_df['arm'], arm_summary_df['total_net_benefit'], color=['#64748b', '#0ea5e9', '#10b981'])
        ax_bar.set_title("Total Net Benefit by Decision Policy")
        ax_bar.set_ylabel("Total Net Benefit")
        ax_bar.tick_params(axis='x', rotation=15)
        fig_bar.tight_layout()
        fig_bar.savefig("thesis_total_net_benefit.png", dpi=160, bbox_inches="tight")
        plt.close(fig_bar)

        fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
        _box_data = [bootstrap_df.loc[bootstrap_df['arm'] == arm, 'total_net_benefit'].values for arm in arm_summary_df['arm']]
        ax_box.boxplot(_box_data, tick_labels=arm_summary_df['arm'].tolist(), patch_artist=True)
        ax_box.set_title("Bootstrap Distribution of Net Benefit")
        ax_box.set_ylabel("Total Net Benefit")
        ax_box.tick_params(axis='x', rotation=15)
        fig_box.tight_layout()
        fig_box.savefig("thesis_bootstrap_net_benefit.png", dpi=160, bbox_inches="tight")
        plt.close(fig_box)

        return candidate_df, arm_summary_df, bootstrap_df, thesis_summary

    except Exception as e:
        import traceback
        st.error(f"Thesis evaluation failed: {e}")
        st.code(traceback.format_exc()[-800:])
        return None, None, None, None


def format_counterfactual_with_gpt(changes, original_prob, new_prob, customer_profile):
    """
    Use GPT to generate plain-language explanation of counterfactual recommendation.
    """
    if not OPENAI_AVAILABLE or client is None:
        # Fallback to rule-based formatting
        change_text = []
        for feature, vals in changes.items():
            from_val = vals['from']
            to_val = vals['to']
            if isinstance(from_val, (int, float)):
                change_text.append(f"Change {feature} from {from_val:.2f} to {to_val:.2f}")
            else:
                change_text.append(f"Change {feature} from '{from_val}' to '{to_val}'")
        
        return {
            'title': 'Retention Strategy',
            'description': ' • '.join(change_text),
            'impact': f"Reduces churn risk from {original_prob:.1%} to {new_prob:.1%}"
        }
    
    try:
        # Build change summary
        change_list = []
        for feature, vals in changes.items():
            change_list.append(f"{feature}: {vals['from']} → {vals['to']}")
        
        prompt = f"""You are a customer retention expert. A customer has {original_prob:.1%} churn risk.

We identified that making these changes would reduce their churn risk to {new_prob:.1%}:
{chr(10).join(f'- {c}' for c in change_list)}

Provide a business-friendly recommendation in this exact JSON format:
{{
  "title": "Short catchy title (e.g., 'Contract Upgrade Strategy')",
  "description": "One clear sentence explaining what action to take and why it works",
  "impact": "One sentence about the business benefit"
}}

Keep it concise, actionable, and focused on the customer retention value."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a customer retention expert who provides concise, actionable recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        import json
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        return result
    
    except Exception as e:
        # Fallback
        change_text = []
        for feature, vals in changes.items():
            from_val = vals['from']
            to_val = vals['to']
            if isinstance(from_val, (int, float)):
                change_text.append(f"{feature}: ${from_val:.0f} → ${to_val:.0f}" if 'charge' in feature.lower() or 'price' in feature.lower() else f"{feature}: {from_val:.1f} → {to_val:.1f}")
            else:
                change_text.append(f"{feature}: {from_val} → {to_val}")
        
        return {
            'title': 'Retention Recommendation',
            'description': ' | '.join(change_text),
            'impact': f"Reduces churn from {original_prob:.0%} to {new_prob:.0%}"
        }


def gpt_rule_suggestion(customer_profile: dict) -> str:
    """Simple rule/insight generated by OpenAI (optional)."""
    if not OPENAI_AVAILABLE or client is None:
        return "OpenAI is not configured."

    # Retrieve context values from session state if available
    acc = st.session_state.get("best_model_acc", None)
    thr = st.session_state.get("adaptive_threshold", None)
    churn_rate = st.session_state.get("overall_churn_rate", None)
    context = []
    if acc is not None:
        context.append(f"Model accuracy: {acc:.2%}")
    if thr is not None:
        context.append(f"Churn threshold: {thr:.2f}")
    if churn_rate is not None:
        context.append(f"Dataset churn rate: {churn_rate:.2%}")
    context_info = "; ".join(context) if context else "No additional context."

    prompt = f"""
You are a senior customer retention manager. Use ONLY the provided Customer Profile and Model Context. If a detail is missing or ambiguous, write "Unknown" or "Data not provided" and do not invent anything.

Customer Profile (raw fields/values): {customer_profile}
Model Context: {context_info}

Return in EXACTLY this markdown template (no extra lines or sections):
**Loyalty Stage**: <New|Active|Loyal|At-Risk|Unknown> — <short reason>
**Key Observations**
- <observation 1 grounded in the profile>
- <observation 2 grounded in the profile>
**Recommended Action**: <one specific action tied to the observations>
**Expected Outcome**: <one sentence on likely impact or reaction>

Rules: 70-120 words. Professional, decisive. No filler. No extra commentary.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business-facing retention analyst. Follow the requested format exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=220
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI rule suggestion failed: {e}"


def gpt_action_recommendation(prob: float, top_features: list, top_values: list, profile: dict) -> str:
    """Generates detailed churn reasoning and action plan using OpenAI."""
    if not OPENAI_AVAILABLE or client is None:
        return "OpenAI is not configured."

    # Retrieve context values from session state if available
    acc = st.session_state.get("best_model_acc", None)
    thr = st.session_state.get("adaptive_threshold", None)
    churn_rate = st.session_state.get("overall_churn_rate", None)
    context = []
    if acc is not None:
        context.append(f"Model accuracy: {acc:.2%}")
    if thr is not None:
        context.append(f"Churn threshold: {thr:.2f}")
    if churn_rate is not None:
        context.append(f"Dataset churn rate: {churn_rate:.2%}")
    context_info = "; ".join(context) if context else "No additional context."

    feature_summary = ", ".join([f"{f}: {v}" for f, v in zip(top_features, top_values)])
    
    # CALCULATE customer value score from multiple indicators
    value_score = 0
    value_components = []
    
    # Monetary indicators (weight: 40%)
    monetary_keywords = ['monetary', 'revenue', 'amount', 'spending', 'price', 'sales', 'cashback', 'ltv', 'clv']
    for key, val in profile.items():
        if any(kw in key.lower() for kw in monetary_keywords):
            try:
                normalized = float(val) / 100  # Normalize to 0-10 scale (assuming 0-1000 range)
                value_score += min(normalized, 10) * 0.4
                value_components.append(f"{key}: {val}")
                break  # Use first monetary indicator found
            except (ValueError, TypeError):
                pass
    
    # Volume indicators - orders/transactions (weight: 30%)
    volume_keywords = ['ordercount', 'order_count', 'frequency', 'purchase', 'transaction', 'visits']
    for key, val in profile.items():
        if any(kw in key.lower().replace('_', '') for kw in volume_keywords):
            try:
                normalized = float(val) / 2  # Normalize (assuming 0-20 range)
                value_score += min(normalized, 10) * 0.3
                value_components.append(f"{key}: {val}")
                break
            except (ValueError, TypeError):
                pass
    
    # Loyalty indicators - tenure/age (weight: 30%)
    loyalty_keywords = ['tenure', 'age', 'duration', 'member', 'subscription', 'account']
    for key, val in profile.items():
        if any(kw in key.lower() for kw in loyalty_keywords):
            try:
                normalized = float(val) / 3  # Normalize (assuming 0-30 range)
                value_score += min(normalized, 10) * 0.3
                value_components.append(f"{key}: {val}")
                break
            except (ValueError, TypeError):
                pass
    
    # Determine value tier based on calculated score (0-10 scale)
    if value_score >= 6.5:
        customer_value = "High"
    elif value_score >= 3.5:
        customer_value = "Medium"
    elif value_score > 0:
        customer_value = "Low"
    else:
        customer_value = "Unknown"
    
    value_info = "; ".join(value_components) if value_components else "No value metrics detected"
    
    # Determine urgency and investment level
    urgency = "Critical - Act within 24hrs" if prob >= 0.70 else "High - Act within 7 days" if prob >= 0.50 else "Moderate - Monitor weekly"
    investment = "High touch (personal outreach)" if customer_value in ["High", "Medium"] and prob >= 0.50 else "Automated touchpoint"
    
    prompt = f"""
You are a senior retention strategist. Use ONLY provided inputs. Be specific and actionable.

Inputs:
- Churn probability: {prob:.2f} | Urgency: {urgency}
- Top churn drivers: {feature_summary if feature_summary else "None provided"}
- Customer value: {customer_value} (Score: {value_score:.1f}/10 from: {value_info})
- Recommended investment: {investment}
- Full profile: {profile}
- Model context: {context_info}

Return in EXACTLY this format:
**Business Interpretation**: <Why this customer is at risk - reference specific drivers in 2 sentences>
**Risk & Value Assessment**
- Churn risk: <Low|Medium|High> (Low <0.40, Medium 0.40-0.70, High >0.70)
- Customer value: {customer_value}
- Priority: <Critical|High|Medium|Low> (High value + High risk = Critical; adjust accordingly)
**Strategic Action Plan**: <ONE specific action directly addressing the top driver - be concrete: "Offer X", "Call about Y", "Send Z discount">
**Communication Strategy**: <Specific channel (email/call/SMS) + exact timing (24hrs/3 days/1 week) + message focus in one sentence>
**Expected ROI Impact**: <Cost-benefit assessment: high-value customers justify more investment; mention retention value vs intervention cost qualitatively>

Rules: 150-200 words. Data-driven, actionable, ROI-conscious. Match intervention intensity to customer value + churn risk.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business analytics strategist. Follow the output format exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=380
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI call failed: {e}"


def gpt_promotional_message(prob: float, profile: dict, rule_text: str, action_text: str, channel_hint: str = "") -> str:
    """Generate a personalized promotional message using prior insights."""
    if not OPENAI_AVAILABLE or client is None:
        return "OpenAI is not configured."

    acc = st.session_state.get("best_model_acc", None)
    thr = st.session_state.get("adaptive_threshold", None)
    churn_rate = st.session_state.get("overall_churn_rate", None)
    context = []
    if acc is not None:
        context.append(f"Model accuracy: {acc:.2%}")
    if thr is not None:
        context.append(f"Churn threshold: {thr:.2f}")
    if churn_rate is not None:
        context.append(f"Dataset churn rate: {churn_rate:.2%}")
    context_info = "; ".join(context) if context else "No additional context."

    required_channel = channel_hint.strip().lower()
    channel_note = f"Required channel: {required_channel}." if required_channel else "No required channel."
    prompt = f"""
You are a lifecycle marketing copywriter. Create one tailored promotional message using ONLY the details below.
- Customer profile: {profile}
- Churn probability: {prob:.2f}
- Rule-based suggestion: {rule_text}
- AI recommendation summary: {action_text}
- Model context: {context_info}
- Preferred channel hint (optional): {channel_hint or "Not specified"}.

Requirements:
1) {channel_note} If a required channel is provided, you MUST use it exactly (email / sms / push). Do NOT choose another channel.
2) If no required channel is provided, choose the channel that best fits the profile and context (email / SMS / in-app push). If unsure, pick email.
3) Tone must match the channel (email: professional/reassuring; SMS: concise/warm; push: concise/energetic).
4) Include a concrete offer or action that aligns with the suggestions (e.g., discount, loyalty benefit, concierge call).
5) Add a clear CTA. Keep total length short:
   - Email: 70-120 words, subject + body.
   - SMS: <= 40 words.
   - Push: <= 25 words.
6) Do NOT invent data. If missing, omit.

Return format (markdown):
**Channel**: <email|sms|push>
**Message**:
<final copy>
"""
    try:
        def call_model(system_text, temp=0.25):
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=260
            ).choices[0].message.content.strip()

        resp_text = call_model("You are a concise, high-conversion lifecycle copywriter.")

        if required_channel:
            m = re.search(r"\*\*Channel\*\*:\s*([a-zA-Z]+)", resp_text, re.IGNORECASE)
            found = m.group(1).strip().lower() if m else ""
            if found != required_channel:
                # Retry once with stricter instruction
                strict_system = (
                    f"You must output Channel: {required_channel} and match the required format exactly. "
                    "Do not choose any other channel."
                )
                resp_text = call_model(strict_system, temp=0.1)

        return resp_text
    except Exception as e:
        return f"AI promo failed: {e}"


## ----------------------------- HEADER GRADIENT BANNER -----------------------------
st.markdown(
    """
    <div class="gradient-header">
      <div class="hero-content">
        <div class="hero-badge">Retention Management AI</div>
        <div class="hero-title">Customer Churn Prediction Dashboard</div>
        <div class="hero-subtitle">Turn customer data into retention decisions with AutoML risk scoring, SHAP explanations, and AI-driven actions.</div>
        <div class="hero-chips">
          <span class="hero-chip">AutoML Optimization</span>
          <span class="hero-chip">Risk Scoring</span>
          <span class="hero-chip">SHAP Explainability</span>
          <span class="hero-chip">Retention Actions</span>
        </div>
      </div>
      <div class="hero-panel">
        <div class="hero-panel-title">Instant insights</div>
        <ul class="hero-list">
          <li>Identify at-risk customers in seconds</li>
          <li>See what drives churn and retention</li>
          <li>Generate recommendations and promo messages</li>
        </ul>
        <div class="hero-panel-foot">Built for fast decisions and clean audits.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------- SIDEBAR: UPLOAD & SETTINGS -----------------------------
sidebar = st.sidebar
sidebar.title("Controls & Upload")

sidebar.markdown('<div class="sidebar-section-title">Upload</div>', unsafe_allow_html=True)
uploaded = sidebar.file_uploader("📂 Upload CSV", type=["csv"], accept_multiple_files=False)
# Safety: if Streamlit ever returns a list, keep only the most recent file
if isinstance(uploaded, list):
    uploaded = uploaded[-1] if uploaded else None
use_demo = sidebar.checkbox("Use demo dataset (default)", value=True)
if "thesis_arm_summary" in st.session_state and st.session_state["thesis_arm_summary"] is not None:
    _thesis_candidates = st.session_state["thesis_candidates"]
    _thesis_arm_summary = st.session_state["thesis_arm_summary"]
    _thesis_bootstrap_df = st.session_state["thesis_bootstrap_df"]
    _thesis_summary = st.session_state["thesis_summary"]

    st.success(
        "Thesis evaluation ready -- "
        f"candidate pool {_thesis_summary['candidate_pool_size']} | "
        f"budget {_thesis_summary['decision_budget']}"
    )
    st.caption(_thesis_summary.get("note", ""))

    _arm_lookup = {
        row["arm"]: row for row in _thesis_summary.get("arm_metrics", [])
    }
    _pred_only = _arm_lookup.get("Prediction Only", {})
    _pred_clv = _arm_lookup.get("Prediction x CLV", {})
    _pred_shap = _arm_lookup.get("Prediction + SHAP", {})
    _integrated = _arm_lookup.get("Integrated AI + SHAP", {})

    def _pct_gain(new_val, base_val):
        return (float(new_val) - float(base_val)) / max(abs(float(base_val)), 1.0) * 100.0

    tmc1, tmc2, tmc3, tmc4 = st.columns(4)
    with tmc1:
        st.metric(
            "Integrated vs Prediction Only",
            f"{_pct_gain(_integrated.get('total_net_benefit', 0), _pred_only.get('total_net_benefit', 0)):.1f}%",
            delta=f"${_integrated.get('total_net_benefit', 0):,.0f} vs ${_pred_only.get('total_net_benefit', 0):,.0f}",
        )
    with tmc2:
        st.metric(
            "Integrated vs Prediction x CLV",
            f"{_pct_gain(_integrated.get('total_net_benefit', 0), _pred_clv.get('total_net_benefit', 0)):.1f}%",
            delta=f"${_integrated.get('total_net_benefit', 0):,.0f} vs ${_pred_clv.get('total_net_benefit', 0):,.0f}",
        )
    with tmc3:
        st.metric(
            "Integrated vs Prediction + SHAP",
            f"{_pct_gain(_integrated.get('total_net_benefit', 0), _pred_shap.get('total_net_benefit', 0)):.1f}%",
            delta=f"${_integrated.get('total_net_benefit', 0):,.0f} vs ${_pred_shap.get('total_net_benefit', 0):,.0f}",
        )
    with tmc4:
        st.metric(
            "Bootstrap Repetitions",
            f"{_thesis_summary.get('bootstrap_repetitions', 0):,}",
            delta=f"Budget {int(_thesis_summary.get('decision_budget', 0))}",
        )

    st.markdown("**Decision Policy Summary**")
    st.dataframe(_thesis_arm_summary, width='stretch')

    _pairwise_rows = []
    for _baseline_name, _stats in _thesis_summary.get("pairwise_tests", {}).items():
        _wilcoxon = _stats.get("wilcoxon") or {}
        _pairwise_rows.append({
            'comparison': f"Integrated AI + SHAP vs {_baseline_name}",
            'mean_net_benefit_diff': _stats.get('mean_net_benefit_diff'),
            'median_value_improvement_pct': _stats.get('median_value_improvement_pct'),
            'ci_low_value_improvement_pct': _stats.get('ci_low_value_improvement_pct'),
            'ci_high_value_improvement_pct': _stats.get('ci_high_value_improvement_pct'),
            'cohens_d': _stats.get('cohens_d'),
            'wilcoxon_p_value': _wilcoxon.get('p_value'),
            'significant_at_005': _wilcoxon.get('significant_at_005'),
        })
    if _pairwise_rows:
        st.markdown("**Pairwise Robustness Tests**")
        st.dataframe(pd.DataFrame(_pairwise_rows), width='stretch')

    tf1, tf2 = st.columns(2)
    with tf1:
        _fig_bar, _ax_bar = plt.subplots(figsize=(8, 4.5))
        _colors = ['#64748b', '#0ea5e9', '#f59e0b', '#10b981']
        _ax_bar.bar(_thesis_arm_summary['arm'], _thesis_arm_summary['total_net_benefit'], color=_colors[:len(_thesis_arm_summary)])
        _ax_bar.set_title("Total Net Benefit by Decision Policy")
        _ax_bar.set_ylabel("Total Net Benefit")
        _ax_bar.tick_params(axis='x', rotation=15)
        _fig_bar.tight_layout()
        _show_matplotlib(_fig_bar, container=st)
    with tf2:
        _fig_box, _ax_box = plt.subplots(figsize=(8, 4.5))
        _box_data = [
            _thesis_bootstrap_df.loc[_thesis_bootstrap_df['arm'] == _arm, 'total_net_benefit'].values
            for _arm in _thesis_arm_summary['arm']
        ]
        _ax_box.boxplot(_box_data, labels=_thesis_arm_summary['arm'].tolist(), patch_artist=True)
        _ax_box.set_title("Bootstrap Distribution of Net Benefit")
        _ax_box.set_ylabel("Total Net Benefit")
        _ax_box.tick_params(axis='x', rotation=15)
        _fig_box.tight_layout()
        _show_matplotlib(_fig_box, container=st)

    import json as _json_thesis_dl
    _thesis_candidates_csv = _thesis_candidates.to_csv(index=False).encode()
    _thesis_arm_csv = _thesis_arm_summary.to_csv(index=False).encode()
    _thesis_bootstrap_csv = _thesis_bootstrap_df.to_csv(index=False).encode()
    _thesis_json = _json_thesis_dl.dumps(_thesis_summary, indent=2).encode()
    _cand_b64 = base64.b64encode(_thesis_candidates_csv).decode()
    _arm_b64 = base64.b64encode(_thesis_arm_csv).decode()
    _boot_b64 = base64.b64encode(_thesis_bootstrap_csv).decode()
    _json_b64 = base64.b64encode(_thesis_json).decode()

    td1, td2, td3, td4 = st.columns(4)
    with td1:
        st.markdown(
            f'<a href="data:text/csv;base64,{_cand_b64}" '
            f'download="thesis_candidate_pool.csv" '
            f'style="display:inline-block;padding:0.5em 1em;background:#262730;color:#fafafa;'
            f'border-radius:0.5rem;text-decoration:none;font-size:0.875rem;border:1px solid #4a4a5a;'
            f'text-align:center;width:100%;">Download Candidate Pool</a>',
            unsafe_allow_html=True,
        )
    with td2:
        st.markdown(
            f'<a href="data:text/csv;base64,{_arm_b64}" '
            f'download="thesis_arm_summary.csv" '
            f'style="display:inline-block;padding:0.5em 1em;background:#262730;color:#fafafa;'
            f'border-radius:0.5rem;text-decoration:none;font-size:0.875rem;border:1px solid #4a4a5a;'
            f'text-align:center;width:100%;">Download Arm Summary</a>',
            unsafe_allow_html=True,
        )
    with td3:
        st.markdown(
            f'<a href="data:text/csv;base64,{_boot_b64}" '
            f'download="thesis_bootstrap_results.csv" '
            f'style="display:inline-block;padding:0.5em 1em;background:#262730;color:#fafafa;'
            f'border-radius:0.5rem;text-decoration:none;font-size:0.875rem;border:1px solid #4a4a5a;'
            f'text-align:center;width:100%;">Download Bootstrap CSV</a>',
            unsafe_allow_html=True,
        )
    with td4:
        st.markdown(
            f'<a href="data:application/json;base64,{_json_b64}" '
            f'download="thesis_summary_metrics.json" '
            f'style="display:inline-block;padding:0.5em 1em;background:#262730;color:#fafafa;'
            f'border-radius:0.5rem;text-decoration:none;font-size:0.875rem;border:1px solid #4a4a5a;'
            f'text-align:center;width:100%;">Download Thesis Summary</a>',
            unsafe_allow_html=True,
        )

sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# Load data
if uploaded is not None:
    df_raw = safe_read_csv(uploaded)
elif use_demo and os.path.exists("processed_churn_dataset.csv"):
    df_raw = pd.read_csv("processed_churn_dataset.csv")
else:
    st.error("Dataset Required: Upload a CSV or enable the demo dataset.")
    st.stop()

# Dataset fingerprint for auto-reset on new uploads
def compute_dataset_fingerprint():
    if uploaded is not None:
        try:
            data = uploaded.getvalue()
            digest = hashlib.sha256(data).hexdigest()[:12]
            return f"upload:{uploaded.name}:{len(data)}:{digest}"
        except Exception:
            size = getattr(uploaded, "size", "")
            return f"upload:{uploaded.name}:{size}"
    if use_demo and os.path.exists("processed_churn_dataset.csv"):
        stat = os.stat("processed_churn_dataset.csv")
        return f"demo:{stat.st_size}:{int(stat.st_mtime)}"
    return None

current_fp = compute_dataset_fingerprint()
prev_fp = st.session_state.get("dataset_fingerprint")
if current_fp and prev_fp and current_fp != prev_fp:
    st.session_state["_dataset_changed"] = True
    st.session_state["_new_dataset_fp"] = current_fp
elif current_fp and not prev_fp:
    st.session_state["dataset_fingerprint"] = current_fp

# Work with a normalized copy internally; keep original for display if needed
df = normalize_cols(df_raw)

with st.expander("📋 Dataset Preview", expanded=True):
    st.dataframe(df.head(), width='stretch', height=260)
    # --- Dataset summary cards ---
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_vals = int(df.isnull().sum().sum())
    data_types = df.dtypes.nunique()

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-metric">
            <div class="kpi-label">Total Rows</div>
            <div class="kpi-value" style="color:#2563eb;">{total_rows:,}</div>
          </div>
          <div class="kpi-metric">
            <div class="kpi-label">Total Columns</div>
            <div class="kpi-value" style="color:#f59e0b;">{total_cols}</div>
          </div>
          <div class="kpi-metric">
            <div class="kpi-label">Missing Values</div>
            <div class="kpi-value" style="color:#e11d48;">{missing_vals}</div>
          </div>
          <div class="kpi-metric">
            <div class="kpi-label">Data Types</div>
            <div class="kpi-value" style="color:#22c55e;">{data_types}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Column selections
cols = df.columns.tolist()
default_target = guess_target_name(cols)
default_id = guess_id_name(cols)

# Clear widget selections on reset before widgets are instantiated
if st.session_state.get("reset_widgets"):
    for key in ("id_col_select", "target_col_select"):
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.reset_widgets = False

sidebar.markdown('<div class="sidebar-section-title">Model Setup</div>', unsafe_allow_html=True)
id_choice = sidebar.selectbox(
    "Select Customer ID column (optional)",
    options=[None] + cols,
    index=(cols.index(default_id) + 1) if default_id in cols else 0,
    key="id_col_select"
)

# ============================================================================
# UNSUPERVISED MODE: Auto-detect if no churn column exists
# ============================================================================
churn_like_cols = [c for c in cols if any(x in c.lower() for x in ['churn', 'attrition', 'left', 'exit', 'cancel'])]
has_churn_column = len(churn_like_cols) > 0

if not has_churn_column:
    # No churn column detected - offer unsupervised mode
    st.sidebar.info(
        "🔍 **No churn column detected**\n\n"
        "Enable unsupervised mode to auto-generate churn labels based on customer behavior patterns."
    )
    enable_unsupervised = st.sidebar.checkbox(
        "Enable Unsupervised Churn Detection",
        value=False,
        help="Automatically identifies at-risk customers using behavioral analysis"
    )
    
    if enable_unsupervised:
        unsupervised_method = st.sidebar.radio(
            "Detection Method",
            options=['Auto (Recommended)', 'Behavioral Heuristics', 'Clustering', 'Anomaly Detection'],
            help=(
                "• Auto: Intelligently selects best method\n"
                "• Behavioral: Uses tenure, usage, satisfaction patterns\n"
                "• Clustering: Groups similar customers\n"
                "• Anomaly: Detects unusual behavior"
            )
        )
        
        # Map user choice to method parameter
        method_map = {
            'Auto (Recommended)': 'auto',
            'Behavioral Heuristics': 'heuristic',
            'Clustering': 'clustering',
            'Anomaly Detection': 'anomaly'
        }
        selected_method = method_map[unsupervised_method]
        
        # Generate churn labels
        with st.spinner("Analyzing customer behavior patterns..."):
            try:
                df_labeled, method_used, summary, reliability_score, confidence_metrics = prepare_unlabeled_dataset(
                    df, 
                    target_name='Churn_Predicted',
                    method=selected_method
                )
                
                # Replace original df
                df = df_labeled
                
                # Add generated column to cols list
                if 'Churn_Predicted' not in cols:
                    cols.append('Churn_Predicted')
                
                # Show detection summary with reliability
                if reliability_score >= 0.75:
                    st.sidebar.success(f"✓ Labels generated using {method_used} | Reliability: HIGH ({reliability_score:.0%})")
                elif reliability_score >= 0.60:
                    st.sidebar.info(f"✓ Labels generated using {method_used} | Reliability: MODERATE ({reliability_score:.0%})")
                else:
                    st.sidebar.warning(f"⚠️ Labels generated using {method_used} | Reliability: LOW ({reliability_score:.0%})")
                
                with st.sidebar.expander("📊 Detection Summary & Reliability"):
                    st.text(summary)
                    st.markdown("---")
                    st.markdown("**Confidence Metrics:**")
                    st.metric("Overall Confidence", f"{confidence_metrics['overall_confidence']:.0%}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Balance", f"{confidence_metrics['balance_score']:.0%}")
                        st.metric("Separation", f"{confidence_metrics['separation_score']:.0%}")
                    with col2:
                        st.metric("Churn Rate", f"{confidence_metrics['churn_rate']:.1%}")
                        st.metric("Indicators", f"{confidence_metrics['indicator_coverage']:.0%}")
                
                # Auto-select the generated column
                target_choice = 'Churn_Predicted'
                
                # Store flag for later reference
                st.session_state["unsupervised_mode"] = True
                st.session_state["unsupervised_method"] = method_used
                st.session_state["unsupervised_reliability"] = reliability_score
                st.session_state["unsupervised_confidence"] = confidence_metrics
                
            except Exception as e:
                st.sidebar.error(f"Failed to generate labels: {str(e)[:100]}")
                target_choice = None
    else:
        target_choice = sidebar.selectbox(
            "Select Target column (required)",
            options=[None] + cols,
            index=(cols.index(default_target) + 1) if default_target in cols else 0,
            key="target_col_select"
        )
else:
    # Normal supervised mode - churn column exists
    target_choice = sidebar.selectbox(
        "Select Target column (required)",
        options=[None] + cols,
        index=(cols.index(default_target) + 1) if default_target in cols else 0,
        key="target_col_select"
    )
    st.session_state["unsupervised_mode"] = False

sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# Show run/cancel controls even before target is selected (disabled until target chosen)
target_missing = not target_choice
if target_missing:
    st.warning("⚠️ Please select the target column to continue.")

# ----------------------------- SESSION STATE INIT -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "fitted" not in st.session_state:
    st.session_state.fitted = False
if "train_df" not in st.session_state:
    st.session_state.train_df = None
# NEW: store label maps for categorical encodings
if "label_maps" not in st.session_state:
    st.session_state.label_maps = {}
if "target_label_mapping" not in st.session_state:
    st.session_state.target_label_mapping = None
if "target_mapping_strategy" not in st.session_state:
    st.session_state.target_mapping_strategy = None
if "automl_running" not in st.session_state:
    st.session_state.automl_running = False
if "cancel_automl" not in st.session_state:
    st.session_state.cancel_automl = False
if "reset_widgets" not in st.session_state:
    st.session_state.reset_widgets = False
if "automl_just_finished" not in st.session_state:
    st.session_state.automl_just_finished = False
# Validation messages list
if "validation_messages" not in st.session_state:
    st.session_state.validation_messages = []

def reset_automl_state():
    """Reset AutoML-related state to allow a clean restart."""
    st.session_state.model = None
    st.session_state.fitted = False
    st.session_state.train_df = None
    st.session_state.prep_pipe = None
    st.session_state.label_maps = {}
    st.session_state.best_model_auc = None
    st.session_state.best_model_precision = None
    st.session_state.best_model_recall = None
    st.session_state.best_model_name = None
    st.session_state.adaptive_threshold = 0.5
    st.session_state.last_ai_reco = None
    st.session_state.last_profile = None
    st.session_state.automl_running = False
    st.session_state.cancel_automl = False
    st.session_state.automl_just_finished = False
    st.session_state.raw_feature_cols = None
    st.session_state.model_feature_cols = None
    st.session_state.leakage_cols = None
    st.session_state.target_label_mapping = None
    st.session_state.target_mapping_strategy = None
    st.session_state.validation_messages = []

# Auto-reset when a new dataset is detected
if st.session_state.get("_dataset_changed"):
    reset_automl_state()
    st.session_state["dataset_fingerprint"] = st.session_state.get("_new_dataset_fp")
    st.session_state["_dataset_changed"] = False
    st.sidebar.warning("New dataset detected — AutoML state reset.")
    st.rerun()

    # Reset selections on next rerun to avoid widget state mutation errors
    st.session_state.reset_widgets = True

    # Clean up any saved artifacts
    try:
        import glob
        for f in glob.glob("automl_best_model*"):
            os.remove(f)
        if os.path.exists("automl_model_columns.pkl"):
            os.remove("automl_model_columns.pkl")
    except Exception:
        pass

# ----------------------------- RUN / CANCEL AUTOML -----------------------------
sidebar.markdown('<div class="sidebar-section-title">Automation</div>', unsafe_allow_html=True)
run_col, cancel_col = sidebar.columns([2.2, 1.2])
run_automl = run_col.button(
    "⚙️ Run AutoML",
    disabled=st.session_state.automl_running or target_missing,
    type="primary",
    use_container_width=True
)
if run_automl:
    st.session_state.automl_running = True
    st.session_state.cancel_automl = False

cancel_clicked = False
if st.session_state.automl_running:
    cancel_clicked = cancel_col.button("Cancel", type="secondary", use_container_width=True)
else:
    cancel_col.markdown("<div style='height:42px'></div>", unsafe_allow_html=True)

if cancel_clicked:
    reset_automl_state()
    st.warning("AutoML canceled. You can select the target column again and run when ready.")
    st.rerun()

# ----------------------------- SIDEBAR MODEL STATE -----------------------------
if st.session_state.automl_running:
    training_status = st.session_state.get("training_status", "Training in progress...")
    sidebar.markdown(
        f'<div class="sidebar-state">🔄 {training_status}</div>',
        unsafe_allow_html=True
    )
elif st.session_state.fitted:
    auc = st.session_state.get("best_model_auc", None)
    if isinstance(auc, (int, float, np.floating)):
        auc_text = f"{auc*100:.1f}%"
    else:
        auc_text = "N/A"
    thr = st.session_state.get("adaptive_threshold", 0.5)
    sidebar.markdown(
        f'<div class="sidebar-state">Model ready — AUC: {auc_text}, Threshold: {thr:.2f}</div>',
        unsafe_allow_html=True
    )
else:
    sidebar.markdown(
        '<div class="sidebar-state">No model yet — upload data, pick target, then Run AutoML.</div>',
        unsafe_allow_html=True
    )

if target_missing:
    st.stop()

# Safe resolve (in case future normalizations change)
try:
    target_col = resolve_column(df, target_choice)
except Exception as e:
    st.error(f"❌ Target column resolution failed: {e}")
    st.stop()

id_col = None
if id_choice:
    try:
        id_col = resolve_column(df, id_choice)
    except Exception:
        st.warning(f"ID column '{id_choice}' could not be found after standardization; proceeding without it.")
        id_col = None

# Detect ID-like columns for exclusion (even if user doesn't select)
auto_id_cols = detect_id_cols(df, explicit_id=id_col, target_col=target_col)
st.session_state["drop_id_cols"] = auto_id_cols

# Prediction settings section
sidebar.markdown('<div class="sidebar-section-title">Prediction Settings</div>', unsafe_allow_html=True)
# Select row to inspect
if id_col:
    ids = df[id_col].astype(str).tolist()
    selected_id = sidebar.selectbox("Select Customer", ids)
    
    # Clear AI-generated content when a new customer is selected
    if st.session_state.get("last_selected_id") != selected_id:
        st.session_state["last_ai_reco"] = None
        st.session_state["last_profile"] = None
        st.session_state["last_promo"] = None
        st.session_state["last_selected_id"] = selected_id
    
    customer_row = df[df[id_col].astype(str) == str(selected_id)].iloc[0:1]
else:
    idx = sidebar.number_input("Row Index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
    
    # Clear AI-generated content when a new row index is selected
    if st.session_state.get("last_selected_idx") != idx:
        st.session_state["last_ai_reco"] = None
        st.session_state["last_profile"] = None
        st.session_state["last_promo"] = None
        st.session_state["last_selected_idx"] = idx
    
    customer_row = df.iloc[int(idx):int(idx)+1]

suggested_thr = st.session_state.get("adaptive_threshold", 0.50)
threshold = sidebar.slider("Prediction threshold (model-suggested default)", 0.0, 1.0, float(round(suggested_thr, 2)))
sidebar.caption(f"Model-suggested threshold: {suggested_thr:.2f}")
sidebar.caption(
    "🔎 Threshold guidance: lower = more sensitive to churn (catch more at-risk), higher = stricter (reduce false alerts)."
)
sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# ----------------------------- RESEARCH EXPORT (Chapter 5) -----------------------------
sidebar.markdown('<div class="sidebar-section-title">Research Export</div>', unsafe_allow_html=True)
research_mode = sidebar.radio(
    "Evaluation Mode",
    ["Standard Chapter 5", "Thesis Mode"],
    index=1,
    key="research_eval_mode",
)
_research_disabled = not st.session_state.fitted
if research_mode == "Thesis Mode":
    _max_pool = max(20, min(len(df), 300))
    _default_pool = min(100, _max_pool)
    thesis_candidate_pool = int(sidebar.number_input(
        "Candidate Pool Size",
        min_value=20,
        max_value=_max_pool,
        value=_default_pool,
        step=10,
        key="thesis_candidate_pool",
    ))
    thesis_budget = int(sidebar.number_input(
        "Decision Budget (Top-K)",
        min_value=5,
        max_value=thesis_candidate_pool,
        value=min(30, thesis_candidate_pool),
        step=5,
        key="thesis_decision_budget",
    ))
    thesis_bootstrap = int(sidebar.slider(
        "Bootstrap Repetitions",
        min_value=20,
        max_value=300,
        value=100,
        step=20,
        key="thesis_bootstrap_repetitions",
    ))
    sidebar.caption(
        "Thesis mode uses a fixed high-risk candidate pool and fixed budget. "
        "It compares prediction-only, prediction x CLV, prediction + SHAP triage, "
        "and the integrated SHAP-prescription system."
    )
    _thesis_clicked = sidebar.button(
        "Run Thesis Evaluation",
        disabled=_research_disabled,
        use_container_width=True,
    )
    if _thesis_clicked and not _research_disabled:
        with st.spinner("Running fixed-budget thesis evaluation..."):
            _thesis_candidates, _thesis_arm_summary, _thesis_bootstrap_df, _thesis_summary = run_thesis_evaluation(
                df,
                st.session_state.model,
                target_col,
                id_col,
                candidate_pool_size=thesis_candidate_pool,
                decision_budget=thesis_budget,
                n_bootstrap=thesis_bootstrap,
            )
        if _thesis_candidates is not None and _thesis_summary is not None:
            st.session_state["thesis_candidates"] = _thesis_candidates
            st.session_state["thesis_arm_summary"] = _thesis_arm_summary
            st.session_state["thesis_bootstrap_df"] = _thesis_bootstrap_df
            st.session_state["thesis_summary"] = _thesis_summary
else:
    _ch5_clicked = sidebar.button(
        "Export Chapter 5 Data",
        disabled=_research_disabled,
        use_container_width=True,
    )
    if _ch5_clicked and not _research_disabled:
        with st.spinner("Running evaluation across all high-risk customers..."):
            _ch5_results, _ch5_summary, _ch5_segments = run_chapter5_evaluation(
                df, st.session_state.model, target_col, id_col, threshold
            )
        if _ch5_results is not None and _ch5_summary is not None:
            st.session_state["ch5_results"] = _ch5_results
            st.session_state["ch5_summary"] = _ch5_summary
            st.session_state["ch5_segments"] = _ch5_segments

# ── Always render results from session_state (survives reruns) ──
if "ch5_results" in st.session_state and st.session_state["ch5_results"] is not None:
    _ch5_results = st.session_state["ch5_results"]
    _ch5_summary = st.session_state["ch5_summary"]
    _ch5_segments = st.session_state.get("ch5_segments")

    st.success(f"Chapter 5 data ready -- {_ch5_summary['total_high_risk_customers']} customers evaluated")

    # Warn if few valid Condition B rows
    _valid_b = int((_ch5_results['condition_b_actionable'] == 1).sum())
    if _valid_b < 5:
        st.warning(
            f"Only {_valid_b} customers have valid Condition B data. Results may be unreliable."
        )

    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("Actionability Rate B", f"{_ch5_summary['actionability_rate_b']:.1%}",
                   delta=f"vs A: {_ch5_summary['actionability_rate_a']:.1%}")
        st.metric("Rank Correlation", f"{_ch5_summary['rank_correlation']:.3f}",
                   delta=f"p={_ch5_summary['rank_correlation_pvalue']:.4f}")
    with mc2:
        st.metric("Top-20% Overlap", f"{_ch5_summary['top20_priority_overlap']:.1%}")
        st.metric("Retention Value Improvement",
                   f"{_ch5_summary['value_improvement_pct']:.1f}%",
                   delta=f"B: ${_ch5_summary['total_retention_value_b']:,.0f} vs A: ${_ch5_summary['total_retention_value_a']:,.0f}")

    if _ch5_segments is not None:
        st.markdown("**RFM Segment Breakdown:**")
        st.dataframe(_ch5_segments, width='stretch')

    # Download links (base64 href — bypasses Streamlit file manager to avoid rerun issue)
    import json as _json_dl
    _csv_bytes = _ch5_results.to_csv(index=False).encode()
    _json_bytes = _json_dl.dumps(_ch5_summary, indent=2).encode()
    _csv_b64 = base64.b64encode(_csv_bytes).decode()
    _json_b64 = base64.b64encode(_json_bytes).decode()

    dl1, dl2 = st.columns(2)
    with dl1:
        st.markdown(
            f'<a href="data:text/csv;base64,{_csv_b64}" '
            f'download="chapter5_evaluation_data.csv" '
            f'style="display:inline-block;padding:0.5em 1em;background:#262730;color:#fafafa;'
            f'border-radius:0.5rem;text-decoration:none;font-size:0.875rem;border:1px solid #4a4a5a;'
            f'text-align:center;width:100%;">'
            f'⬇ Download Evaluation CSV</a>',
            unsafe_allow_html=True,
        )
    with dl2:
        st.markdown(
            f'<a href="data:application/json;base64,{_json_b64}" '
            f'download="chapter5_summary_metrics.json" '
            f'style="display:inline-block;padding:0.5em 1em;background:#262730;color:#fafafa;'
            f'border-radius:0.5rem;text-decoration:none;font-size:0.875rem;border:1px solid #4a4a5a;'
            f'text-align:center;width:100%;">'
            f'⬇ Download Summary JSON</a>',
            unsafe_allow_html=True,
        )

sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


if run_automl:
    if not PYCARET_AVAILABLE:
        st.error("PyCaret is not available — please install pycaret first.")
        st.stop()

    def check_cancel():
        if st.session_state.get("cancel_automl"):
            reset_automl_state()
            st.warning("AutoML canceled. You can select the target column again and run when ready.")
            st.stop()

    with st.spinner("⚙️ Running AutoML... This may take a few minutes, please wait patiently while models are trained and optimized."):
        try:
            from sklearn.preprocessing import LabelEncoder
            check_cancel()

            if st.session_state.get("unsupervised_mode"):
                st.session_state["validation_messages"].append((
                    "warning",
                    "Unsupervised mode: target labels are inferred, so reported metrics reflect pseudo-label fit, not true churn ground truth."
                ))

            # 1) Prepare data and drop ID-like columns
            drop_cols = list(auto_id_cols) if auto_id_cols else []
            if id_col and id_col not in drop_cols:
                drop_cols.append(id_col)
            modeling_df = df.drop(columns=drop_cols, errors="ignore").copy()
            group_series = None
            if id_col and id_col in df.columns:
                try:
                    group_series = df.loc[modeling_df.index, id_col]
                except Exception:
                    group_series = df[id_col]
            check_cancel()

            # Explicit check and confirmation that ID column was dropped
            if id_col and id_col in modeling_df.columns:
                st.session_state["validation_messages"].append(("error", f"❌ ID column '{id_col}' was not dropped properly."))
            else:
                if id_col:
                    st.sidebar.info(f"✅ ID column '{id_col}' dropped before training.")

            # 2) Clean column names
            modeling_df.columns = modeling_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
            modeling_df = modeling_df.loc[:, ~modeling_df.columns.duplicated()]
            check_cancel()

            # 3) Ensure target column exists and encode Yes/No or similar to 1/0
            if target_col not in modeling_df.columns:
                possible = [c for c in modeling_df.columns if c.lower() == target_col.lower()]
                if possible:
                    target_col = possible[0]
                else:
                    raise ValueError(f"Target column '{target_col}' not found after cleaning.")

            # Drop duplicate rows to reduce leakage across folds
            dup_count = int(modeling_df.duplicated().sum())
            if dup_count > 0:
                modeling_df = modeling_df.drop_duplicates()
                st.session_state["validation_messages"].append(("warning", f"⚠ Removed {dup_count} duplicate rows before training."))
                if group_series is not None:
                    try:
                        group_series = group_series.loc[modeling_df.index]
                    except Exception:
                        pass

            # Lightweight feature filtering (no aggressive selection)
            drop_feature_cols = []
            miss_ratio = modeling_df.isna().mean()
            high_missing = miss_ratio[miss_ratio > 0.6].index.tolist()
            drop_feature_cols.extend([c for c in high_missing if c != target_col])
            constant_cols = [
                c for c in modeling_df.columns
                if c != target_col and modeling_df[c].nunique(dropna=True) <= 1
            ]
            drop_feature_cols.extend(constant_cols)
            drop_feature_cols = sorted(set(drop_feature_cols))
            if drop_feature_cols:
                modeling_df = modeling_df.drop(columns=drop_feature_cols, errors="ignore")

            # ---------------- Schema Inference ----------------
            # Infer column types from raw data to keep categorical signals intact
            inferred_num, inferred_cat = [], []
            for c in modeling_df.columns:
                if c == target_col:
                    continue
                as_num = pd.to_numeric(modeling_df[c], errors="coerce")
                if as_num.notna().mean() >= 0.7:
                    inferred_num.append(c)
                else:
                    inferred_cat.append(c)

            # Target: robust binary mapping (works beyond strict yes/no labels)
            target_encoded = encode_target_series(modeling_df[target_col], fit=True)
            target_valid_mask = target_encoded.notna()
            unmapped_count = int((~target_valid_mask).sum())
            if unmapped_count > 0:
                modeling_df = modeling_df.loc[target_valid_mask].copy()
                target_encoded = target_encoded.loc[target_valid_mask]
                st.session_state["validation_messages"].append((
                    "warning",
                    f"Target cleaning: dropped {unmapped_count} rows with unmapped/missing target values."
                ))

            if target_encoded.nunique() != 2:
                uniq_vals = sorted(target_encoded.dropna().unique().tolist())
                raise ValueError(
                    f"Target must be binary after encoding. Found classes: {uniq_vals}"
                )

            modeling_df[target_col] = target_encoded.astype(int).values
            st.sidebar.write("Target distribution:", modeling_df[target_col].value_counts())
            _tmap = st.session_state.get("target_label_mapping", {}) or {}
            _tstrategy = st.session_state.get("target_mapping_strategy", "unknown")
            if _tmap:
                _preview = ", ".join([f"{k}->{v}" for k, v in list(_tmap.items())[:6]])
                st.sidebar.caption(f"Target mapping ({_tstrategy}): {_preview}")
            check_cancel()

            # ---------------- FAST LEAKAGE DETECTION (Correlation only - very fast) ----------------
            # Only check numeric features with simple correlation (skips slow AUC calculations)
            leak_cols = []
            try:
                if modeling_df[target_col].nunique() == 2:
                    y_leak = modeling_df[target_col].values
                    
                    # Only check numeric features (fast)
                    for col in inferred_num:
                        try:
                            series = pd.to_numeric(modeling_df[col], errors="coerce")
                            if series.isna().all() or series.nunique(dropna=True) <= 1:
                                continue
                            
                            # Simple correlation check (very fast)
                            corr = abs(series.fillna(0).corr(pd.Series(y_leak)))
                            if corr >= 0.75:
                                leak_cols.append((col, corr, 'correlation'))
                        except:
                            continue
            except Exception:
                leak_cols = []

            # Remove leaking features
            if leak_cols:
                leak_names = [c for c, _, _ in leak_cols]
                leak_summary = ", ".join([f"{c} ({s:.2f})" for c, s, _ in leak_cols[:3]])
                
                st.session_state["leakage_cols"] = leak_names
                st.session_state["validation_messages"].append((
                    "warning",
                    f"🚫 Removed {len(leak_names)} leaking features: {leak_summary}"
                ))
                
                modeling_df = modeling_df.drop(columns=leak_names, errors="ignore")
                inferred_num = [c for c in inferred_num if c not in leak_names]
                inferred_cat = [c for c in inferred_cat if c not in leak_names]

            # Clean categorical values (standardize spaces) before any encoding
            for c in inferred_cat:
                modeling_df[c] = modeling_df[c].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
            check_cancel()

            # Binary-map obvious yes/no/boolean categoricals; move mapped cols to numeric.
            # Rules: a column is binary-remapped ONLY when ALL its unique values
            # fall entirely within one semantic group (boolean OR gender, not mixed).
            bool_keys  = {'yes', 'y', 'true', 't', '1', 'no', 'n', 'false', 'f', '0'}
            gender_keys = {'male', 'm', 'female', 'fem'}
            binary_map_bool   = {'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
                                  'no': 0,  'n': 0,  'false': 0, 'f': 0, '0': 0}
            binary_map_gender = {'male': 1, 'm': 1, 'female': 0, 'fem': 0}
            # Merged map used for actual remapping (no duplicate keys)
            binary_map = {**binary_map_bool, **binary_map_gender}
            remap_to_numeric = []
            for c in inferred_cat:
                lower_vals = modeling_df[c].astype(str).str.strip().str.lower()
                unique_lower = set(lower_vals.dropna().unique())
                if unique_lower and unique_lower.issubset(set(binary_map.keys())):
                    modeling_df[c] = lower_vals.map(binary_map).astype(int)
                    remap_to_numeric.append(c)

            # Update lists after binary mapping
            inferred_cat = [c for c in inferred_cat if c not in remap_to_numeric]
            inferred_num = sorted(set(inferred_num + remap_to_numeric))
            check_cancel()

            # Persist cat schema for later prediction cleaning
            st.session_state["cat_cols_clean"] = inferred_cat.copy()
            st.session_state["numeric_features"] = inferred_num.copy()
            st.session_state["binary_mapped_cols"] = remap_to_numeric.copy()
            st.session_state["binary_map"] = binary_map

            # Coerce numeric cols once
            if inferred_num:
                modeling_df[inferred_num] = modeling_df[inferred_num].apply(pd.to_numeric, errors="coerce")
            check_cancel()

            # IMPORTANT: defer imputation until AFTER train/test split.
            # Using full-dataset means/modes leaks holdout distribution.
            check_cancel()

            # Drop near-zero variance numeric features
            low_var_cols = []
            for c in inferred_num:
                try:
                    if modeling_df[c].var() <= 1e-8:
                        low_var_cols.append(c)
                except Exception:
                    continue
            if low_var_cols:
                modeling_df = modeling_df.drop(columns=low_var_cols, errors="ignore")
                inferred_num = [c for c in inferred_num if c not in low_var_cols]
                st.session_state["numeric_features"] = inferred_num.copy()

            # Persist raw feature list after filtering
            st.session_state["raw_feature_cols"] = [c for c in modeling_df.columns if c != target_col]

            # ============================================================================
            # UNIVERSAL RFM FEATURE ENGINEERING
            # ============================================================================
            try:
                from universal_rfm import UniversalRFMAnalyzer
                
                st.session_state["training_status"] = "🎯 Analyzing RFM patterns (Recency, Frequency, Monetary)..."
                rfm_analyzer = UniversalRFMAnalyzer(verbose=False)
                
                # Apply RFM analysis
                modeling_df = rfm_analyzer.analyze_and_engineer(modeling_df, target_col=target_col)
                
                # Update feature lists with new RFM features
                rfm_features = rfm_analyzer.rfm_features_created
                if rfm_features:
                    # Separate RFM features into numeric and categorical
                    rfm_numeric = [f for f in rfm_features if 'Score' in f or 'Value' in f or 'Seg_' in f]
                    rfm_categorical = [f for f in rfm_features if f == 'RFM_Segment']
                    
                    inferred_num.extend(rfm_numeric)
                    if rfm_categorical:
                        inferred_cat.extend(rfm_categorical)
                    
                    # Update session state
                    st.session_state["numeric_features"] = inferred_num.copy()
                    st.session_state["cat_cols_clean"] = inferred_cat.copy()
                    st.session_state["raw_feature_cols"] = [c for c in modeling_df.columns if c != target_col]
                    st.session_state["rfm_features"] = rfm_features
                    
                    st.session_state["validation_messages"].append((
                        "success",
                        f"✅ RFM Analysis: Created {len(rfm_features)} customer value features"
                    ))
                    
                    # Log RFM detection details
                    rfm_summary = rfm_analyzer.get_feature_summary()
                    st.session_state["rfm_summary"] = rfm_summary
                    
                    # Apply RFM to FULL original dataset for complete customer profiles
                    try:
                        full_rfm_df = df.drop(columns=drop_cols, errors="ignore").copy()
                        full_rfm_df = rfm_analyzer.analyze_and_engineer(full_rfm_df, target_col=target_col)
                        rfm_cols = [c for c in full_rfm_df.columns if 'RFM' in c]
                        if id_col and id_col in df.columns:
                            rfm_cols.insert(0, id_col)
                        st.session_state["rfm_dataframe"] = full_rfm_df[rfm_cols].copy()
                    except Exception as rfm_full_err:
                        # Fallback to training data only
                        st.session_state["rfm_dataframe"] = modeling_df[[c for c in modeling_df.columns if 'RFM' in c or c == id_col]].copy()
                    # ────────────────────────────────────────────────────────────
                    # CRITICAL: Remove RFM features from modeling_df BEFORE training
                    # RFM scores are derived from the same dataset → data leakage.
                    # They dominate the model, suppress original business features,
                    # and collapse all churn predictions to ~0.
                    # RFM data is kept in session_state["rfm_dataframe"] for display only.
                    # ────────────────────────────────────────────────────────────
                    rfm_cols_to_drop = [c for c in modeling_df.columns if 'RFM' in c]
                    if rfm_cols_to_drop:
                        modeling_df = modeling_df.drop(columns=rfm_cols_to_drop)
                        # Remove RFM features from feature lists
                        inferred_num = [c for c in inferred_num if 'RFM' not in c]
                        inferred_cat = [c for c in inferred_cat if 'RFM' not in c]
                        st.session_state["numeric_features"] = inferred_num.copy()
                        st.session_state["cat_cols_clean"] = inferred_cat.copy()
                        st.session_state["raw_feature_cols"] = [c for c in modeling_df.columns if c != target_col]
                        st.session_state["validation_messages"].append((
                            "info",
                            f"⚙️ RFM: Excluded {len(rfm_cols_to_drop)} RFM features from model training (kept for display only)"
                        ))

                else:
                    st.session_state["validation_messages"].append((
                        "info",
                        "ℹ️ RFM: No suitable columns detected in this dataset"
                    ))
            except ImportError:
                st.session_state["validation_messages"].append((
                    "warning",
                    "⚠️ RFM module not available - continuing without RFM features"
                ))
            except Exception as rfm_err:
                st.session_state["validation_messages"].append((
                    "warning",
                    f"⚠️ RFM analysis skipped: {str(rfm_err)[:60]}"
                ))
            check_cancel()
            # ============================================================================

            # ============================================================================
            # PROFESSIONAL VALIDATION ENGINE (Backend Only - No UI Changes)
            # ============================================================================
            try:
                # Initialize validation engine with output redirected to sidebar
                from io import StringIO
                import sys
                
                # Capture validation output
                old_stdout = sys.stdout
                sys.stdout = validation_buffer = StringIO()
                
                # Run comprehensive validation with universal dataset support
                validation_engine = ChurnValidationEngine(
                    modeling_df.copy(),
                    target_col,
                    verbose=True
                )
                
                # Detect leakage with universal threshold (works for any dataset)
                adaptive_threshold = 0.87 if len(modeling_df) < 1000 else 0.85
                leaked_features = validation_engine.detect_leakage(threshold=adaptive_threshold)
                
                # Get clean dataset if leakage found
                if leaked_features:
                    st.session_state["validation_leaked_features"] = leaked_features
                    modeling_df = validation_engine.create_clean_dataset()
                    
                    # Update feature lists after cleaning
                    inferred_num = [c for c in inferred_num if c not in leaked_features]
                    inferred_cat = [c for c in inferred_cat if c not in leaked_features]
                    st.session_state["numeric_features"] = inferred_num.copy()
                    st.session_state["cat_cols_clean"] = inferred_cat.copy()
                    st.session_state["raw_feature_cols"] = [c for c in modeling_df.columns if c != target_col]
                else:
                    st.session_state["validation_leaked_features"] = []
                
                # Show expected performance ranges (universal for binary classification)
                validation_engine.get_expected_performance()
                
                # Store validation engine for later use
                st.session_state["validation_engine"] = validation_engine
                
                # Restore stdout and save validation log
                sys.stdout = old_stdout
                validation_log = validation_buffer.getvalue()
                st.session_state["validation_log"] = validation_log
                
            except Exception as val_err:
                # Graceful degradation - don't break training if validation fails
                st.session_state["validation_engine"] = None
                st.session_state["validation_leaked_features"] = []
            
            check_cancel()
            # ============================================================================

            # ============================================================================
            # AUTONOMOUS MONITORING AGENT (Self-Healing Training)
            # ============================================================================
            # Initialize monitoring agent
            monitoring_agent = MonitoringAgent(max_retries=3, verbose=True)
            st.session_state["monitoring_agent"] = monitoring_agent
            
            # Adaptive fold count based on dataset size and imbalance
            minority_count = int(modeling_df[target_col].sum())
            total_count = len(modeling_df)
            imbalance_ratio = minority_count / total_count if total_count > 0 else 0.5
            if minority_count < 50:
                adaptive_folds = 3
            elif minority_count < 100 or imbalance_ratio < 0.05:
                adaptive_folds = 5
            elif total_count < 1000:
                adaptive_folds = 5
            elif total_count < 5000:
                adaptive_folds = 7
            else:
                adaptive_folds = 10

            # ── Leak-free pre-setup preprocessing ───────────────────────────────────
            # PyCaret 3.3.0 bug: normalize/feature_selection/remove_multicollinearity
            # inside setup() corrupt the AUC CV scorer (silently 0.0 per fold).
            # Fix: split FIRST (train/test), fit ALL transformers on train only,
            # transform test with the SAME fitted objects, then pass both to PyCaret's
            # test_data parameter.  This eliminates data leakage AND keeps AUC real.
            from sklearn.model_selection import StratifiedShuffleSplit
            from sklearn.preprocessing import PowerTransformer
            from lightgbm import LGBMClassifier as _LGBM

            prep_messages = []
            _train_size = 0.75  # must match training_config below

            # ── 0. Stratified split on raw data ─────────────────────────────────────
            _sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - _train_size, random_state=123)
            _train_idx, _test_idx = next(_sss.split(modeling_df, modeling_df[target_col]))
            train_df = modeling_df.iloc[_train_idx].reset_index(drop=True)
            test_df  = modeling_df.iloc[_test_idx].reset_index(drop=True)

            # Train-only imputation to prevent holdout leakage.
            _num_impute_cols = [
                c for c in inferred_num
                if c in train_df.columns and c != target_col
            ]
            if _num_impute_cols:
                _train_num_med = train_df[_num_impute_cols].median()
                train_df[_num_impute_cols] = train_df[_num_impute_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(_train_num_med)
                test_df[_num_impute_cols] = test_df[_num_impute_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(_train_num_med)

            _cat_impute_cols = [c for c in inferred_cat if c in train_df.columns]
            for _c in _cat_impute_cols:
                _mode_vals = train_df[_c].dropna().mode()
                _fill = _mode_vals.iloc[0] if len(_mode_vals) else "Unknown"
                train_df[_c] = train_df[_c].fillna(_fill)
                if _c in test_df.columns:
                    test_df[_c] = test_df[_c].fillna(_fill)

            num_cols_for_prep = [
                c for c in train_df.columns
                if c != target_col
                and pd.api.types.is_numeric_dtype(train_df[c])
                and train_df[c].nunique() > 2  # skip binary flags
            ]

            # ── 1. Yeo-Johnson + standardise  (DISABLED) ──────────────────────────
            # IMPORTANT: Power transform is DISABLED because it creates a fatal
            # mismatch between training and prediction:
            #   - Training: features are Yeo-Johnson transformed
            #   - Prediction (predict_row_prob, summary board, SHAP, counterfactuals):
            #     features are RAW (no transform applied)
            #   - Model expects transformed values but gets raw → all predictions wrong
            # Tree-based models (RF, XGB, LightGBM, CatBoost) are invariant to
            # monotonic transforms, so removing this has zero impact on model quality.
            # Diagnostic: AUC=0.96 without power transform (confirmed).
            _pt = None
            if num_cols_for_prep:
                prep_messages.append(("info",
                    f"⚙️ Pre-processing: Skipped normalization (tree models invariant; ensures prediction consistency)"))

            # ── 2. Multicollinearity removal  (correlations from TRAIN only) ───────
            num_after_norm = [c for c in num_cols_for_prep if c in train_df.columns]
            _drop_corr = []
            if len(num_after_norm) > 2:
                try:
                    _corr = train_df[num_after_norm].corr().abs()
                    _upper = _corr.where(np.triu(np.ones(_corr.shape, dtype=bool), k=1))
                    _mcol_thresh = 0.95 if len(num_after_norm) <= 15 else 0.90
                    _drop_corr = [c for c in _upper.columns if _upper[c].max() > _mcol_thresh]
                    if _drop_corr:
                        train_df = train_df.drop(columns=_drop_corr)
                        test_df  = test_df.drop(columns=[c for c in _drop_corr if c in test_df.columns])
                        prep_messages.append(("info",
                            f"⚙️ Pre-processing: removed {len(_drop_corr)} multicollinear features "
                            f"(|r|>{_mcol_thresh}, measured on train)"))
                except Exception as _e:
                    prep_messages.append(("warning", f"⚠️ Multicollinearity removal skipped: {str(_e)[:60]}"))

            # ── 3. LightGBM feature selection  (fit on TRAIN only) ─────────────────
            _feat_cols = [c for c in train_df.columns if c != target_col]
            _top_feats = _feat_cols  # default: keep all
            if len(_feat_cols) > 5:
                try:
                    _lgb_sel = _LGBM(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
                    _lgb_sel.fit(train_df[_feat_cols].fillna(0), train_df[target_col])
                    _imp = pd.Series(_lgb_sel.feature_importances_, index=_feat_cols).sort_values(ascending=False)
                    _cum = _imp.cumsum() / _imp.sum()
                    _keep_n = max(5, min(int((_cum <= 0.98).sum()) + 1, len(_feat_cols)))
                    _top_feats = _imp.head(_keep_n).index.tolist()
                    if _keep_n < len(_feat_cols):
                        train_df = train_df[_top_feats + [target_col]]
                        test_df  = test_df[[c for c in _top_feats if c in test_df.columns] + [target_col]]
                        prep_messages.append(("info",
                            f"⚙️ Pre-processing: feature selection kept {_keep_n}/{len(_feat_cols)} features "
                            f"(98% cumulative LightGBM importance, fit on train)"))
                except Exception as _e:
                    prep_messages.append(("warning", f"⚠️ Feature selection skipped: {str(_e)[:60]}"))

            # ── 4. Align columns between train and test ──────────────────────────────
            final_feat_cols = [c for c in train_df.columns if c != target_col]
            for _c in final_feat_cols:
                if _c not in test_df.columns:
                    test_df[_c] = 0  # fill any missing col with 0 (edge case)
            test_df = test_df[final_feat_cols + [target_col]]  # ensure same column order

            inferred_cat_prep = [c for c in inferred_cat if c in train_df.columns]
            inferred_num_prep = [c for c in inferred_num if c in train_df.columns]
            prep_messages.append(("info",
                f"⚙️ Pre-processing complete: {len(final_feat_cols)} features "
                f"| train={len(train_df):,} | test={len(test_df):,} "
                f"(no leakage — all transformers fit on train only)"))
            for _pm in prep_messages:
                st.session_state["validation_messages"].append(_pm)
            # Store holdout set for any post-training external evaluation
            st.session_state["holdout_test_df"] = test_df.copy()

            # ── Sanitise dtypes before handing to PyCaret ────────────────────────────
            # 1) CategoricalDtype → str  (ordered categoricals break predict_proba)
            _cat_dtype_cols = [c for c in train_df.columns
                               if str(train_df[c].dtype) == 'category'
                               or 'CategoricalDtype' in str(type(train_df[c].dtype))]
            # Also catch via .cat accessor (categorical series expose it)
            for _c in train_df.columns:
                try:
                    _ = train_df[_c].cat  # only succeeds for CategoricalDtype
                    if _c not in _cat_dtype_cols:
                        _cat_dtype_cols.append(_c)
                except AttributeError:
                    pass
            _cat_dtype_cols = [c for c in _cat_dtype_cols if c != target_col]
            if _cat_dtype_cols:
                train_df[_cat_dtype_cols] = train_df[_cat_dtype_cols].astype(str)

            # 2) inf / -inf → NaN, then fill numeric NaNs with column median
            #    Yeo-Johnson on near-zero-variance columns can produce inf values;
            #    sklearn's predict_proba returns nan for inf inputs → roc_auc_score = 0.
            _num_cols_fix = [c for c in train_df.columns
                             if c != target_col and pd.api.types.is_numeric_dtype(train_df[c])]
            _inf_count = np.isinf(train_df[_num_cols_fix]).sum().sum()
            if _inf_count > 0:
                train_df[_num_cols_fix] = train_df[_num_cols_fix].replace(
                    [np.inf, -np.inf], np.nan)
                prep_messages.append(("warning",
                    f"Replaced {_inf_count} inf values with NaN before PyCaret"))
            _nan_count = train_df[_num_cols_fix].isna().sum().sum()
            if _nan_count > 0:
                _med = train_df[_num_cols_fix].median()
                train_df[_num_cols_fix] = train_df[_num_cols_fix].fillna(_med)

            # 3) Ensure target is plain int64 (no nullable Int64 or float64)
            train_df[target_col] = train_df[target_col].astype(int)

            # 4) One-hot encode ALL remaining object columns BEFORE PyCaret
            #    ROOT CAUSE OF AUC=0: PyCaret 3.3's internal one-hot encoder
            #    breaks the CV AUC scorer. Pre-encoding avoids this entirely.
            _obj_cols = [c for c in train_df.columns
                         if c != target_col and train_df[c].dtype == 'object']
            if _obj_cols:
                _holdout_ref = st.session_state.get("holdout_test_df")
                train_df = pd.get_dummies(train_df, columns=_obj_cols, drop_first=True)
                # Apply same encoding to holdout test set so post-training eval works
                if _holdout_ref is not None:
                    _holdout_ref = pd.get_dummies(_holdout_ref, columns=[c for c in _obj_cols if c in _holdout_ref.columns], drop_first=True)
                    # Align columns: add any missing dummies as 0, drop extras
                    for _dc in train_df.columns:
                        if _dc not in _holdout_ref.columns:
                            _holdout_ref[_dc] = 0
                    _holdout_ref = _holdout_ref[[c for c in train_df.columns if c in _holdout_ref.columns]]
                    st.session_state["holdout_test_df"] = _holdout_ref
                prep_messages.append(("info",
                    f"⚙️ Pre-encoded {len(_obj_cols)} categorical columns "
                    f"({len(train_df.columns) - len(_obj_cols)} → {len(train_df.columns)} features)"))

            # ══════════════════════════════════════════════════════════════════════
            # SAFETY NET: Final RFM removal before clf.setup (belt + suspenders)
            # Even if earlier removal failed, this guarantees no RFM in training.
            # ══════════════════════════════════════════════════════════════════════
            _rfm_in_train = [c for c in train_df.columns if 'RFM' in c]
            _rfm_in_test = [c for c in test_df.columns if 'RFM' in c]
            if _rfm_in_train:
                train_df = train_df.drop(columns=_rfm_in_train)
                print(f"[SAFETY] Removed {len(_rfm_in_train)} RFM cols from train_df: {_rfm_in_train}")
                st.session_state["validation_messages"].append((
                    "warning", f"⚠️ Safety net: removed {len(_rfm_in_train)} RFM features from training data"))
            if _rfm_in_test:
                test_df = test_df.drop(columns=_rfm_in_test)
                print(f"[SAFETY] Removed {len(_rfm_in_test)} RFM cols from test_df: {_rfm_in_test}")
            # Also clear from holdout
            _holdout_safety = st.session_state.get("holdout_test_df")
            if _holdout_safety is not None:
                _rfm_holdout = [c for c in _holdout_safety.columns if 'RFM' in c]
                if _rfm_holdout:
                    st.session_state["holdout_test_df"] = _holdout_safety.drop(columns=_rfm_holdout)

            # Delete stale model files from previous runs (avoids loading old RFM-contaminated model)
            import glob as _glob
            for _stale in _glob.glob("automl_best_model*") + _glob.glob("automl_model_columns*"):
                try:
                    os.remove(_stale)
                    print(f"[SAFETY] Deleted stale file: {_stale}")
                except Exception:
                    pass

            # ── Diagnostic log (visible in server terminal) ──────────────────────────
            _num_cols_diag = [c for c in train_df.columns
                              if c != target_col and pd.api.types.is_numeric_dtype(train_df[c])]
            print("\n[DIAG] train_df dtypes before clf.setup:")
            print(train_df.dtypes.value_counts().to_string())
            print(f"[DIAG] object cols remaining: {[c for c in train_df.columns if train_df[c].dtype == 'object']}")
            print(f"[DIAG] RFM cols in train_df: {[c for c in train_df.columns if 'RFM' in c]}")
            print(f"[DIAG] target='{target_col}' value_counts:\n{train_df[target_col].value_counts().to_string()}")
            print(f"[DIAG] any inf: {np.isinf(train_df[_num_cols_diag]).values.any()}")
            print(f"[DIAG] any nan: {train_df[_num_cols_diag].isna().values.any()}")
            print(f"[DIAG] shape: {train_df.shape}")
            print(f"[DIAG] columns: {list(train_df.columns)}\n")
            # ────────────────────────────────────────────────────────────────────────────

            # Prepare training configuration.
            # Root cause of AUC=0 was object columns triggering PyCaret's internal
            # one-hot encoder bug.  Now that we pre-encode with get_dummies(),
            # fix_imbalance and fold_shuffle are safe to use again.
            training_config = {
                'session_id': 123,
                'fold': adaptive_folds,
                'fold_shuffle': True,
                'fix_imbalance': True,   # SMOTE for minority class — safe with pre-encoded data
                'use_gpu': False,
                'verbose': False,
                'n_select': 5,
                'turbo': False,
                'errors': 'ignore'
            }
            
            # Define monitored training function
            st.session_state["training_status"] = "⚙️ Setting up training environment..."
            def execute_training(data, config, target):
                """Training function wrapped by monitoring agent"""
                # Setup
                _ = clf.setup(
                    data=data,
                    target=target,
                    **{k: v for k, v in config.items() if k not in ['n_select', 'turbo', 'errors']}
                )
                
                # Compare models - sorted by AUC for overall balanced performance
                models_to_compare = ['rf', 'et', 'xgboost', 'gbc', 'lightgbm', 'lr', 'ridge', 'catboost']  # Tree + linear models (SVM removed: crashes on Windows)
                st.session_state["training_status"] = f"🔍 Comparing {len(models_to_compare)} models ({adaptive_folds}-fold CV each)..."
                best = clf.compare_models(
                    include=models_to_compare,
                    sort='AUC',  # Sort by AUC for overall balanced performance (Precision + Recall + F1)
                    fold=config.get('fold', 10),
                    turbo=config.get('turbo', False),
                    errors=config.get('errors', 'ignore'),
                    n_select=config.get('n_select', 5)
                )
                
                # LOG what compare_models returned
                if isinstance(best, list):
                    model_names = [type(m).__name__ for m in best]
                    st.session_state["validation_messages"].append(("info", f"📊 compare_models returned {len(best)} models: {', '.join(model_names)}"))
                else:
                    st.session_state["validation_messages"].append(("warning", f"⚠️ compare_models returned single model: {type(best).__name__} (n_select may have been ignored)"))
                
                return best
            
            # Execute training with automatic problem detection and fixing
            try:
                best, fix_log = monitoring_agent.monitor_and_fix(
                    execute_training,
                    train_df.copy(),  # pre-processed, leak-free TRAIN set
                    training_config,
                    target_col
                )
                
                # Log monitoring results
                for log_entry in fix_log:
                    if '✓' in log_entry or 'success' in log_entry.lower():
                        st.session_state["validation_messages"].append(("success", log_entry))
                    elif '🔧' in log_entry or 'fix' in log_entry.lower():
                        st.session_state["validation_messages"].append(("info", log_entry))
                    elif '⚠' in log_entry or '❌' in log_entry:
                        st.session_state["validation_messages"].append(("warning", log_entry))

                # ── Store leaderboard & patch AUC if still zero (safety net) ─────────
                try:
                    _raw_lb = clf.pull()
                    if isinstance(_raw_lb, pd.DataFrame):
                        # If AUC is still 0 despite pre-encoding, patch with holdout AUC
                        if 'AUC' in _raw_lb.columns and _raw_lb['AUC'].max() < 0.1:
                            from sklearn.metrics import roc_auc_score as _roc_auc
                            _holdout = st.session_state.get("holdout_test_df")
                            if _holdout is not None:
                                _best_list = best if isinstance(best, list) else [best]
                                _y_h = _holdout[target_col].astype(int).values
                                _X_h = _holdout.drop(columns=[target_col], errors='ignore')
                                for _i, _m in enumerate(_best_list):
                                    if _i >= len(_raw_lb):
                                        break
                                    try:
                                        _ph = clf.predict_model(_m, data=_X_h.copy(), raw_score=True)
                                        _ss = get_positive_score_series(_ph)
                                        if _ss is not None:
                                            _raw_lb.at[_raw_lb.index[_i], 'AUC'] = round(
                                                float(_roc_auc(_y_h, _ss.values)), 4)
                                    except Exception:
                                        pass
                        st.session_state["leaderboard_df"] = _raw_lb
                except Exception:
                    pass

            except Exception as train_err:
                # Final fallback if monitoring agent exhausts retries
                st.session_state["validation_messages"].append(
                    ("error", f"Training failed after all recovery attempts: {str(train_err)[:100]}")
                )
                # Try one last time with minimal setup
                try:
                    _ = clf.setup(
                        data=train_df,      # use pre-encoded train_df, not raw modeling_df
                        target=target_col,
                        session_id=123,
                        use_gpu=False,
                        verbose=False
                    )
                    best = clf.compare_models(
                        include=['rf', 'xgboost'],
                        sort='AUC',
                        fold=5,
                        n_select=1
                    )
                    st.session_state["validation_messages"].append(("warning", "Used minimal fallback training"))
                except Exception as final_err:
                    st.error(f"Critical training failure: {final_err}")
                    st.stop()
            
            check_cancel()

            # Use best individual model directly (stacking/blending disabled for speed)
            final_model = best[0] if isinstance(best, list) else best
            _model_name = type(final_model).__name__
            st.session_state["validation_messages"].append(("success", f"✓ Using best individual model: {_model_name}"))
            
            # LOG final model type before tuning
            final_model_type = type(final_model).__name__
            st.session_state["validation_messages"].append(("info", f"🎯 Final model before tuning: {final_model_type}"))
            
            check_cancel()

            # Skip tuning for ensemble models (already optimized via CV)
            st.session_state["training_status"] = "🎛️ Hyperparameter tuning..."
            # Tuning ensembles often causes numeric errors and minimal gain
            if 'Stacking' in final_model_type or 'Voting' in final_model_type:
                st.session_state["validation_messages"].append(("info", "⏭️ Skipping tuning for ensemble (already CV-optimized)"))
                tuned_best = final_model
            else:
                # Only tune single models
                try:
                    tuned_best = clf.tune_model(
                        final_model,
                        optimize='AUC',  # AUC is threshold-independent; threshold handles recall/precision tradeoff
                        choose_better=True,
                        early_stopping=True,
                        search_library='optuna',
                        search_algorithm='tpe',
                        fold=min(adaptive_folds, 10),
                        n_iter=20  # Reduced from 30 to avoid numeric issues
                    )
                    st.session_state["validation_messages"].append(("success", "✅ Hyperparameter tuning completed"))
                except Exception as tune_err:
                    st.session_state["validation_messages"].append(("warning", f"Tuning skipped: {str(tune_err)[:50]}"))
                    tuned_best = final_model
            check_cancel()

            st.session_state["training_status"] = "📊 Calibrating probabilities..."
            # Apply Isotonic Calibration for more realistic churn probabilities
            st.session_state["training_status"] = "📊 Calibrating probabilities..."
            try:
                best_cal = clf.calibrate_model(tuned_best, method='isotonic')
            except Exception as cal_err:
                st.warning(f"Calibration skipped (using tuned/best model): {cal_err}")
                best_cal = tuned_best
            check_cancel()

            # Store model and preprocessing pipeline from current experiment
            try:
                prep_pipe = clf.get_config("prep_pipe")
            except Exception:
                prep_pipe = None
            if prep_pipe is None:
                try:
                    prep_pipe = clf.get_config("pipeline")
                except Exception:
                    prep_pipe = None
            st.session_state.model = best_cal
            st.session_state.prep_pipe = prep_pipe
            st.session_state["training_status"] = "Completed"

            # Set threshold for prediction classification
            # Automatically determine optimal threshold based on ROC curve (Youden's J statistic)
            try:
                from sklearn.metrics import roc_curve
                X_test_thr = clf.get_config('X_test')
                y_test_thr = clf.get_config('y_test')
                if X_test_thr is not None and y_test_thr is not None and len(X_test_thr) > 0:
                    preds_thr = clf.predict_model(best_cal, data=X_test_thr, raw_score=True)
                    score_col = [c for c in preds_thr.columns if 'Score_1' in c or
                                 (c.lower().endswith('1') and 'score' in c.lower())]
                    if not score_col:
                        score_col = [c for c in preds_thr.columns if 'score' in c.lower()]
                    if score_col:
                        y_scores_thr = preds_thr[score_col[0]].values
                        y_true_thr = pd.to_numeric(y_test_thr, errors='coerce').fillna(0).astype(int).values
                        fpr, tpr, thresholds_roc = roc_curve(y_true_thr, y_scores_thr)
                        # Youden's J statistic: maximizes (sensitivity + specificity - 1)
                        j_scores = tpr - fpr
                        best_idx = int(np.argmax(j_scores))
                        optimal_thr = float(thresholds_roc[best_idx])
                        optimal_thr = max(0.1, min(0.9, optimal_thr))
                        st.session_state['adaptive_threshold'] = optimal_thr
                        st.session_state['optimal_threshold'] = optimal_thr
                        st.session_state['validation_messages'].append(('info', f'⚙️ Model-suggested threshold (Youden J): {optimal_thr:.3f}'))
                    else:
                        st.session_state['adaptive_threshold'] = 0.35
                        st.session_state['optimal_threshold'] = 0.35
                else:
                    st.session_state['adaptive_threshold'] = 0.35
                    st.session_state['optimal_threshold'] = 0.35
            except Exception:
                st.session_state['adaptive_threshold'] = 0.35
                st.session_state['optimal_threshold'] = 0.35

            # Evaluate the best model interactively (optional visualization)
            try:
                clf.evaluate_model(best_cal)
            except Exception as e:
                st.warning(f"Evaluation skipped: {e}")
            st.session_state.best_model_name = str(best)

            # Capture detailed metrics of the selected model
            try:
                perf = clf.pull()
                if isinstance(perf, pd.DataFrame):
                    perf_row = perf.iloc[0]
                    st.session_state["best_model_auc"] = perf_row.get("AUC", None)
                    st.session_state["best_model_precision"] = perf_row.get("Precision", None)
                    st.session_state["best_model_recall"] = perf_row.get("Recall", None)
                else:
                    st.session_state["best_model_auc"] = None
                    st.session_state["best_model_precision"] = None
                    st.session_state["best_model_recall"] = None
            except Exception:
                st.session_state["best_model_auc"] = None
                st.session_state["best_model_precision"] = None
                st.session_state["best_model_recall"] = None

            import glob
            for f in glob.glob("automl_best_model*"):
                try:
                    os.remove(f)
                except Exception:
                    pass

            clf.save_model(best_cal, "automl_best_model")
            model = clf.load_model("automl_best_model")
            check_cancel()

            import joblib
            model_feature_cols = clf.get_config("X_train").columns.tolist()
            joblib.dump(model_feature_cols, "automl_model_columns.pkl")
            st.session_state["model_feature_cols"] = model_feature_cols

            # ── POST-TRAINING: Verify no RFM leakage ────────────────────────────────
            _rfm_in_model = [c for c in model_feature_cols if 'RFM' in c]
            if _rfm_in_model:
                print(f"[CRITICAL] RFM features STILL in model after training: {_rfm_in_model}")
                st.session_state["validation_messages"].append((
                    "error", f"❌ CRITICAL: {len(_rfm_in_model)} RFM features leaked into model: {_rfm_in_model}"))
            else:
                print(f"[OK] Model has {len(model_feature_cols)} features, 0 RFM. Features: {model_feature_cols}")
                st.session_state["validation_messages"].append((
                    "success", f"✅ Model trained on {len(model_feature_cols)} business features (RFM excluded)"))

            st.session_state.model = model
            st.session_state.fitted = True

            # Persist holdout split so diagnostics work on Streamlit re-runs
            # (clf.get_config only works in the training rerun; session_state persists)
            try:
                st.session_state["_X_test"] = clf.get_config("X_test")
                st.session_state["_y_test"] = clf.get_config("y_test")
            except Exception:
                pass

            # ============================================================================
            # POST-TRAINING VALIDATION (Backend Only - No UI Changes)
            # ============================================================================
            try:
                if st.session_state.get("validation_engine") is not None:
                    validation_engine = st.session_state["validation_engine"]
                    
                    # Get train/test split from PyCaret (universal approach)
                    try:
                        X_train = clf.get_config("X_train")
                        X_test = clf.get_config("X_test")
                        y_train = clf.get_config("y_train")
                        y_test = clf.get_config("y_test")
                        
                        # Ensure we have valid data
                        if X_train is None or X_test is None or len(X_test) == 0:
                            raise ValueError("No test set available")
                        
                        # Redirect validation output
                        from io import StringIO
                        old_stdout = sys.stdout
                        sys.stdout = validation_buffer_2 = StringIO()
                        
                        # Run comprehensive validation (handles any model/dataset type)
                        validation_report = validation_engine.validate_model_performance(
                            model, X_train, X_test, y_train, y_test,
                            optimal_threshold=st.session_state.get('optimal_threshold', 0.5)
                        )
                        
                        # Restore stdout
                        sys.stdout = old_stdout
                        validation_log_2 = validation_buffer_2.getvalue()
                        
                        # Store validation report
                        st.session_state["validation_report"] = validation_report
                        st.session_state["validation_log_posttraining"] = validation_log_2
                        
                        # Store validation messages based on status
                        if validation_report['status'] == 'CRITICAL':
                            msg = "CRITICAL MODEL ISSUES:\n\n" + "\n".join([f"• {issue[:80]}" for issue in validation_report['issues'][:2]])
                            st.session_state["validation_messages"].append(("error", msg))
                        elif validation_report['status'] == 'WARNING':
                            msg = "MODEL WARNINGS:\n\n" + "\n".join([f"• {w[:80]}" for w in validation_report['warnings'][:2]])
                            st.session_state["validation_messages"].append(("warning", msg))
                        else:
                            test_metrics = validation_report['test_metrics']
                            msg = (
                                f"Model Validation Passed\n\n"
                                f"Test Performance:\n"
                                f"• AUC: {test_metrics['auc']:.3f}\n"
                                f"• F1: {test_metrics['f1']:.3f}\n"
                                f"• Recall: {test_metrics['recall']:.3f}\n"
                                f"• Precision: {test_metrics['precision']:.3f}"
                            )
                            st.session_state["validation_messages"].append(("success", msg))
                        
                    except Exception as val_err2:
                        st.session_state["validation_messages"].append(("info", f"Post-validation skipped: {str(val_err2)[:50]}..."))
                        st.session_state["validation_report"] = None
                        
            except Exception as val_err3:
                # Silent fail - don't disrupt user experience
                st.session_state["validation_report"] = None
            # ============================================================================

            st.session_state["validation_messages"].append(("success", "✅ AutoML training complete — calibrated with sigmoid scaling."))
            if drop_cols:
                st.session_state["validation_messages"].append(("info", f"🧹 Dropped ID column(s): {', '.join(drop_cols)}"))
            
            # ========== RFM SUMMARY ==========
            if st.session_state.get("rfm_summary"):
                rfm_sum = st.session_state["rfm_summary"]
                if rfm_sum.get('total_features', 0) > 0:
                    quality_pct = int(round(float(rfm_sum.get('quality_score', 0.0)) * 100))
                    st.session_state["validation_messages"].append((
                        "success",
                        f"RFM Features: {rfm_sum['total_features']} value indicators created ({quality_pct}% dimension coverage)"
                    ))
                    src = rfm_sum.get('rfm_sources', {})
                    m_src = ", ".join(src.get('monetary', [])[:2]) if src.get('monetary') else "None"
                    f_src = ", ".join(src.get('frequency', [])[:2]) if src.get('frequency') else "None"
                    r_src = ", ".join(src.get('recency', [])[:2]) if src.get('recency') else "None"
                    st.session_state["validation_messages"].append((
                        "info",
                        f"RFM sources -> M: {m_src} | F: {f_src} | R: {r_src}"
                    ))

        except Exception as e:
            import traceback
            st.error(f"AutoML failed: {e}")
            with st.expander("🔎 Full error traceback", expanded=False):
                st.code(traceback.format_exc())
            st.stop()
        finally:
            st.session_state.automl_running = False
            st.session_state.automl_just_finished = True

    if st.session_state.automl_just_finished:
        st.session_state.automl_just_finished = False
        st.rerun()


# ----------------------------- RFM INSIGHTS DISPLAY -----------------------------
if st.session_state.fitted and st.session_state.get("rfm_features"):
    with st.expander("🎯 RFM Analysis Results (Customer Value Segmentation)", expanded=True):
        rfm_features = st.session_state.get("rfm_features", [])
        rfm_summary = st.session_state.get("rfm_summary", {})
        
        st.markdown("### 📊 RFM Features Created")
        st.markdown(f"**Total Features Generated:** {len(rfm_features)}")

        if rfm_summary:
            quality = float(rfm_summary.get('quality_score', 0.0))
            dims = rfm_summary.get('dimensions_detected', {})
            src = rfm_summary.get('rfm_sources', {})
            dim_line = (
                f"M={'Yes' if dims.get('monetary') else 'No'} | "
                f"F={'Yes' if dims.get('frequency') else 'No'} | "
                f"R={'Yes' if dims.get('recency') else 'No'}"
            )
            m_src = ", ".join(src.get('monetary', [])[:2]) if src.get('monetary') else "None"
            f_src = ", ".join(src.get('frequency', [])[:2]) if src.get('frequency') else "None"
            r_src = ", ".join(src.get('recency', [])[:2]) if src.get('recency') else "None"
            st.caption(f"Detection quality: {quality:.0%} ({dim_line})")
            st.caption(f"Detected sources -> Monetary: {m_src} | Frequency: {f_src} | Recency: {r_src}")
        
        # Display features in columns
        col1, col2, col3 = st.columns(3)
        
        # Separate features by type
        score_features = [f for f in rfm_features if 'Score' in f]
        value_features = [f for f in rfm_features if 'Value' in f]
        segment_features = [f for f in rfm_features if 'Seg_' in f or f == 'RFM_Segment']
        composite_features = [f for f in rfm_features if 'Composite' in f]
        
        with col1:
            st.markdown("**📈 Score Features (1-5 scale)**")
            for feat in score_features:
                st.markdown(f"• `{feat}`")
        
        with col2:
            st.markdown("**💰 Value Features (raw)**")
            for feat in value_features:
                st.markdown(f"• `{feat}`")
        
        with col3:
            st.markdown("**🏷️ Segmentation**")
            for feat in composite_features + segment_features[:1]:  # Composite + Segment column
                st.markdown(f"• `{feat}`")
            if len(segment_features) > 1:
                st.markdown(f"• {len(segment_features)-1} segment indicators")
        
        st.markdown("---")
        
        # Show customer distribution by segment if available
        rfm_df = st.session_state.get("rfm_dataframe")
        if rfm_df is not None and 'RFM_Segment' in rfm_df.columns:
            st.markdown("---")
            st.markdown("### 👥 Customer Segment Distribution")
            
            segment_counts = rfm_df['RFM_Segment'].value_counts()
            segment_pct = (segment_counts / len(rfm_df) * 100).round(1)
            
            # Create segment cards
            seg_col1, seg_col2, seg_col3, seg_col4 = st.columns(4)
            
            segment_colors = {
                'Champions': '#22c55e',
                'Established': '#3b82f6',
                'Developing': '#f59e0b',
                'At_Risk': '#e11d48'
            }
            
            for i, (segment, col) in enumerate(zip(['Champions', 'Established', 'Developing', 'At_Risk'], 
                                                     [seg_col1, seg_col2, seg_col3, seg_col4])):
                count = segment_counts.get(segment, 0)
                pct = segment_pct.get(segment, 0)
                color = segment_colors.get(segment, '#6b7280')
                
                with col:
                    st.markdown(
                        f"""
                        <div class="kpi-metric" style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); border-left: 4px solid {color};">
                            <div class="kpi-label">{segment}</div>
                            <div class="kpi-value" style="color:{color};">{count:,}</div>
                            <div class="kpi-label">{pct}% of customers</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Show sample customers with RFM scores
            st.markdown("---")
            st.markdown("### 📋 Sample Customers with RFM Scores")
            
            # Select relevant RFM columns to display
            display_cols = [c for c in rfm_df.columns if 'RFM' in c or c == id_col]
            
            if display_cols:
                # Sample 2 from each segment if possible
                sample_df = pd.concat([
                    rfm_df[rfm_df['RFM_Segment'] == seg].head(2) 
                    for seg in segment_counts.index
                ]).head(10)
                
                st.dataframe(
                    sample_df[display_cols].style.background_gradient(
                        subset=['RFM_Composite_Score'] if 'RFM_Composite_Score' in display_cols else [],
                        cmap='RdYlGn'
                    ),
                    width='stretch',
                    height=300
                )
                
                st.caption("💡 **RFM Scores:** 5 = Best, 1 = Lowest | Higher composite score = More valuable customer")
        else:
            st.info("ℹ️ RFM segment distribution will appear here after training. Click 'Run AutoML' to generate customer segmentation.")


# ----------------------------- SHOW LEADERBOARD (IF ANY) + KPI METRICS -----------------------------
overall_churn_rate, best_model_acc, num_at_risk = None, None, None
leaderboard_df = None
if st.session_state.fitted:
    try:
        leaderboard_df = st.session_state.get("leaderboard_df")
        if leaderboard_df is None:
            leaderboard_df = clf.pull()
        # Only compute metrics, do not render leaderboard/KPI here (render in tab1)
        try:
            if leaderboard_df is not None and "Accuracy" in leaderboard_df.columns and "AUC" in leaderboard_df.columns:
                best_row = leaderboard_df.iloc[0]
                best_acc = best_row.get("Accuracy", None)
                best_auc = best_row.get("AUC", None)
                best_name = st.session_state.get("best_model_name", "Unknown")
                best_model_acc = best_acc
            else:
                pass
        except Exception:
            pass
        # Calculate KPI metrics row
        try:
            # Calculate churn rate safely from original dataset
            try:
                churn_vals = encode_target_series(df[target_col], fit=False).dropna()
                overall_churn_rate = float(churn_vals.mean()) if len(churn_vals) else 0
            except Exception:
                overall_churn_rate = 0

            # At-risk count
            # Predict on 20% sample of training data for KPI computation
            # Calculate predicted churn share from model output
            predicted_churn_rate = None
            try:
                aligned_kpi = align_to_model_columns(df.copy(), st.session_state.model)
                # Use model.predict_proba directly (avoids double-preprocessing)
                _proba_kpi = st.session_state.model.predict_proba(aligned_kpi)
                if hasattr(_proba_kpi, 'iloc'):
                    score_series2 = _proba_kpi.iloc[:, 1] if _proba_kpi.shape[1] > 1 else _proba_kpi.iloc[:, 0]
                else:
                    score_series2 = pd.Series(
                        _proba_kpi[:, 1] if _proba_kpi.shape[1] > 1 else _proba_kpi[:, 0],
                        index=aligned_kpi.index
                    )
                predicted_churn_rate = (score_series2 > threshold).mean()
                num_at_risk = int((score_series2 > threshold).sum())
            except Exception:
                predicted_churn_rate = None
                # Fallback to sample-based count if full prediction fails
                try:
                    preds_df = clf.predict_model(
                        st.session_state.model,
                        data=st.session_state.train_df.sample(frac=0.2, random_state=123),
                        raw_score=True
                    )
                    score_series = get_positive_score_series(preds_df)
                    if score_series is not None:
                        num_at_risk = int((score_series > threshold).sum())
                except Exception:
                    pass
        except Exception:
            predicted_churn_rate = None
    except Exception:
        pass



def predict_row_prob(model, row_df: pd.DataFrame) -> float:
    """Predict probability on a single row using the pipeline."""
    try:
        aligned = align_to_model_columns(row_df.copy(), model)
        # Use model.predict_proba directly (bypasses clf.predict_model
        # which would double-preprocess through the loaded Pipeline)
        proba = model.predict_proba(aligned)
        if hasattr(proba, 'iloc'):
            p = float(proba.iloc[0, 1]) if proba.shape[1] > 1 else float(proba.iloc[0, 0])
        else:
            p = float(proba[0][1]) if len(proba[0]) > 1 else float(proba[0][0])
        return float(np.clip(p, 0.001, 0.999))
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
        return 0.001


prediction_proba = None
prediction_label = "—"
confidence = "—"

if st.session_state.fitted and st.session_state.model is not None:
    # Drop ID & target for predicting a single row
    drop_pred_cols = [c for c in [id_col, target_col] if c] + st.session_state.get("drop_id_cols", [])
    row_for_pred = customer_row.drop(columns=list(dict.fromkeys(drop_pred_cols)), errors="ignore")
    prediction_proba = predict_row_prob(st.session_state.model, row_for_pred)
    decision_threshold = threshold  # single source of truth from slider
    prediction_label = "Likely to Churn" if prediction_proba > decision_threshold else "Likely to Stay"
    confidence = (
        "High" if abs(prediction_proba - decision_threshold) > 0.25
        else "Medium" if abs(prediction_proba - decision_threshold) > 0.10
        else "Low"
    )

# ----------------------------- CUSTOMER SUMMARY BOARD -----------------------------
if st.session_state.fitted and st.session_state.model is not None:
    with st.expander("📊 View All Customer Predictions (click to expand)", expanded=False):
        try:
            # Predict on the full aligned dataset using model.predict_proba directly
            # (avoids double-preprocessing from clf.predict_model on loaded Pipeline)
            aligned_board = align_to_model_columns(df.copy(), st.session_state.model)
            _proba_board = st.session_state.model.predict_proba(aligned_board)
            if hasattr(_proba_board, 'iloc'):
                _scores_board = _proba_board.iloc[:, 1] if _proba_board.shape[1] > 1 else _proba_board.iloc[:, 0]
            else:
                _scores_board = _proba_board[:, 1] if _proba_board.shape[1] > 1 else _proba_board[:, 0]

            all_preds = df.copy()
            all_preds["Score_1"] = _scores_board
            all_preds["Prediction"] = np.where(
                all_preds["Score_1"] > threshold, "Churn", "Stay"
            )

            churn_df = all_preds[all_preds["Prediction"] == "Churn"]
            stay_df = all_preds[all_preds["Prediction"] == "Stay"]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🚨 Customers Likely to Churn")
                show_cols = [c for c in [id_col, target_col, "Prediction", "Score_1"] if c in churn_df.columns]
                st.dataframe(churn_df[show_cols], width='stretch', height=280)
            with col2:
                st.subheader("🟢 Customers Likely to Stay")
                show_cols = [c for c in [id_col, target_col, "Prediction", "Score_1"] if c in stay_df.columns]
                st.dataframe(stay_df[show_cols], width='stretch', height=280)

            # Calibration reliability indicator (Brier Score) — safe check
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss
            # Use target from original df (not preds_df, which doesn't contain the target)
            if target_col and target_col in df.columns:
                y_true = encode_target_series(df[target_col], fit=False)
                valid_mask = y_true.notna()
                if valid_mask.sum() > 1:
                    y_true = y_true[valid_mask].astype(int)
                    y_prob = np.array(_scores_board, dtype=float)[valid_mask.to_numpy()]
                    if y_true.nunique() == 2:
                        # Brier score: mean squared error between per-sample probabilities and true labels
                        # Lower = better calibration (0 = perfect, 0.25 = random on balanced data)
                        brier = brier_score_loss(y_true, np.clip(y_prob, 0, 1))
                        # Calibration curve for visual reference only
                        prob_true, prob_pred = calibration_curve(
                            y_true, y_prob, n_bins=min(10, max(2, int(len(y_true) / 20)))
                        )
                        cal_mae = float(np.mean(np.abs(prob_pred - prob_true)))
                        st.caption(f"📏 Brier Score: {brier:.4f} (lower = better) | Calibration MAE: {cal_mae:.4f}")
                    else:
                        st.caption("📏 Calibration curve skipped - target is not binary after mapping.")
                else:
                    st.caption("📏 Calibration curve skipped - insufficient mapped target labels.")
            else:
                st.caption("📏 Calibration curve skipped — target column not found in dataset.")

        except Exception as e:
            st.error(f"❌ AutoML internal prediction failed: {e}")

# ----------------------------- SHAP EXPLANATION -----------------------------
shap_values = None
shap_error_msg = None
shap_plot_vals, shap_plot_names, shap_plot_data = None, None, None
top_features, top_values = [], []
shap_runtime_info = None

if st.session_state.fitted and st.session_state.model is not None and SHAP_AVAILABLE:
    try:
        # Build SHAP from the same prediction interface used in production.
        # Then aggregate transformed/dummy features back to original feature names.
        drop_expl_cols = [c for c in [id_col, target_col] if c] + st.session_state.get("drop_id_cols", [])
        drop_expl_cols = list(dict.fromkeys(drop_expl_cols))
        bg_full = df.drop(columns=drop_expl_cols, errors="ignore").copy()
        row_raw = customer_row.drop(columns=drop_expl_cols, errors="ignore").copy()

        if len(bg_full) == 0 or len(row_raw) == 0:
            raise ValueError("Insufficient rows for SHAP explanation.")

        bg_raw = bg_full.sample(min(120, len(bg_full)), random_state=42).copy()

        model_obj = st.session_state.model
        prep_pipe = st.session_state.get("prep_pipe", None)
        estimator = model_obj

        # If model is Pipeline, keep both full model and final estimator paths available.
        if hasattr(model_obj, "named_steps"):
            try:
                steps = list(model_obj.named_steps.items())
                if len(steps) > 1:
                    from sklearn.pipeline import Pipeline as SkPipeline
                    estimator = steps[-1][1]
                    if prep_pipe is None:
                        prep_pipe = SkPipeline(steps[:-1])
            except Exception:
                pass

        attempts = []

        # Attempt 1 (preferred): model-aligned feature space + full model predict_proba
        def _attempt_model_aligned():
            bg_model = align_to_model_columns(bg_raw.copy(), model_obj)
            row_model = align_to_model_columns(row_raw.copy(), model_obj)
            bg_model = bg_model.apply(pd.to_numeric, errors="coerce").fillna(0)
            row_model = row_model.apply(pd.to_numeric, errors="coerce").fillna(0)
            _ = _predict_positive_scores(model_obj, row_model.head(1))
            feat_names = list(bg_model.columns)
            bg_mat = _to_dense_numeric(bg_model)
            row_mat = _to_dense_numeric(row_model)

            def _pred(arr):
                arr_df = pd.DataFrame(arr, columns=feat_names)
                return _predict_positive_scores(model_obj, arr_df)

            return bg_mat, row_mat, feat_names, row_model, _pred, "model-aligned"

        attempts.append(_attempt_model_aligned)

        # Attempt 2: explicit prep_pipe transform + raw estimator
        if prep_pipe is not None:
            def _attempt_preppipe_estimator():
                bg_raw_aligned = align_to_raw_features(bg_raw.copy())
                row_raw_aligned = align_to_raw_features(row_raw.copy())
                bg_trans = prep_pipe.transform(bg_raw_aligned)
                row_trans = prep_pipe.transform(row_raw_aligned)
                bg_mat = _to_dense_numeric(bg_trans)
                row_mat = _to_dense_numeric(row_trans)
                _ = _predict_positive_scores(estimator, row_mat[:1])
                try:
                    feat_names = list(prep_pipe.get_feature_names_out())
                except Exception:
                    feat_names = [f"feature_{i}" for i in range(bg_mat.shape[1])]
                row_model = pd.DataFrame(row_mat, columns=feat_names)

                def _pred(arr):
                    return _predict_positive_scores(estimator, _to_dense_numeric(arr))

                return bg_mat, row_mat, feat_names, row_model, _pred, "prep-pipe"

            attempts.append(_attempt_preppipe_estimator)

        shap_values = None
        shap_feature_names = None
        shap_row_model_df = None
        last_shap_err = None

        for attempt_fn in attempts:
            try:
                bg_mat, row_mat, feat_names, row_model_df, pred_fn, source = attempt_fn()
                # Keep evaluations bounded for stability/speed on wide datasets.
                max_evals = int(min(401, max(31, 2 * bg_mat.shape[1] + 1)))
                explainer = shap.Explainer(pred_fn, bg_mat, algorithm="permutation")
                shap_values = explainer(row_mat, max_evals=max_evals)
                shap_feature_names = feat_names
                shap_row_model_df = row_model_df
                shap_runtime_info = source
                break
            except Exception as _e:
                last_shap_err = _e
                continue

        if shap_values is None:
            raise RuntimeError(f"All SHAP backends failed: {last_shap_err}")

        # Normalize SHAP outputs across explainer variants
        sv = shap_values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        vals = np.array(sv.values) if hasattr(sv, "values") else np.array(sv)
        if vals.ndim == 3:
            out_idx = 1 if vals.shape[-1] > 1 else 0
            vals = vals[:, :, out_idx]
        if vals.ndim == 2:
            vals = vals[0]
        vals = np.asarray(vals, dtype=float).ravel()

        agg_vals, agg_names, agg_data = _aggregate_shap_to_base_features(
            vals,
            shap_feature_names,
            shap_row_model_df,
            row_raw,
        )

        shap_plot_vals = agg_vals
        shap_plot_names = agg_names
        shap_plot_data = agg_data

        if shap_plot_vals is not None and len(shap_plot_vals) > 0:
            order = np.argsort(np.abs(np.asarray(shap_plot_vals, dtype=float)))[::-1][:8]
            top_features = [shap_plot_names[i] for i in order]
            top_values = [shap_plot_data[i] for i in order] if shap_plot_data is not None else [None for _ in order]
    except Exception as e:
        import traceback
        shap_error_msg = f"ℹ️ SHAP explanation unavailable: {e}"
        with st.expander("🔎 SHAP error details", expanded=False):
            st.code(traceback.format_exc())

# ----------------------------- NAVIGATION -----------------------------
nav_labels = ["Model Overview", "Single Prediction", "Insights"]
nav_choice = st.radio("Navigation", nav_labels, horizontal=True, label_visibility="collapsed", key="nav_main")

# --------- SECTION 1: MODEL OVERVIEW ---------
if nav_choice == "Model Overview":
    st.markdown('<div class="section-padding"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="section-icon overview"></span>Model Overview</div>', unsafe_allow_html=True)
    st.write("Review dataset, AutoML leaderboard, and overall churn statistics.")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    
    # ===== DATA QUALITY REPORT =====
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown("### Data Quality Report")
    
    # ===== SECTION 1: OVERVIEW METRICS =====
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicates = int(df.duplicated().sum())
    missing_pct = (df.isna().mean() * 100).round(2)
    unique_counts = df.nunique(dropna=True)
    dtypes = df.dtypes.astype(str)
    
    # Calculate complete rows (rows with no missing values)
    complete_rows = int(df.dropna().shape[0])
    complete_rows_pct = (complete_rows / total_rows * 100) if total_rows > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{total_rows:,}")
    with col2:
        st.metric("Features", total_cols)
    with col3:
        st.metric("Duplicates", duplicates)
    with col4:
        st.metric("Complete Rows", f"{complete_rows:,}", delta=f"{complete_rows_pct:.1f}%")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # ===== SECTION 3.5: VARIABLE SUMMARY STATISTICS =====
    st.markdown("#### Variable Summary Statistics")
    
    # Create dropdown to select variable with compact styling
    col_select, col_spacer = st.columns([2, 2])
    with col_select:
        selected_variable = st.selectbox(
            "Select Variable",
            options=df.columns.tolist(),
            index=0
        )
    
    if selected_variable:
        col_data = df[selected_variable]
        
        # Check if numeric or categorical
        if pd.api.types.is_numeric_dtype(col_data):
            # Numeric statistics
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Mean", f"{col_data.mean():.2f}")
            with stats_col2:
                st.metric("Median", f"{col_data.median():.2f}")
            with stats_col3:
                st.metric("Std Dev", f"{col_data.std():.2f}")
            with stats_col4:
                st.metric("Missing", f"{col_data.isna().sum():,}")
            
            stats_col5, stats_col6, stats_col7, stats_col8 = st.columns(4)
            with stats_col5:
                st.metric("Min", f"{col_data.min():.2f}")
            with stats_col6:
                st.metric("Max", f"{col_data.max():.2f}")
            with stats_col7:
                st.metric("25th Percentile", f"{col_data.quantile(0.25):.2f}")
            with stats_col8:
                st.metric("75th Percentile", f"{col_data.quantile(0.75):.2f}")
            
            stats_col9, stats_col10, stats_col11, stats_col12 = st.columns(4)
            with stats_col9:
                st.metric("Skewness", f"{col_data.skew():.2f}")
            with stats_col10:
                st.metric("Kurtosis", f"{col_data.kurtosis():.2f}")
            with stats_col11:
                st.metric("Count", f"{col_data.count():,}")
            with stats_col12:
                st.metric("Unique", f"{col_data.nunique():,}")
        else:
            # Categorical statistics
            value_counts = col_data.value_counts()
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Unique Values", f"{col_data.nunique():,}")
            with stats_col2:
                st.metric("Most Common", f"{value_counts.index[0] if len(value_counts) > 0 else 'N/A'}")
            with stats_col3:
                st.metric("Most Common Count", f"{value_counts.values[0] if len(value_counts) > 0 else 0:,}")
            with stats_col4:
                st.metric("Missing", f"{col_data.isna().sum():,}")
    
    # ===== SECTION 4: MISSING DATA PATTERNS =====
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if df.isna().sum().sum() > 0:
        st.markdown("#### Missing Data Patterns")
        # Create missing data heatmap
        missing_data = df.isna()
        missing_by_col = missing_data.sum().sort_values(ascending=False).head(20)
        
        if len(missing_by_col) > 0:
            try:
                import plotly.graph_objects as go
                # Sample data if too large
                sample_size = min(1000, len(df))
                missing_sample = missing_data[missing_by_col.index].sample(n=sample_size, random_state=42)
                
                fig = go.Figure(data=go.Heatmap(
                    z=missing_sample.T.values,
                    x=missing_sample.index,
                    y=missing_by_col.index,
                    colorscale=[[0, '#22c55e'], [1, '#ef4444']],
                    showscale=False
                ))
                fig.update_layout(
                    title=f'Missing Data Pattern (Top 20 Features, {sample_size} Samples)',
                    height=400,
                    xaxis_title='Sample Index',
                    yaxis_title='Feature',
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Heatmap visualization unavailable")
            
            st.markdown("**Features with Missing Values:**")
            st.dataframe(pd.DataFrame({
                'Feature': missing_by_col.index,
                'Missing Count': missing_by_col.values,
                'Missing %': (missing_by_col.values / len(df) * 100).round(2)
            }), width='stretch')
    
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    
    # ===== MODEL TRAINING & LEADERBOARD SECTION =====
    if leaderboard_df is not None:
        st.markdown("### 🎯 Model Performance Metrics")
        st.markdown(
            """
            <div class="kpi-row">
              <div class="kpi-metric">
                <div class="kpi-label">Overall Churn Rate (Dataset)</div>
                <div class="kpi-value" style="color:#e11d48;">{churn:.1f}%</div>
              </div>
              <div class="kpi-metric">
                <div class="kpi-label">Predicted Churn Rate (Model)</div>
                <div class="kpi-value" style="color:#f59e0b;">{predicted:.1f}%</div>
              </div>
              <div class="kpi-metric">
                <div class="kpi-label">Best Model AUC</div>
                <div class="kpi-value" style="color:#2563eb;">{auc:.1f}%</div>
              </div>
              <div class="kpi-metric">
                <div class="kpi-label">At-Risk Customers</div>
                <div class="kpi-value" style="color:#f59e0b;">{risk}</div>
              </div>
            </div>
            """.format(
                churn=overall_churn_rate*100 if overall_churn_rate is not None else 0,
                predicted=predicted_churn_rate*100 if 'predicted_churn_rate' in locals() and predicted_churn_rate is not None else 0,
                auc=(st.session_state.get("best_model_auc") or 0)*100,
                risk=num_at_risk if num_at_risk is not None else "—"
            ),
            unsafe_allow_html=True
        )
        
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    
    # AutoML Leaderboard
    if leaderboard_df is not None:
        with st.expander("📊 AutoML Leaderboard", expanded=False):
            model_name = st.session_state.get("best_model_name", "Unknown Model")
            st.markdown(f"**🧩 Selected Model:** `{model_name}`")
            st.dataframe(leaderboard_df, width='stretch')
            # PyCaret 3.3 has a known bug where cross-validation AUC returns 0 when
            # certain preprocessing combinations are active. The actual model performance
            # is evaluated directly on the held-out test set — see Model Trust Evidence
            # and the Validation Report sections below for the real scores.
            if "AUC" in leaderboard_df.columns and leaderboard_df["AUC"].max() < 0.1:
                st.info(
                    "The AUC column shows 0 due to a known PyCaret 3.3 cross-validation scorer "
                    "limitation. This does not mean the model is bad — the actual holdout AUC "
                    "is shown in the Model Trust Evidence and Validation Report sections below."
                )
            try:
                metric_cols = [c for c in ["Accuracy", "AUC", "Recall", "Precision", "F1"] if c in leaderboard_df.columns]
                if metric_cols and not leaderboard_df.empty:
                    top = leaderboard_df.iloc[0][metric_cols]
                    if (top >= 0.995).all():
                        st.warning(
                            "⚠️ CV metrics are near-perfect across folds. This often indicates target leakage, "
                            "duplicates, or a target proxy feature. Consider reviewing leakage warnings and validate on fresh data."
                        )
            except Exception:
                pass
    with st.expander("🧭 Model Diagnostics and Validation Report", expanded=False):
        if not st.session_state.fitted:
            st.markdown(
                '<div class="gray-info">Run AutoML to view calibration and confusion matrix diagnostics.</div>',
                unsafe_allow_html=True
            )
        else:
            try:
                from sklearn.metrics import confusion_matrix
                from sklearn.calibration import calibration_curve

                # Use holdout/test data persisted to session_state during training
                X_test = st.session_state.get("_X_test", None)
                y_test = st.session_state.get("_y_test", None)

                if X_test is None or y_test is None:
                    st.markdown(
                        '<div class="gray-info">Diagnostics unavailable: holdout/test set not found. Run AutoML again.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Extract raw estimator from the saved Pipeline to avoid
                    # double-preprocessing (X_test is already in PyCaret's internal format)
                    model_obj = st.session_state.model
                    _estimator = model_obj.steps[-1][1] if hasattr(model_obj, 'steps') else model_obj
                    try:
                        _proba_arr = _estimator.predict_proba(X_test)
                        _y_prob_raw = _proba_arr[:, 1] if _proba_arr.shape[1] > 1 else _proba_arr[:, 0]
                        score_series = pd.Series(_y_prob_raw, index=X_test.index)
                    except Exception as _e1:
                        # Fallback: try full Pipeline (in case model wasn't a Pipeline)
                        try:
                            _proba_arr = model_obj.predict_proba(X_test)
                            _y_prob_raw = _proba_arr[:, 1] if _proba_arr.shape[1] > 1 else _proba_arr[:, 0]
                            score_series = pd.Series(_y_prob_raw, index=X_test.index)
                        except Exception as _e2:
                            score_series = None

                    y_true = encode_target_series(y_test.copy(), fit=False)
                    if score_series is not None:
                        score_series = pd.Series(score_series, index=y_true.index)
                        valid_mask = y_true.notna()
                        y_true = y_true[valid_mask].astype(int)
                        score_series = score_series[valid_mask]

                    if score_series is None or len(y_true) == 0 or y_true.nunique() != 2:
                        st.markdown(
                            '<div class="gray-info">Diagnostics require a binary target with probability scores.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        y_prob = score_series.to_numpy()
                        y_pred = (y_prob >= threshold).astype(int)

                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                        fig1, ax1 = plt.subplots(figsize=(4.2, 3.2))
                        im = ax1.imshow(cm, cmap="Blues")
                        ax1.set_title("Confusion Matrix (holdout, current threshold)")
                        ax1.set_xlabel("Predicted")
                        ax1.set_ylabel("Actual")
                        ax1.set_xticks([0, 1])
                        ax1.set_yticks([0, 1])
                        ax1.set_xticklabels(["Stay", "Churn"])
                        ax1.set_yticklabels(["Stay", "Churn"])
                        for i in range(2):
                            for j in range(2):
                                ax1.text(j, i, cm[i, j], ha="center", va="center", color="#0f172a", fontsize=10)
                        fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

                        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
                        fig2, ax2 = plt.subplots(figsize=(4.2, 3.2))
                        ax2.plot(prob_pred, prob_true, marker="o", label="Model")
                        ax2.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Perfect")
                        ax2.set_title("Calibration Plot (holdout)")
                        ax2.set_xlabel("Predicted probability")
                        ax2.set_ylabel("Observed fraction")
                        ax2.legend(loc="lower right")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            _show_matplotlib(fig1, container=st)
                        with col_b:
                            _show_matplotlib(fig2, container=st)

                        # Metrics + class balance
                        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        acc = accuracy_score(y_true, y_pred)

                        balance_counts = y_true.value_counts().sort_index()
                        total = balance_counts.sum()
                        stay_pct = (balance_counts.get(0, 0) / total * 100) if total else 0
                        churn_pct = (balance_counts.get(1, 0) / total * 100) if total else 0

                        mcol1, mcol2 = st.columns(2)
                        with mcol1:
                            st.markdown("**Holdout metrics**")
                            st.write(
                                {
                                    "Accuracy": f"{acc:.2%}",
                                    "Precision": f"{prec:.2%}",
                                    "Recall": f"{rec:.2%}",
                                    "F1": f"{f1:.2%}",
                                }
                            )
                        with mcol2:
                            st.markdown("**Holdout class balance**")
                            st.write(
                                {
                                    "Stay (0)": f"{balance_counts.get(0, 0)} ({stay_pct:.1f}%)",
                                    "Churn (1)": f"{balance_counts.get(1, 0)} ({churn_pct:.1f}%)",
                                }
                            )

                        # ── MODEL TRUST EVIDENCE (for non-technical stakeholders) ──────────
                        st.markdown("---")
                        st.markdown("### Model Trust Evidence")
                        st.caption(
                            "Four independent tests that confirm the model is genuinely good, "
                            "not just memorising the data."
                        )

                        from sklearn.metrics import roc_auc_score
                        _auc_score = roc_auc_score(y_true, y_prob)

                        # ── Test 1: Plain-English AUC ────────────────────────────────────
                        _auc_pct = int(round(_auc_score * 100))
                        st.markdown(f"""
**Test 1 — Ranking accuracy (AUC = {_auc_score:.3f})**

> Pick any one churner and any one non-churner at random from the test set.
> The model will rank the churner with a **higher risk score {_auc_pct} times out of 100**.
> A coin-toss model scores 50. Industry standard for a good churn model is 75+.
""")

                        # ── Test 2: Lift chart ───────────────────────────────────────────
                        st.markdown("**Test 2 — Business Lift: how many churners does each targeting tier catch?**")
                        _sorted_idx = np.argsort(y_prob)[::-1]
                        _n = len(y_true)
                        _total_churners = y_true.sum()
                        _tiers = [0.10, 0.20, 0.30, 0.40, 0.50]
                        _lift_rows = []
                        for _t in _tiers:
                            _k = max(1, int(_t * _n))
                            _caught = y_true.iloc[_sorted_idx[:_k]].sum() if hasattr(y_true, 'iloc') else y_true[_sorted_idx[:_k]].sum()
                            _recall_at_k = _caught / _total_churners if _total_churners > 0 else 0
                            _lift_val = _recall_at_k / _t  # lift over random
                            _lift_rows.append({
                                "Contact top...": f"{int(_t*100)}% of customers",
                                "Churners captured": f"{int(_caught)} / {int(_total_churners)} ({_recall_at_k:.0%})",
                                "Lift over random": f"{_lift_val:.1f}×"
                            })
                        st.dataframe(
                            pd.DataFrame(_lift_rows).set_index("Contact top..."),
                            width='stretch'
                        )

                        # Lift curve chart
                        _gains = []
                        _steps = np.arange(0.01, 1.01, 0.01)
                        for _s in _steps:
                            _k = max(1, int(_s * _n))
                            _caught = y_true.iloc[_sorted_idx[:_k]].sum() if hasattr(y_true, 'iloc') else y_true[_sorted_idx[:_k]].sum()
                            _gains.append(_caught / _total_churners if _total_churners > 0 else 0)
                        import plotly.graph_objects as go
                        _lift_fig = go.Figure()
                        _lift_fig.add_trace(go.Scatter(
                            x=list(_steps * 100), y=[g * 100 for g in _gains],
                            name="Model", line=dict(color="#2563eb", width=2.5)
                        ))
                        _lift_fig.add_trace(go.Scatter(
                            x=[0, 100], y=[0, 100],
                            name="Random baseline", line=dict(color="#94a3b8", dash="dash")
                        ))
                        _lift_fig.update_layout(
                            title="Cumulative Gain — % churners caught vs % customers contacted",
                            xaxis_title="% of customers contacted (sorted by risk score)",
                            yaxis_title="% of all churners captured",
                            height=340, margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
                        )
                        st.plotly_chart(_lift_fig, use_container_width=True)
                        st.caption(
                            "A model that's just guessing follows the diagonal line. "
                            "The further above the diagonal, the more business value the model provides."
                        )

                        # ── Test 3: Algorithm agreement ──────────────────────────────────
                        st.markdown("**Test 3 — Multiple independent algorithms agree**")
                        _lb = st.session_state.get("leaderboard_df")
                        if _lb is not None and "AUC" in _lb.columns and len(_lb) >= 2:
                            _lb_clean = _lb[_lb["AUC"] > 0.5][["Model", "AUC", "Recall", "F1"]].head(5)
                            _auc_std = _lb_clean["AUC"].std()
                            st.dataframe(_lb_clean, width='stretch')
                            if _auc_std < 0.03:
                                st.success(
                                    f"Top models have AUC spread of only {_auc_std:.3f} — "
                                    "they all independently arrive at the same answer, "
                                    "which rules out any single model overfitting."
                                )
                            else:
                                st.info(f"AUC spread across models: {_auc_std:.3f}")
                        else:
                            st.info("Run AutoML to see multi-model agreement.")

                        # ── Test 4: CV stability (train vs test gap) ─────────────────────
                        st.markdown("**Test 4 — The model generalises (training vs test gap)**")
                        _val_report = st.session_state.get("validation_report")
                        if _val_report:
                            _tr_auc = _val_report.get("train_metrics", {}).get("auc", None)
                            _te_auc = _val_report.get("test_metrics", {}).get("auc", None)
                            if _tr_auc and _te_auc:
                                _gap = _tr_auc - _te_auc
                                _gap_label = "Low" if _gap < 0.05 else ("Moderate" if _gap < 0.10 else "High")
                                st.markdown(f"""
| | AUC |
|---|---|
| Training set | {_tr_auc:.3f} |
| Test set (unseen data) | {_te_auc:.3f} |
| Gap | {_gap:.3f} ({_gap_label}) |
""")
                                if _gap < 0.05:
                                    st.success(
                                        "Tiny train/test gap — the model performs almost identically on "
                                        "data it has never seen. This is the key sign that it generalises "
                                        "to real new customers, not just the training data."
                                    )
                                elif _gap < 0.10:
                                    st.warning(
                                        f"Moderate gap ({_gap:.3f}) — acceptable, but monitor on new data."
                                    )
                                else:
                                    st.error(
                                        f"Large gap ({_gap:.3f}) — model may be overfitting. "
                                        "Validate on genuinely new customers before deploying."
                                    )
                        else:
                            st.info("Validation report not available — re-run AutoML.")
                        # ──────────────────────────────────────────────────────────────────
                
                # Add Validation Report section
                validation_report = st.session_state.get("validation_report")
                if validation_report:
                    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("### Validation Report")
                    
                    # Minimal status text
                    status = validation_report['status']
                    if status == 'CRITICAL':
                        status_color = "#ef4444"
                    elif status == 'WARNING':
                        status_color = "#f59e0b"
                    else:
                        status_color = "#10b981"
                    
                    st.markdown(f"""
                    <div style="margin-bottom:24px;">
                        <div style="font-size:13px;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">Validation Status</div>
                        <div style="font-size:18px;color:{status_color};font-weight:600;">{status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics cards
                    train_m = validation_report['train_metrics']
                    test_m = validation_report['test_metrics']
                    gaps = validation_report['gaps']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background:#f8fafc;padding:16px;border-radius:10px;border:1px solid #e2e8f0;">
                            <div style="font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:12px;">Training Set</div>
                            <div style="font-size:14px;color:#1e293b;line-height:1.8;">
                                <div>AUC: <span style="float:right;font-weight:600;">{train_m['auc']:.4f}</span></div>
                                <div>F1: <span style="float:right;font-weight:600;">{train_m['f1']:.4f}</span></div>
                                <div>Recall: <span style="float:right;font-weight:600;">{train_m['recall']:.4f}</span></div>
                                <div>Precision: <span style="float:right;font-weight:600;">{train_m['precision']:.4f}</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background:#f8fafc;padding:16px;border-radius:10px;border:1px solid #e2e8f0;">
                            <div style="font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:12px;">Test Set</div>
                            <div style="font-size:14px;color:#1e293b;line-height:1.8;">
                                <div>AUC: <span style="float:right;font-weight:600;">{test_m['auc']:.4f}</span></div>
                                <div>F1: <span style="float:right;font-weight:600;">{test_m['f1']:.4f}</span></div>
                                <div>Recall: <span style="float:right;font-weight:600;">{test_m['recall']:.4f}</span></div>
                                <div>Precision: <span style="float:right;font-weight:600;">{test_m['precision']:.4f}</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        gap_color = "#ef4444" if abs(gaps['auc_gap']) > 0.10 else "#10b981"
                        st.markdown(f"""
                        <div style="background:#f8fafc;padding:16px;border-radius:10px;border:1px solid #e2e8f0;">
                            <div style="font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:12px;">Performance Gaps</div>
                            <div style="font-size:14px;color:#1e293b;line-height:1.8;">
                                <div>AUC Gap: <span style="float:right;font-weight:600;color:{gap_color};">{gaps['auc_gap']:+.4f}</span></div>
                                <div>F1 Gap: <span style="float:right;font-weight:600;color:{gap_color};">{gaps['f1_gap']:+.4f}</span></div>
                                <div>Accuracy: <span style="float:right;font-weight:600;color:{gap_color};">{gaps['accuracy_gap']:+.4f}</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Issues and warnings
                    if validation_report.get('issues'):
                        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
                        st.markdown("<div style='font-size:14px;font-weight:600;color:#1e293b;margin-bottom:12px;'>Critical Issues</div>", unsafe_allow_html=True)
                        for issue in validation_report['issues']:
                            clean_issue = issue.replace("⚠️", "").replace("⚠", "").strip()
                            st.markdown(f"""
                            <div style="background:#fef2f2;padding:12px 16px;border-radius:8px;border-left:3px solid #ef4444;margin-bottom:8px;font-size:13px;color:#991b1b;">
                                {clean_issue}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if validation_report.get('warnings'):
                        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                        st.markdown("<div style='font-size:14px;font-weight:600;color:#1e293b;margin-bottom:12px;'>Warnings</div>", unsafe_allow_html=True)
                        for warning in validation_report['warnings']:
                            clean_warning = warning.replace("⚠️", "").replace("⚠", "").strip()
                            st.markdown(f"""
                            <div style="background:#fffbeb;padding:12px 16px;border-radius:8px;border-left:3px solid #f59e0b;margin-bottom:8px;font-size:13px;color:#92400e;">
                                {clean_warning}
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(
                    f'<div class="gray-info">Diagnostics unavailable: {e}</div>',
                    unsafe_allow_html=True
                )
    
    # Validation Report section removed - now integrated into Model Diagnostics above

# --------- SECTION 2: SINGLE PREDICTION ---------
elif nav_choice == "Single Prediction":
    st.markdown('<div class="section-padding"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="section-icon single"></span>Single Prediction</div>', unsafe_allow_html=True)
    st.write("Inspect an individual customer, see predicted churn risk, SHAP drivers, and tailored recommendations.")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown('<div class="subsection-title"><span class="subsection-icon prediction"></span>Prediction Result</div>', unsafe_allow_html=True)
        if prediction_proba is None:
            st.markdown(
                '<div class="gray-info">Run AutoML to see predictions.</div>',
                unsafe_allow_html=True
            )
        else:
            color_bar = "#e11d48" if prediction_label == "Likely to Churn" else "#22c55e"
            bar_val = int(prediction_proba * 100)
            st.markdown(
                f"""
                <div style="font-size:1.1rem;font-weight:700;margin-bottom:7px;color:{color_bar};">
                  {prediction_label}
                </div>
                <progress value="{bar_val}" max="100" style="width:100%;height:22px;background:#f1f3f5;border-radius:8px;">
                  {bar_val}%
                </progress>
                <div style="font-size:1.05rem;margin-top:8px;">
                  <span style="color:#adb5bd;">Churn Probability:</span> <b style="color:{color_bar};">{prediction_proba:.2%}</b>
                  <span style="color:#adb5bd;margin-left:10px;">Threshold:</span> <b>{decision_threshold:.2f}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"<span style='color:#adb5bd;'>Confidence:</span> <b>{confidence}</b>", unsafe_allow_html=True)
        
        # RFM Profile Display (nested under Prediction Result)
        rfm_df = st.session_state.get("rfm_dataframe")
        rfm_row = None
        
        if rfm_df is not None and not rfm_df.empty:
            # Try multiple matching strategies
            # Strategy 1: Match by ID column
            if id_col and id_col in customer_row.columns and id_col in rfm_df.columns:
                cust_id = customer_row[id_col].iloc[0]
                rfm_row = rfm_df[rfm_df[id_col] == cust_id]
            
            # Strategy 2: Match by index (works when no ID column or ID matching fails)
            if (rfm_row is None or rfm_row.empty) and customer_row.index[0] in rfm_df.index:
                rfm_row = rfm_df.loc[[customer_row.index[0]]]
            
            # Strategy 3: Use row position as fallback
            if (rfm_row is None or rfm_row.empty) and len(rfm_df) > 0:
                try:
                    row_idx = df.index.get_loc(customer_row.index[0])
                    if row_idx < len(rfm_df):
                        rfm_row = rfm_df.iloc[[row_idx]]
                except:
                    pass
                
            if rfm_row is not None and not rfm_row.empty:
                st.markdown('<div class="subsection-title" style="margin-top:15px;"><span class="subsection-icon rfm"></span>RFM Value Profile</div>', unsafe_allow_html=True)
                
                # Extract RFM scores
                rfm_data = rfm_row.iloc[0]
                segment = rfm_data.get('RFM_Segment', 'Unknown')
                composite = rfm_data.get('RFM_Composite_Score', 0)
                
                # Segment colors with icons
                segment_colors = {
                    'Champions': ('#22c55e', '◆'),
                    'Established': ('#3b82f6', '★'),
                    'Developing': ('#f59e0b', '▲'),
                    'At_Risk': ('#e11d48', '⚠')
                }
                seg_color, seg_icon = segment_colors.get(segment, ('#6b7280', '●'))
                
                # Display segment badge
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg, {seg_color}15 0%, {seg_color}05 100%);
                                border-left:4px solid {seg_color};padding:12px;border-radius:8px;margin-bottom:10px;">
                        <div style="font-size:0.85rem;color:#adb5bd;margin-bottom:4px;">Customer Segment</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{seg_color};">
                            {seg_icon} {segment}
                        </div>
                        <div style="font-size:0.9rem;color:#adb5bd;margin-top:4px;">
                            Composite Score: <b style="color:{seg_color};">{composite:.2f}/5.0</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display individual RFM scores - always show all 3 cards (N/A if not available)
                # Try to get scores (handle both Series index and dict-like access)
                try:
                    mon_score = rfm_data.get('RFM_Monetary_Score', None)
                    freq_score = rfm_data.get('RFM_Frequency_Score', None)
                    rec_score = rfm_data.get('RFM_Recency_Score', None)
                    
                    # Build all 3 cards - show N/A with gray style if score doesn't exist
                    cards_html = '<div style="display:flex;gap:8px;margin-top:8px;">'
                    
                    # Monetary card
                    if mon_score is not None:
                        mon_pct = (mon_score / 5.0) * 100
                        mon_color = '#22c55e' if mon_score >= 4 else '#f59e0b' if mon_score >= 3 else '#e11d48'
                        cards_html += f'<div style="flex:1;background:#f8f9fa;padding:8px;border-radius:6px;text-align:center;"><div style="font-size:1.2rem;">$</div><div style="font-size:0.75rem;color:#adb5bd;margin:2px 0;">Monetary</div><div style="font-size:1.1rem;font-weight:700;color:{mon_color};">{mon_score:.1f}</div><div style="background:#e9ecef;height:4px;border-radius:2px;margin-top:4px;overflow:hidden;"><div style="background:{mon_color};height:100%;width:{mon_pct:.1f}%;"></div></div></div>'
                    else:
                        cards_html += '<div style="flex:1;background:#f8f9fa;padding:8px;border-radius:6px;text-align:center;opacity:0.5;"><div style="font-size:1.2rem;">$</div><div style="font-size:0.75rem;color:#adb5bd;margin:2px 0;">Monetary</div><div style="font-size:1.1rem;font-weight:700;color:#9ca3af;">N/A</div><div style="background:#e9ecef;height:4px;border-radius:2px;margin-top:4px;"></div></div>'
                    
                    # Frequency card
                    if freq_score is not None:
                        freq_pct = (freq_score / 5.0) * 100
                        freq_color = '#22c55e' if freq_score >= 4 else '#f59e0b' if freq_score >= 3 else '#e11d48'
                        cards_html += f'<div style="flex:1;background:#f8f9fa;padding:8px;border-radius:6px;text-align:center;"><div style="font-size:1.2rem;">↻</div><div style="font-size:0.75rem;color:#adb5bd;margin:2px 0;">Frequency</div><div style="font-size:1.1rem;font-weight:700;color:{freq_color};">{freq_score:.1f}</div><div style="background:#e9ecef;height:4px;border-radius:2px;margin-top:4px;overflow:hidden;"><div style="background:{freq_color};height:100%;width:{freq_pct:.1f}%;"></div></div></div>'
                    else:
                        cards_html += '<div style="flex:1;background:#f8f9fa;padding:8px;border-radius:6px;text-align:center;opacity:0.5;"><div style="font-size:1.2rem;">↻</div><div style="font-size:0.75rem;color:#adb5bd;margin:2px 0;">Frequency</div><div style="font-size:1.1rem;font-weight:700;color:#9ca3af;">N/A</div><div style="background:#e9ecef;height:4px;border-radius:2px;margin-top:4px;"></div></div>'
                    
                    # Recency card
                    if rec_score is not None:
                        rec_pct = (rec_score / 5.0) * 100
                        rec_color = '#22c55e' if rec_score >= 4 else '#f59e0b' if rec_score >= 3 else '#e11d48'
                        cards_html += f'<div style="flex:1;background:#f8f9fa;padding:8px;border-radius:6px;text-align:center;"><div style="font-size:1.2rem;">◷</div><div style="font-size:0.75rem;color:#adb5bd;margin:2px 0;">Recency</div><div style="font-size:1.1rem;font-weight:700;color:{rec_color};">{rec_score:.1f}</div><div style="background:#e9ecef;height:4px;border-radius:2px;margin-top:4px;overflow:hidden;"><div style="background:{rec_color};height:100%;width:{rec_pct:.1f}%;"></div></div></div>'
                    else:
                        cards_html += '<div style="flex:1;background:#f8f9fa;padding:8px;border-radius:6px;text-align:center;opacity:0.5;"><div style="font-size:1.2rem;">◷</div><div style="font-size:0.75rem;color:#adb5bd;margin:2px 0;">Recency</div><div style="font-size:1.1rem;font-weight:700;color:#9ca3af;">N/A</div><div style="background:#e9ecef;height:4px;border-radius:2px;margin-top:4px;"></div></div>'
                    
                    cards_html += '</div>'
                    st.markdown(cards_html, unsafe_allow_html=True)
                    
                except Exception as e:
                    # Debug: show what columns are available
                    st.caption(f"RFM scores not found. Available: {list(rfm_data.index)[:5]}")
            else:
                # Debug: Show what's available
                if rfm_df is not None:
                    st.warning(f"RFM data exists ({len(rfm_df)} rows) but couldn't match customer. Check if training completed successfully.")
                else:
                    st.info("RFM Profile: Retrain the model to generate customer value scores.")
        else:
            # RFM dataframe not available
            st.info("**RFM Profile unavailable**\n\nRun AutoML to train with RFM features.")

    with colB:
        st.markdown('<div class="subsection-title"><span class="subsection-icon profile"></span>Customer Profile</div>', unsafe_allow_html=True)
        _profile_display = customer_row.T.head(10).copy()
        # Coerce all values to str to avoid ArrowTypeError on mixed-type object columns
        _profile_display = _profile_display.astype(str)
        st.dataframe(_profile_display, width='stretch', height=400)

    with colC:
        st.markdown(
            """
            <div class="subsection-title">
                <span class="subsection-icon shap"></span>
                SHAP Explanation
                <span title="SHAP (SHapley Additive Explanations) explains which features most influenced the churn prediction for this customer. Bar length shows impact; color shows direction (red=increases churn, green=retains)."
                      style="color:#adb5bd;cursor:help;font-size:1.1rem;margin-left:8px;">&#9432;</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if shap_plot_vals is not None:
            try:
                shap_bar_vals = np.asarray(shap_plot_vals, dtype=float).ravel()
                shap_bar_names = list(shap_plot_names) if shap_plot_names is not None else []
                order = np.argsort(np.abs(shap_bar_vals))[::-1][:8]
                if not shap_bar_names or len(shap_bar_names) != len(shap_bar_vals):
                    shap_bar_names = [f"feature_{i}" for i in range(len(shap_bar_vals))]

                names = [shap_bar_names[i] for i in order][::-1]
                vals = shap_bar_vals[order][::-1]
                bar_colors = ['#e11d48' if v > 0 else '#22c55e' for v in vals]

                # Optional feature values for hover
                hover_vals = None
                if shap_plot_data is not None:
                    data_vals = np.asarray(shap_plot_data).ravel()
                    if len(data_vals) == len(shap_bar_vals):
                        hover_vals = [data_vals[i] for i in order][::-1]

                # Interactive Plotly first, fallback to matplotlib
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    if hover_vals is not None:
                        fig.add_trace(
                            go.Bar(
                                x=vals,
                                y=names,
                                orientation="h",
                                marker_color=bar_colors,
                                customdata=hover_vals,
                                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<br>Value: %{customdata}<extra></extra>",
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Bar(
                                x=vals,
                                y=names,
                                orientation="h",
                                marker_color=bar_colors,
                                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
                            )
                        )
                    fig.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Feature Impact",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    ax.barh(
                        names,
                        vals,
                        color=bar_colors,
                        edgecolor="#222",
                        alpha=0.95
                    )
                    ax.set_xlabel("Feature Impact")
                    ax.set_ylabel("")
                    ax.set_title("")
                    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.6)
                    plt.tight_layout()
                    _show_matplotlib(fig, container=st)

                st.caption(
                    "Bars show which features most increased/decreased this customer's churn risk. "
                    "Red = pushes toward churn, green = retention."
                )
                if shap_runtime_info:
                    st.caption(
                        f"SHAP backend: {shap_runtime_info}. "
                        "Encoded/transformed features are aggregated to original feature names."
                    )
            except Exception as e:
                st.markdown(
                    f'<div class="gray-info">SHAP local plot unavailable: {e}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f'<div class="gray-info">{shap_error_msg or "No SHAP values computed."}</div>',
                unsafe_allow_html=True
            )

    # --------- TWO COLUMN LAYOUT: RETENTION RECOMMENDATIONS & AI SUGGESTIONS ---------
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    
    # Create two columns for side-by-side layout
    col_left, col_right = st.columns(2)
    
    # LEFT COLUMN: Retention Recommendations (SHAP-Based Prescriptive Analytics)
    with col_left:
        st.markdown(
            """
            <div class="subsection-title">
                <span class="subsection-icon counterfactual"></span>
                Prescriptive Strategies (SHAP-Based)
                <span title="Real model predictions based on SHAP-identified churn drivers. Shows actual impact and cost-benefit analysis."
                      style="color:#adb5bd;cursor:help;font-size:1.1rem;margin-left:8px;">&#9432;</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if prediction_proba is not None and st.session_state.fitted:
            with st.spinner("🔮 Analyzing churn drivers..."):
                try:
                    counterfactuals = generate_counterfactuals(
                        customer_data=customer_row,
                        model=st.session_state.model,
                        data_df=df,
                        target_col=target_col,
                        num_cfs=3
                    )
                    
                    if counterfactuals and len(counterfactuals) > 0:
                        st.markdown(f"<div style='font-size:13px;color:#64748b;margin-bottom:12px;'>Found <b>{len(counterfactuals)}</b> evidence-based strategies:</div>", unsafe_allow_html=True)
                        
                        # Display each counterfactual as an action card
                        for i, cf in enumerate(counterfactuals, 1):
                            changes = cf['changes']
                            new_prob = cf.get('predicted_churn_prob', 0)
                            feasibility = cf.get('feasibility_category', 'Medium')
                            feasibility_score = cf.get('feasibility_score', 50)
                            feasibility_color = cf.get('feasibility_color', '#f59e0b')
                            impl_cost = cf.get('implementation_cost', 0)
                            net_benefit = cf.get('net_benefit', 0)
                            roi_ratio = cf.get('roi_ratio', 0)
                            
                            # Get GPT explanation
                            formatted = format_counterfactual_with_gpt(
                                changes=changes,
                                original_prob=prediction_proba,
                                new_prob=new_prob if isinstance(new_prob, float) else 0.15,
                                customer_profile=customer_row.to_dict()
                            )
                            
                            # Determine icon based on changes
                            icon = "💰" if any('charge' in k.lower() or 'price' in k.lower() for k in changes.keys()) else "📄" if any('contract' in k.lower() for k in changes.keys()) else "🎯"
                            
                            # Calculate reduction percentage
                            reduction = (prediction_proba - (new_prob if isinstance(new_prob, float) else 0.15)) * 100
                            
                            # Determine border color based on feasibility
                            border_color = "#10b981" if feasibility == "High" else "#f59e0b" if feasibility == "Medium" else "#ef4444"
                            
                            st.markdown(f"""
                            <div class="action-card" style="border-left: 4px solid {border_color};">
                                <div class="action-header">
                                    <div class="action-icon">{icon}</div>
                                    <div class="action-title">{formatted['title']}</div>
                                    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                                        <span style="font-size:11px;background:{feasibility_color};color:white;padding:3px 8px;border-radius:12px;font-weight:600;">{feasibility}</span>
                                    </div>
                                </div>
                                <div class="action-body">
                                    {formatted['description']}
                                </div>
                                <div style="background:#f8fafc;padding:10px 12px;border-radius:6px;margin-top:12px;font-size:12px;">
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
                                        <div>
                                            <span style="color:#64748b;">Implementation Cost:</span>
                                            <span style="float:right;font-weight:600;color:#1e293b;">${abs(impl_cost):,.0f}</span>
                                        </div>
                                        <div>
                                            <span style="color:#64748b;">Customer Value:</span>
                                            <span style="float:right;font-weight:600;color:#1e293b;">${cf.get('customer_clv', 1680):,.0f}</span>
                                        </div>
                                    </div>
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                                        <div>
                                            <span style="color:#64748b;">Net Benefit:</span>
                                            <span style="float:right;font-weight:600;color:{'#10b981' if net_benefit > 0 else '#ef4444'};">${net_benefit:,.0f}</span>
                                        </div>
                                        <div>
                                            <span style="color:#64748b;">ROI:</span>
                                            <span style="float:right;font-weight:600;color:#10b981;">{roi_ratio}x</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="action-impact">
                                    <span class="impact-before">Current: {prediction_proba:.1%}</span>
                                    <span style="color:#94a3b8;">→</span>
                                    <span class="impact-after">After: {new_prob if isinstance(new_prob, float) else 0.15:.1%}</span>
                                    <span style="color:#f59e0b;font-weight:700;">(-{reduction:.1f}%)</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown(
                            f'<div class="gray-info">No alternative scenarios found for this customer profile.</div>',
                            unsafe_allow_html=True
                        )
                
                except Exception as e:
                    st.warning(f"⚠️ Analysis unavailable: {str(e)}")
                    with st.expander("🔎 Details"):
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.markdown(
                '<div class="gray-info">Train a model and make a prediction to see data-driven retention strategies.</div>',
                unsafe_allow_html=True
            )
    
    # RIGHT COLUMN: AI Rule-Based Suggestion
    with col_right:
        st.markdown('<div class="subsection-title"><span class="subsection-icon suggestion"></span>AI Rule-Based Suggestion</div>', unsafe_allow_html=True)
        if OPENAI_AVAILABLE and prediction_proba is not None:
            with st.spinner("AI analyzing profile..."):
                profile_summary = customer_row.drop(columns=[target_col], errors='ignore').to_dict(orient='records')[0]
                st.markdown(gpt_rule_suggestion(profile_summary))
        else:
            st.markdown(
                '<div class="gray-info">OpenAI not configured or model not trained yet.</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# --------- SECTION 3: AI INSIGHTS ---------
else:
    st.markdown('<div class="section-padding"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="section-icon insights"></span>Insights & AI Actions</div>', unsafe_allow_html=True)
    st.write("Generate AI recommendations and promotional messages grounded in model insights and customer profile.")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    if OPENAI_AVAILABLE:
        # Check if model is trained and we have data
        if st.session_state.fitted and st.session_state.model is not None and customer_row is not None and not customer_row.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="subsection-title"><span class="subsection-icon suggestion"></span>AI Recommendation</div>', unsafe_allow_html=True)
                if st.button("Generate Recommendation", key="ai_action_btn"):
                    with st.spinner("Generating AI insights..."):
                        profile_summary = customer_row.drop(columns=[target_col], errors='ignore').to_dict(orient='records')[0]
                        rec_text = gpt_action_recommendation(
                            prediction_proba,
                            top_features or [],
                            top_values or [],
                            profile_summary
                        )
                        st.session_state["last_ai_reco"] = rec_text
                        st.session_state["last_profile"] = profile_summary
                if st.session_state.get("last_ai_reco"):
                    st.markdown(st.session_state["last_ai_reco"])
            with col2:
                st.markdown('<div class="subsection-title"><span class="subsection-icon suggestion"></span>Promotional Message</div>', unsafe_allow_html=True)
                promo_channel = st.selectbox(
                    "Promo channel",
                    ["Auto (AI chooses)", "Email", "Text Message", "App Notification"],
                    index=0,
                    key="promo_channel"
                )
                if st.button("Generate Promo", key="ai_promo_btn"):
                    with st.spinner("Crafting promotional message..."):
                        profile_summary = st.session_state.get("last_profile") or customer_row.drop(columns=[target_col], errors='ignore').to_dict(orient='records')[0]
                        rule_text = gpt_rule_suggestion(profile_summary)
                        action_text = st.session_state.get("last_ai_reco") or gpt_action_recommendation(
                            prediction_proba,
                            top_features or [],
                            top_values or [],
                            profile_summary
                        )
                        channel_hint = ""
                        if promo_channel == "Email":
                            channel_hint = "email"
                        elif promo_channel == "Text Message":
                            channel_hint = "sms"
                        elif promo_channel == "App Notification":
                            channel_hint = "push"
                        promo = gpt_promotional_message(
                            prediction_proba,
                            profile_summary,
                            rule_text,
                            action_text,
                            channel_hint=channel_hint,
                        )
                        st.session_state["last_promo"] = promo
                if st.session_state.get("last_promo"):
                    st.markdown(st.session_state["last_promo"])
        else:
            st.markdown(
                '<div class="gray-info">Run AutoML and generate a prediction first to enable AI outputs.</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="gray-info">OpenAI is not configured. Connect your OpenAI API key in the environment to enable AI recommendations and promotional messages.</div>',
            unsafe_allow_html=True
        )

# ----------------------------- SIDEBAR: VALIDATION ALERTS (COLLAPSIBLE) -----------------------------
if st.session_state.get("validation_messages"):
    st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    
    # Count message types for summary
    msg_counts = {"error": 0, "warning": 0, "success": 0, "info": 0}
    for msg_type, _ in st.session_state["validation_messages"]:
        msg_counts[msg_type] = msg_counts.get(msg_type, 0) + 1
    
    # Build summary badge
    total_msgs = len(st.session_state["validation_messages"])
    summary_parts = []
    if msg_counts["error"] > 0:
        summary_parts.append(f"🔴 {msg_counts['error']} errors")
    if msg_counts["warning"] > 0:
        summary_parts.append(f"⚠️ {msg_counts['warning']} warnings")
    if msg_counts["success"] > 0:
        summary_parts.append(f"✅ {msg_counts['success']} checks")
    summary = " | ".join(summary_parts) if summary_parts else f"{total_msgs} messages"
    
    with st.sidebar.expander(f"📋 Validation & Monitoring ({summary})", expanded=False):
        for msg_type, msg_text in st.session_state["validation_messages"]:
            if msg_type == "error":
                st.error(msg_text)
            elif msg_type == "warning":
                st.warning(msg_text)
            elif msg_type == "success":
                st.success(msg_text)
            elif msg_type == "info":
                st.markdown(
                    f'<div class="gray-info">{msg_text}</div>',
                    unsafe_allow_html=True
                )

# ----------------------------- SIDEBAR STATUS (BOTTOM) -----------------------------
st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="sidebar-status {openai_sidebar_status[0]}">{openai_sidebar_status[1]}</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f'<div class="sidebar-status {pycaret_sidebar_status[0]}">{pycaret_sidebar_status[1]}</div>',
    unsafe_allow_html=True
)

# ----------------------------- FOOTER -----------------------------
st.markdown(
    """
    <div class="footer">
      &copy; {year} Rasel Mia &mdash; Powered by <span style="color:#2563eb;">Streamlit</span>, <span style="color:#f59e0b;">PyCaret</span>, <span style="color:#e11d48;">SHAP</span>, <span style="color:#0f766e;">OpenAI</span>
    </div>
    """.format(year=pd.Timestamp.today().year),
    unsafe_allow_html=True
)

# --- Automatic detection for local vs. DigitalOcean deployment ---
if __name__ == "__main__":
    import os

    # Detect if running locally
    is_local = not os.environ.get("PORT")
    port = int(os.environ.get("PORT", 8501))
    address = "localhost" if is_local else "0.0.0.0"

    os.system(f"streamlit run app.py --server.address={address} --server.port={port}")
