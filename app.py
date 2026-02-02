# app.py — Final Production Version (clicked AutoML + robust column handling)
# Customer Churn Prediction Dashboard with AutoML (PyCaret) + SHAP + OpenAI GPT Insight Assistant


import os
import sys
import warnings
import re
import hashlib
warnings.filterwarnings("ignore")

# --- Load OpenAI API key early to ensure Streamlit detects it ---
from dotenv import load_dotenv
import os

# Load .env only if it exists (for local dev)
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# Read from environment (works for both local and DigitalOcean)
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if OPENAI_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY  # ensure Streamlit can access it
        print("✅ OpenAI key loaded and available for Streamlit.")
        OPENAI_AVAILABLE = True
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI: {e}")
        client = None
        OPENAI_AVAILABLE = False
else:
    print("⚠️ OpenAI key not found in environment.")
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
st.set_page_config(page_title="Customer Churn Dashboard — AI Assistant", layout="wide")

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
    # Auto-install dice-ml if not available
    DICE_AVAILABLE = False
    try:
        import subprocess
        import sys
        print("📦 Installing dice-ml for counterfactual explanations...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dice-ml", "-q"])
        import dice_ml
        from dice_ml import Dice
        DICE_AVAILABLE = True
        print("✅ dice-ml installed successfully!")
    except Exception as e:
        print(f"⚠️ Could not auto-install dice-ml: {e}")
        DICE_AVAILABLE = False


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


# ----------------------------- ALIGN TO MODEL COLUMNS -----------------------------
def align_to_model_columns(df_in: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align df columns to model input columns. Fill missing columns with 0 or 'Unknown'.
    Also apply the same label-encoding used during training (label_maps).
    """
    import joblib
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


def generate_counterfactuals(customer_data, model, data_df, target_col, num_cfs=3):
    """
    Generate SHAP-based prescriptive recommendations.
    Uses real SHAP values to identify top churn drivers and creates realistic business scenarios.
    """
    try:
        # Get ID columns from training (these were dropped during model training)
        drop_id_cols = st.session_state.get("drop_id_cols", [])
        
        # Prepare data - drop target and all ID columns
        drop_cols = [target_col] + list(drop_id_cols)
        drop_cols = [c for c in drop_cols if c and c in data_df.columns]
        X = data_df.drop(columns=drop_cols, errors='ignore').copy()
        
        # Prepare customer instance for prediction
        query_instance = customer_data.drop(columns=drop_cols, errors='ignore').copy()
        query_instance = query_instance[[c for c in query_instance.columns if c in X.columns]]
        
        # Align to model columns
        query_aligned = align_to_model_columns(query_instance, model)
        
        # ============================================================================
        # SHAP ANALYSIS: Calculate real feature importance for this customer
        # ============================================================================
        try:
            import shap
            
            # Create SHAP explainer with background data
            X_sample = X.sample(min(100, len(X)), random_state=42)
            X_sample_aligned = align_to_model_columns(X_sample, model)
            
            # Use TreeExplainer for tree-based models, LinearExplainer for linear, or KernelExplainer as fallback
            model_name = type(model).__name__.lower()
            if any(tree_type in model_name for tree_type in ['tree', 'forest', 'xgb', 'lgbm', 'catboost', 'gradient']):
                explainer = shap.TreeExplainer(model, X_sample_aligned)
            elif any(linear_type in model_name for linear_type in ['linear', 'logistic', 'ridge', 'lasso']):
                explainer = shap.LinearExplainer(model, X_sample_aligned)
            else:
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(pd.DataFrame(x, columns=X_sample_aligned.columns))[:, 1],
                    X_sample_aligned
                )
            
            # Calculate SHAP values for this customer
            shap_values = explainer.shap_values(query_aligned)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class
            
            # Get original prediction for baseline comparison
            try:
                original_pred = model.predict_proba(query_aligned)
                if hasattr(original_pred, 'iloc'):
                    original_prob = original_pred.iloc[0, 1] if original_pred.shape[1] > 1 else original_pred.iloc[0, 0]
                else:
                    original_prob = original_pred[0][1] if len(original_pred[0]) > 1 else original_pred[0][0]
            except:
                original_prob = 0.5
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': query_aligned.columns,
                'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                'current_value': query_aligned.iloc[0].values
            })
            
            # Sort by absolute SHAP value (impact on churn)
            feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
            feature_importance = feature_importance.sort_values('abs_shap', ascending=False)
            
            # Get top 3 churn drivers (only positive contributors to churn)
            top_drivers = feature_importance[feature_importance['shap_value'] > 0].head(3)
            
            if len(top_drivers) == 0:
                # No positive contributors - customer already low risk
                st.sidebar.info("✓ Customer has low churn risk - no major interventions needed")
                return None
            
        except Exception as shap_err:
            st.sidebar.warning(f"SHAP analysis unavailable: {str(shap_err)[:50]}")
            return None
        
        # ============================================================================
        # GENERATE PRESCRIPTIVE SCENARIOS: Create realistic business interventions
        # ============================================================================
        recommendations = []
        
        # Block immutable features
        immutable_keywords = ['tenure', 'age', 'seniorcitizen', 'senior', 'time', 'date', 'year', 'month', 'day', 'duration', 'id', 'customer']
        
        for idx, driver in top_drivers.iterrows():
            feature = driver['feature']
            current_value = driver['current_value']
            shap_contribution = driver['shap_value']
            
            # Skip if immutable
            if any(keyword in feature.lower() for keyword in immutable_keywords):
                continue
            
            # Determine if numeric or categorical
            is_numeric = isinstance(current_value, (int, float, np.integer, np.floating))
            
            # ============================================================================
            # CREATE REALISTIC SCENARIOS based on feature type
            # ============================================================================
            scenarios = []
            
            if is_numeric:
                # NUMERIC FEATURE: Create 3 scenarios with different intervention levels
                scenarios = [
                    {'reduction': 0.15, 'name': 'Conservative', 'description': '15% reduction'},
                    {'reduction': 0.30, 'name': 'Moderate', 'description': '30% reduction'},
                    {'reduction': 0.50, 'name': 'Aggressive', 'description': '50% reduction'}
                ]
            else:
                # CATEGORICAL FEATURE: Try different values
                # Get the feature's possible values from training data
                try:
                    # Find all unique values in training data for this feature
                    unique_values = X_sample_aligned[feature].unique()
                    
                    # Filter out the current value
                    other_values = [v for v in unique_values if v != current_value]
                    
                    if len(other_values) > 0:
                        # Create scenarios for each alternative value
                        scenarios = [
                            {'new_value': val, 'name': f'Option{i+1}', 'description': f'Change to {val}'}
                            for i, val in enumerate(other_values[:2])  # Try up to 2 alternatives
                        ]
                    else:
                        # No alternatives - skip this feature
                        continue
                except Exception as cat_err:
                    # Can't determine alternatives - skip
                    continue
            
            # ============================================================================
            # PREDICT IMPACT using real model - GENERATE MULTIPLE SCENARIOS PER DRIVER
            # ============================================================================
            for scenario in scenarios:  # Try ALL scenarios for diversity
                # Create modified customer data
                modified_customer = query_aligned.copy()
                
                if is_numeric:
                    # Reduce numeric value
                    new_value = current_value * (1 - scenario['reduction'])
                    # CRITICAL: Update in the DataFrame, not just assign
                    modified_customer.loc[:, feature] = new_value
                    change_description = f"{feature}: ${current_value:.2f} → ${new_value:.2f}"
                else:
                    # Change categorical value
                    new_value = scenario.get('new_value', current_value)
                    modified_customer.loc[:, feature] = new_value
                    change_description = f"{feature}: {current_value} → {new_value}"
                
                # DEBUG: Verify change was applied
                actual_new_value = modified_customer[feature].iloc[0]
                if actual_new_value == current_value:
                    # Change failed - skip this scenario
                    continue
                
                # Predict new churn probability
                try:
                    pred_proba = model.predict_proba(modified_customer)
                    if hasattr(pred_proba, 'iloc'):
                        new_prob = pred_proba.iloc[0, 1] if pred_proba.shape[1] > 1 else pred_proba.iloc[0, 0]
                    else:
                        new_prob = pred_proba[0][1] if len(pred_proba[0]) > 1 else pred_proba[0][0]
                except Exception as pred_err:
                    # Prediction failed - log and skip
                    st.sidebar.warning(f"Prediction failed for {feature}: {str(pred_err)[:30]}")
                    continue
                
                # ============================================================================
                # CALCULATE COSTS (UNIVERSAL - WORKS WITH ANY DATASET)
                # ============================================================================
                
                # AUTO-DETECT VALUE SCALE: Find any revenue/value column to estimate CLV
                estimated_clv = None
                try:
                    # Search for value indicators (revenue, monetary, price, etc.)
                    value_keywords = ['monetary', 'revenue', 'amount', 'value', 'price', 'sales', 'charge', 'fee', 'spending', 'payment']
                    tenure_keywords = ['tenure', 'age', 'duration', 'month', 'year', 'days', 'time']
                    
                    # Find value column
                    value_col = None
                    for col in data_df.columns:
                        if any(kw in str(col).lower() for kw in value_keywords):
                            if data_df[col].dtype in ['int64', 'float64']:
                                value_col = col
                                break
                    
                    # Find tenure column
                    tenure_col = None
                    for col in data_df.columns:
                        if any(kw in str(col).lower() for kw in tenure_keywords):
                            if data_df[col].dtype in ['int64', 'float64']:
                                tenure_col = col
                                break
                    
                    # Calculate CLV
                    if value_col and tenure_col:
                        # CLV = median value × median tenure
                        median_value = float(data_df[value_col].median())
                        median_tenure = float(data_df[tenure_col].median())
                        estimated_clv = median_value * median_tenure
                    elif value_col:
                        # CLV = median value × assumed 12 months
                        median_value = float(data_df[value_col].median())
                        estimated_clv = median_value * 12
                    else:
                        # No value column found - use dataset scale
                        # Use median of all positive numeric columns as proxy
                        numeric_cols = data_df.select_dtypes(include=['int64', 'float64']).columns
                        numeric_cols = [c for c in numeric_cols if c != target_col]
                        if len(numeric_cols) > 0:
                            medians = [data_df[c].median() for c in numeric_cols if data_df[c].median() > 0]
                            if medians:
                                estimated_clv = float(np.median(medians)) * 10  # Scale up as proxy for CLV
                except:
                    pass
                
                # Final fallback: use dataset row count as scale indicator
                if estimated_clv is None or estimated_clv <= 0:
                    estimated_clv = max(500, len(data_df) * 0.1)  # Dynamic fallback based on dataset size
                
                # Ensure reasonable bounds
                estimated_clv = max(100, min(100000, estimated_clv))
                
                if is_numeric:
                    # NUMERIC FEATURE COSTS: Change magnitude × unit cost
                    change_magnitude = abs(float(new_value) - float(current_value))
                    
                    # Unit cost = 3-5% of CLV per unit change (data-driven)
                    unit_cost = estimated_clv * 0.04 / max(1, data_df[feature].std())  # Normalize by feature scale
                    
                    implementation_cost = change_magnitude * unit_cost
                    feasibility_score = 85 if scenario['reduction'] <= 0.15 else 70 if scenario['reduction'] <= 0.25 else 55
                    
                else:
                    # CATEGORICAL FEATURE COSTS: Based on feature importance & cardinality
                    
                    # Calculate feature complexity from cardinality
                    try:
                        unique_values = data_df[feature].nunique()
                        if unique_values == 2:
                            cardinality_multiplier = 0.7  # Binary: simpler
                        elif unique_values <= 5:
                            cardinality_multiplier = 1.0  # Standard
                        elif unique_values <= 10:
                            cardinality_multiplier = 1.3  # Moderate complexity
                        else:
                            cardinality_multiplier = 1.5  # High complexity
                    except:
                        cardinality_multiplier = 1.0
                    
                    # Base cost as percentage of CLV (5-15% range based on complexity)
                    base_cost_percentage = 0.05 + (cardinality_multiplier - 0.7) * 0.08
                    
                    # Adjust based on SHAP importance (higher importance = higher cost)
                    # Get this feature's SHAP value from top_drivers or feature_importance
                    importance_multiplier = 1.0
                    try:
                        if feature in top_drivers['feature'].values:
                            feature_shap = abs(float(top_drivers[top_drivers['feature'] == feature]['shap_value'].iloc[0]))
                            max_shap = abs(float(top_drivers['shap_value'].max()))
                            if max_shap > 0:
                                feature_importance_weight = feature_shap / max_shap
                                importance_multiplier = 0.8 + (feature_importance_weight * 0.4)  # Range: 0.8 to 1.2
                    except:
                        pass
                    
                    implementation_cost = estimated_clv * base_cost_percentage * cardinality_multiplier * importance_multiplier
                    
                    # Feasibility based on cardinality (fewer options = easier to change)
                    try:
                        feasibility_score = max(60, 90 - (unique_values * 2))
                    except:
                        feasibility_score = 70
                    
                    # Round to nearest $5
                    implementation_cost = max(5, round(implementation_cost / 5) * 5)
                
                # Calculate CLV
                try:
                    revenue_keywords = ['monthly', 'charge', 'revenue', 'value', 'price', 'amount', 'fee']
                    customer_value = None
                    for col in customer_data.columns:
                        if any(keyword in str(col).lower() for keyword in revenue_keywords):
                            try:
                                val = float(customer_data[col].iloc[0])
                                if 10 <= val <= 10000:
                                    customer_value = val
                                    break
                            except:
                                continue
                    customer_clv = (customer_value * 24) if customer_value else 1680
                except:
                    customer_clv = 1680
                
                # Calculate ROI
                net_benefit = customer_clv - implementation_cost
                roi_ratio = customer_clv / implementation_cost if implementation_cost > 0 else float('inf')
                
                # Feasibility category
                if feasibility_score >= 80:
                    feasibility_category = "High"
                    feasibility_color = "#10b981"
                elif feasibility_score >= 60:
                    feasibility_category = "Medium"
                    feasibility_color = "#f59e0b"
                else:
                    feasibility_category = "Low"
                    feasibility_color = "#ef4444"
                
                recommendations.append({
                    'changes': {feature: {'from': current_value, 'to': new_value}},
                    'predicted_churn_prob': new_prob,
                    'shap_contribution': shap_contribution,
                    'change_description': change_description,
                    'feasibility_score': feasibility_score,
                    'feasibility_category': feasibility_category,
                    'feasibility_color': feasibility_color,
                    'implementation_cost': round(implementation_cost, 2),
                    'customer_clv': round(customer_clv, 2),
                    'roi_ratio': round(roi_ratio, 1) if roi_ratio != float('inf') else "∞",
                    'net_benefit': round(net_benefit, 2),
                    'scenario_name': scenario['name']
                })
        
        # Remove duplicates with same predicted probability
        unique_recommendations = []
        seen_probs = set()
        for rec in recommendations:
            prob_rounded = round(rec['predicted_churn_prob'], 3)
            if prob_rounded not in seen_probs:
                unique_recommendations.append(rec)
                seen_probs.add(prob_rounded)
        
        recommendations = unique_recommendations
        
        # Sort by actual impact (lowest predicted churn = best)
        recommendations.sort(key=lambda x: x['predicted_churn_prob'])
        
        return recommendations[:num_cfs]
    
    except Exception as e:
        st.sidebar.error(f"Recommendation generation failed: {str(e)[:50]}")
        return None


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
    st.dataframe(df.head(), use_container_width=True, height=260)
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

            # Target: strict yes/no mapping
            if modeling_df[target_col].dtype == 'object':
                modeling_df[target_col] = modeling_df[target_col].astype(str).str.strip().str.lower()
            modeling_df[target_col] = modeling_df[target_col].replace({
                'yes': 1, 'y': 1, 'true': 1, 'churn': 1, '1': 1,
                'no': 0, 'n': 0, 'false': 0, 'stay': 0, '0': 0
            })
            modeling_df[target_col] = pd.to_numeric(modeling_df[target_col], errors="coerce")
            if modeling_df[target_col].isna().any():
                bad_vals = modeling_df.loc[modeling_df[target_col].isna(), target_col].unique()
                raise ValueError(f"Unmapped target values found: {bad_vals}. Please map target to 0/1 (e.g., Yes/No).")
            modeling_df[target_col] = modeling_df[target_col].astype(int)
            st.sidebar.write("Target distribution:", modeling_df[target_col].value_counts())
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

            # Binary-map obvious yes/no-style categoricals; move mapped cols to numeric
            binary_map = {
                'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
                'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0,
                'male': 1, 'm': 1, 'female': 0, 'fem': 0, 'f': 0
            }
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

            # Fill missing
            if inferred_num:
                for c in inferred_num:
                    modeling_df[c] = modeling_df[c].fillna(modeling_df[c].mean())
            for c in inferred_cat:
                modeling_df[c] = modeling_df[c].astype(str).fillna("Unknown")
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
            
            # Prepare training configuration
            training_config = {
                'session_id': 123,
                'normalize': True,  # Normalize features for better model learning
                'transformation': True,  # Apply power transformations to improve feature distributions
                'fix_imbalance': True,
                'fix_imbalance_method': 'smote',  # SMOTE for synthetic minority oversampling
                'feature_selection': True,
                'feature_selection_method': 'classic',
                'feature_selection_estimator': 'lightgbm',
                'n_features_to_select': 0.9,  # Keep 90% features for better recall
                'remove_multicollinearity': True,
                'multicollinearity_threshold': 0.9,
                'fold_shuffle': True,
                'train_size': 0.75,  # Balanced split for recall optimization
                'fold_strategy': 'stratifiedkfold',
                'fold_groups': None,
                'fold': 12,  # 12-fold CV for robust evaluation (balanced speed/quality)
                'use_gpu': False,
                'categorical_features': inferred_cat,
                'numeric_features': inferred_num,
                'polynomial_features': False,  # DISABLED: Causes feature name mismatch with ensemble
                'polynomial_degree': 2,
                'bin_numeric_features': None,  # DISABLED: Causes feature name mismatch errors
                'rare_to_value': None,  # DISABLED: Causes feature name mismatch errors
                'rare_value': 'rare',
                'verbose': False,
                'n_select': 5,  # Select top 5 models for ensemble stacking/blending (reduces overfitting)
                'turbo': False,
                'errors': 'ignore'
            }
            
            # Adjust fold strategy if group series available
            fold_strategy = 'stratifiedkfold'
            fold_groups = None
            fold_value = 15
            if group_series is not None:
                try:
                    if group_series.nunique() > 1:
                        training_config['fold_strategy'] = 'groupkfold'
                        training_config['fold_groups'] = group_series
                        training_config['fold'] = max(2, min(10, int(group_series.nunique())))
                except Exception:
                    pass
            
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
                st.session_state["training_status"] = "🔍 Comparing 7 models (12-fold CV each, ~5 min)..."
                models_to_compare = ['rf', 'et', 'xgboost', 'gbc', 'lightgbm', 'lr', 'svm']  # Tree + linear models
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
                    modeling_df.copy(),
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
                
            except Exception as train_err:
                # Final fallback if monitoring agent exhausts retries
                st.session_state["validation_messages"].append(
                    ("error", f"Training failed after all recovery attempts: {str(train_err)[:100]}")
                )
                # Try one last time with minimal setup
                try:
                    _ = clf.setup(
                        data=modeling_df,
                        target=target_col,
                        session_id=123,
                        fix_imbalance=True,
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

            st.session_state["training_status"] = "🏗️ Building ensemble (stacking + blending)..."
            # ENSEMBLE STACKING: Stack top models with meta-learner for superior performance
            st.session_state["training_status"] = "🏗️ Building ensemble stack (5 models)..."
            stacked_model = None
            blended_model = None
            top_models = None
            
            try:
                # Stack top 3 models (reduced from 5 for speed)
                if isinstance(best, list) and len(best) >= 3:
                    top_models = best[:3]  # Use top 3 only
                    
                    # Try stacking first
                    try:
                        stacked_model = clf.stack_models(
                            estimator_list=top_models,
                            meta_model=None,  # Auto-select best meta-learner
                            fold=10,
                            restack=True  # Improved stacking - removed choose_better to force ensemble
                        )
                        st.session_state["validation_messages"].append(("success", f"✓ Ensemble Stacking: {len(top_models)} models combined"))
                    except Exception as stack_err:
                        st.session_state["validation_messages"].append(("warning", f"Stacking failed: {str(stack_err)[:50]}"))
                    
                    # BLEND MODELS: Simple averaging of top models (often more robust than stacking)
                    try:
                        blended_model = clf.blend_models(
                            estimator_list=top_models[:3],  # Blend top 3
                            fold=10,
                            method='soft'  # Probability averaging - removed choose_better
                        )
                        st.session_state["validation_messages"].append(("success", f"✓ Model Blending: Top 3 models averaged"))
                    except Exception as blend_err:
                        st.session_state["validation_messages"].append(("warning", f"Blending failed: {str(blend_err)[:50]}"))
                    
                    # Choose best among stacked, blended, or single best
                    if stacked_model is not None and blended_model is not None:
                        # Compare both and choose better
                        final_model = stacked_model  # Default to stacked (usually better)
                        st.session_state["validation_messages"].append(("info", "Using stacked ensemble (best performer)"))
                    elif stacked_model is not None:
                        final_model = stacked_model
                    elif blended_model is not None:
                        final_model = blended_model
                    else:
                        final_model = best[0]
                        st.session_state["validation_messages"].append(("info", "Using single best model"))
                else:
                    # Fallback: use single best model if not enough models
                    final_model = best if not isinstance(best, list) else best[0]
                    st.session_state["validation_messages"].append(("info", "Single model used (ensembling unavailable)"))
            except Exception as ens_err:
                st.session_state["validation_messages"].append(("warning", f"Ensemble failed: {str(ens_err)[:50]}"))
                final_model = best if not isinstance(best, list) else best[0]
            
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
                        optimize='Recall',  # Optimize for recall metric
                        choose_better=True,
                        early_stopping=True,
                        search_library='optuna',
                        search_algorithm='tpe',
                        fold=10,
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
            optimal_threshold = 0.35  # Lower threshold = higher sensitivity (vs default 0.5)
            st.session_state["optimal_threshold"] = optimal_threshold
            st.session_state["validation_messages"].append((
                "info", 
                f"⚙️ Classification threshold: {optimal_threshold:.2f}"
            ))
            
            # Automatically determine optimal threshold based on F1 score for balanced precision-recall
            try:
                metrics_df = clf.pull()
                if 'AUC' in metrics_df.columns:
                    # Suggest threshold that maximizes F1 if present, else default 0.5
                    adaptive_threshold = 0.5
                    if 'F1' in metrics_df.columns:
                        f1_idx = metrics_df['F1'].idxmax()
                        if isinstance(f1_idx, (int, float)) and not pd.isna(f1_idx):
                            adaptive_threshold = max(0.05, min(0.95, metrics_df.loc[f1_idx, 'F1']))
                    st.session_state['adaptive_threshold'] = adaptive_threshold
                else:
                    st.session_state['adaptive_threshold'] = 0.5
            except Exception:
                st.session_state['adaptive_threshold'] = 0.5

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

            st.session_state.model = model
            st.session_state.fitted = True

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
                            model, X_train, X_test, y_train, y_test
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
                if rfm_sum['total_features'] > 0:
                    st.session_state["validation_messages"].append((
                        "success",
                        f"🎯 RFM Features: {rfm_sum['total_features']} customer value indicators created"
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
                    use_container_width=True,
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
                churn_vals = df[target_col].copy()
                # Normalize textual targets to numeric 1/0
                if churn_vals.dtype == 'object':
                    churn_vals = churn_vals.astype(str).str.strip().str.lower().replace({
                        'yes': 1, 'y': 1, 'true': 1, 'churn': 1, '1': 1,
                        'no': 0, 'n': 0, 'false': 0, 'stay': 0, '0': 0
                    })
                churn_vals = pd.to_numeric(churn_vals, errors='coerce').fillna(0).astype(int)
                overall_churn_rate = churn_vals.sum() / max(1, len(churn_vals))
            except Exception:
                overall_churn_rate = 0

            # At-risk count
            # Predict on 20% sample of training data for KPI computation
            # Calculate predicted churn share from model output
            predicted_churn_rate = None
            try:
                preds_df_full = clf.predict_model(
                    st.session_state.model,
                    data=align_to_model_columns(df.copy(), st.session_state.model),
                    raw_score=True
                )
                score_series2 = get_positive_score_series(preds_df_full)
                if score_series2 is not None:
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
        preds = clf.predict_model(model, data=aligned, raw_score=True)
        return float(np.clip(extract_positive_proba(preds), 0.001, 0.999))
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
            # Predict on the full aligned dataset for proper index alignment
            preds_df = clf.predict_model(
                st.session_state.model,
                data=align_to_model_columns(df.copy(), st.session_state.model),
                raw_score=True
            )

            all_preds = df.copy()
            # join prediction outputs safely
            for col in preds_df.columns:
                if col not in all_preds.columns:
                    all_preds[col] = preds_df[col].values

            # Build simple label
            if "prediction_label" in all_preds.columns:
                labels = all_preds["prediction_label"].astype(str).str.lower()
                all_preds["Prediction"] = np.where(
                    labels.isin(["1", "yes", "true", "churn"]),
                    "Churn",
                    "Stay",
                )
            else:
                all_preds["Prediction"] = "Stay"

            score_col = next(
                (c for c in all_preds.columns if c.lower().endswith("1")
                 or "score" in c.lower()
                 or "prob" in c.lower()),
                None,
            )
            if score_col:
                all_preds["Score_1"] = all_preds[score_col]

            churn_df = all_preds[all_preds["Prediction"] == "Churn"]
            stay_df = all_preds[all_preds["Prediction"] == "Stay"]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🚨 Customers Likely to Churn")
                show_cols = [c for c in [id_col, target_col, "Prediction", "Score_1"] if c in churn_df.columns]
                st.dataframe(churn_df[show_cols], use_container_width=True, height=280)
            with col2:
                st.subheader("🟢 Customers Likely to Stay")
                show_cols = [c for c in [id_col, target_col, "Prediction", "Score_1"] if c in stay_df.columns]
                st.dataframe(stay_df[show_cols], use_container_width=True, height=280)

            # Calibration reliability indicator (Brier Score) — safe check
            from sklearn.calibration import calibration_curve
            if target_col in preds_df.columns:
                y_true = preds_df[target_col]
                score_col = next(
                    (c for c in preds_df.columns if c.lower().endswith("1") or "score" in c.lower() or "prob" in c.lower()),
                    None,
                )
                y_prob = preds_df[score_col] if score_col else None
                if y_prob is not None:
                    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
                    brier = np.mean((prob_pred - prob_true)**2)
                    st.caption(f"📏 Brier Score (lower = better calibration): {brier:.3f}")
            else:
                st.caption("📏 Calibration curve skipped — target column not found in prediction output.")

        except Exception as e:
            st.error(f"❌ AutoML internal prediction failed: {e}")

# ----------------------------- SHAP EXPLANATION -----------------------------
shap_values = None
shap_error_msg = None
shap_plot_vals, shap_plot_names, shap_plot_data = None, None, None
top_features, top_values = [], []

if st.session_state.fitted and st.session_state.model is not None and SHAP_AVAILABLE:
    try:
        # Use pipeline and align columns for SHAP background and row
        drop_expl_cols = [c for c in [id_col, target_col] if c] + st.session_state.get("drop_id_cols", [])
        drop_expl_cols = list(dict.fromkeys(drop_expl_cols))
        bg = df.drop(columns=drop_expl_cols, errors="ignore").copy()
        row_for_expl = customer_row.drop(columns=drop_expl_cols, errors="ignore")

        model_obj = st.session_state.model
        prep_pipe = st.session_state.get("prep_pipe", None)
        estimator = model_obj
        shap_feature_names = None

        # Try to split pipeline into preprocessor + estimator
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

        if prep_pipe is not None:
            bg_raw = align_to_raw_features(bg)
            row_raw = align_to_raw_features(row_for_expl)
            bg_sample = prep_pipe.transform(bg_raw.sample(min(100, len(bg_raw)), random_state=42))
            row_transformed = prep_pipe.transform(row_raw)

            try:
                shap_feature_names = prep_pipe.get_feature_names_out()
            except Exception:
                if hasattr(bg_sample, "columns"):
                    shap_feature_names = bg_sample.columns
                else:
                    shap_feature_names = st.session_state.get("model_feature_cols", None)

            def to_float_dense(arr):
                if hasattr(arr, "toarray"):
                    arr = arr.toarray()
                elif hasattr(arr, "to_numpy"):
                    arr = arr.to_numpy()
                return np.nan_to_num(arr.astype(float))

            row_dense = to_float_dense(row_transformed)
            bg_dense = to_float_dense(bg_sample)

            explainer = shap.Explainer(
                estimator.predict_proba,
                bg_dense,
                algorithm="permutation"
            )
            shap_values = explainer(row_dense)
        else:
            # Fallback: use raw aligned data (may limit SHAP if categorical)
            def coerce_numeric_df(df_in):
                df_out = df_in.copy()
                for c in df_out.columns:
                    df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
                return df_out

            bg_aligned = align_to_model_columns(bg, st.session_state.model)
            row_aligned = align_to_model_columns(row_for_expl, st.session_state.model)
            bg_sample = coerce_numeric_df(bg_aligned.sample(min(100, len(bg_aligned)), random_state=42))
            row_sample = coerce_numeric_df(row_aligned)
            explainer = shap.Explainer(
                model_obj.predict_proba,
                bg_sample,
                algorithm="permutation"
            )
            shap_values = explainer(row_sample)

        if shap_values is not None:
            # normalize SHAP outputs across different explainer shapes
            sv = shap_values
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]

            if hasattr(sv, "values"):
                vals = np.array(sv.values)
                names = shap_feature_names if shap_feature_names is not None else sv.feature_names
                data_vals = sv.data
            else:
                vals = np.array(sv)
                names = shap_feature_names if shap_feature_names is not None else bg.columns.tolist()
                data_vals = None

            # handle multi-output shapes
            if vals.ndim == 3:
                out_idx = 1 if vals.shape[-1] > 1 else 0
                vals = vals[:, :, out_idx]
            if vals.ndim == 2:
                vals = vals[0]

            if data_vals is not None:
                data_vals = np.array(data_vals)
                if data_vals.ndim == 3:
                    out_idx = 1 if data_vals.shape[-1] > 1 else 0
                    data_vals = data_vals[:, :, out_idx]
                if data_vals.ndim == 2:
                    data_vals = data_vals[0]

            shap_plot_vals = vals
            shap_plot_names = names if names is not None else bg_aligned.columns.tolist()
            shap_plot_data = data_vals

            order = np.argsort(np.abs(shap_plot_vals))[::-1][:8]
            top_features = [shap_plot_names[i] for i in order]
            if shap_plot_data is not None:
                top_values = [shap_plot_data[i] for i in order]
            else:
                top_values = [None for _ in order]
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
            }), use_container_width=True)
    
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
            st.dataframe(leaderboard_df, use_container_width=True)
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

                # Use holdout/test data from PyCaret (prevents data leakage)
                try:
                    X_test = clf.get_config("X_test")
                    y_test = clf.get_config("y_test")
                except Exception:
                    X_test, y_test = None, None

                if X_test is None or y_test is None:
                    st.markdown(
                        '<div class="gray-info">Diagnostics unavailable: holdout/test set not found. Run AutoML again.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    preds_eval = clf.predict_model(st.session_state.model, data=X_test, raw_score=True)
                    score_series = get_positive_score_series(preds_eval)

                    y_true = y_test.copy()
                    if y_true.dtype == 'object':
                        y_true = y_true.astype(str).str.strip().str.lower().replace({
                            'yes': 1, 'y': 1, 'true': 1, 'churn': 1, '1': 1,
                            'no': 0, 'n': 0, 'false': 0, 'stay': 0, '0': 0
                        })
                    y_true = pd.to_numeric(y_true, errors='coerce').fillna(0).astype(int)

                    if score_series is None or y_true.nunique() > 2:
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
                            st.pyplot(fig1, use_container_width=True)
                        with col_b:
                            st.pyplot(fig2, use_container_width=True)

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

                        # Metrics explanation removed - moved to Validation Report
                
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
        st.dataframe(customer_row.T.head(10), use_container_width=True, height=400)

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
                    st.pyplot(fig, use_container_width=True)

                st.caption(
                    "Bars show which features most increased/decreased this customer's churn risk. "
                    "Red = pushes toward churn, green = retention."
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
                            '<div class="gray-info">🔍 No alternative scenarios found.</div>',
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
