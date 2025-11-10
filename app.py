# app.py — Final Production Version (clicked AutoML + robust column handling)
# Customer Churn Prediction Dashboard with AutoML (PyCaret) + SHAP + OpenAI GPT Insight Assistant


import os
import sys
import warnings
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

# ----------------------------- ENV INFO -----------------------------
st.set_page_config(page_title="Customer Churn Dashboard — AI Assistant", layout="wide")

# Show OpenAI API status in sidebar
if OPENAI_AVAILABLE:
    st.sidebar.success("OpenAI API is connected")
else:
    st.sidebar.warning("⚠️ OpenAI API not configured. Some AI features will be unavailable.")

# PyCaret (robust import)
PYCARET_AVAILABLE = True
try:
    import pycaret
    from pycaret import classification as clf
    st.sidebar.success(f"PyCaret version: {pycaret.__version__}")
except Exception as e:
    PYCARET_AVAILABLE = False
    st.sidebar.error(f"❌ PyCaret not detected: {e}")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
.card {
    background-color: #1b263b;
    border-radius: 12px;
    padding: 28px 20px 20px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin-bottom: 18px;
    color: #f8f9fa;
}
.metric-green { color: #2a9d8f; font-weight: 700; font-size: 1.2rem; }
.metric-red { color: #e63946; font-weight: 700; font-size: 1.2rem; }
.metric-primary { color: #0077b6; font-weight: 700; }
.metric-accent { color: #ffb703; font-weight: 700; }
.small-muted { color: #adb5bd; font-size: 12px; }
.kpi-row { display: flex; gap: 36px; justify-content: flex-start; margin: 16px 0 30px 0; }
.kpi-metric {
    background: #0d1b2a;
    border-radius: 10px;
    padding: 20px 26px 14px 26px;
    min-width: 180px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    color: #f8f9fa;
    text-align: center;
}
.kpi-label { font-size: 14px; color: #adb5bd; }
.kpi-value { font-size: 1.6rem; font-weight: bold; }
.section-padding { margin-top: 32px !important; }
.stTabs [role="tab"] { font-size: 1.1rem; padding: 12px 24px !important; }
.gray-info {
    background: #f1f3f5;
    color: #495057;
    border-radius: 8px;
    padding: 18px 18px 14px 18px;
    margin-bottom: 12px;
    font-size: 1.08rem;
    text-align: center;
}
.footer {
    color: #adb5bd;
    background: none;
    text-align: center;
    font-size: 0.98rem;
    margin-top: 36px;
    padding: 18px 0 4px 0;
}
.gradient-header {
    background: linear-gradient(90deg, #0077b6 0%, #1b263b 100%);
    border-radius: 18px;
    padding: 34px 24px 24px 32px;
    margin-bottom: 32px;
    color: #f8f9fa !important;
    box-shadow: 0 3px 18px rgba(0,0,0,0.11);
}
.gradient-title { font-size: 2.6rem; font-weight: 800; color: #fff; letter-spacing: -1px; margin-bottom: 8px; }
.gradient-subtitle { font-size: 1.25rem; color: #ffb703; margin-bottom: 0; }
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
You are a senior customer retention manager analyzing this customer's situation to make a professional business recommendation.

Customer Profile: {customer_profile}

Model Context: {context_info}

Provide your insights in the following structure:
1) Loyalty Stage: (New, Active, Loyal, or At-Risk)
2) Key Observations: 2 concise bullet points highlighting main behavioral or contextual signals.
3) Recommended Action: A short, actionable retention strategy (e.g., targeted offer, proactive call, upgrade suggestion, engagement initiative).
4) Expected Outcome: The likely business impact or customer reaction.
Use a professional business tone and concise phrasing.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise customer insight assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
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
    prompt = f"""
You are a senior business strategy manager providing data-driven recommendations to prevent customer churn and improve retention.

Churn probability: {prob:.2f}
Top feature signals: {feature_summary}
Customer profile: {profile}

Model Context: {context_info}

Please generate a concise executive-level summary with the following:
1) Business Interpretation: Explain the meaning of this prediction for management in one short paragraph.
2) Risk & Value Assessment: Classify the customer (Low/Medium/High value, Low/Medium/High churn risk) and justify.
3) Strategic Action Plan: Provide one clear retention recommendation aligned with business ROI (discount, engagement campaign, feature upsell, loyalty benefit, etc.).
4) Communication Strategy: Recommend how and through which channel to communicate this action (email, app notification, phone call, etc.).
5) Expected ROI Impact: Describe the likely financial or engagement impact if the recommendation is followed.

Keep the language professional, data-driven, and general enough to apply to any business context.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise business analytics assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI call failed: {e}"


## ----------------------------- HEADER GRADIENT BANNER -----------------------------
st.markdown(
    """
    <div class="gradient-header">
      <div class="gradient-title">Customer Churn Prediction Dashboard</div>
      <div class="gradient-subtitle">AI-Powered Insights & Retention Actions</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------- SIDEBAR: UPLOAD & SETTINGS -----------------------------
sidebar = st.sidebar
sidebar.title("Controls & Upload")

uploaded = sidebar.file_uploader("📂 Upload CSV", type=["csv"])
use_demo = sidebar.checkbox("Use demo dataset (default)", value=True)

# Load data
if uploaded is not None:
    df_raw = safe_read_csv(uploaded)
elif use_demo and os.path.exists("processed_churn_dataset.csv"):
    df_raw = pd.read_csv("processed_churn_dataset.csv")
else:
    st.error("Please upload a dataset or place 'processed_churn_dataset.csv' in the app folder.")
    st.stop()

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
        <div class="kpi-row" style="margin-top:15px;">
          <div class="kpi-metric">
            <div class="kpi-label">Total Rows</div>
            <div class="kpi-value" style="color:#00b4d8;">{total_rows:,}</div>
          </div>
          <div class="kpi-metric">
            <div class="kpi-label">Total Columns</div>
            <div class="kpi-value" style="color:#ffb703;">{total_cols}</div>
          </div>
          <div class="kpi-metric">
            <div class="kpi-label">Missing Values</div>
            <div class="kpi-value" style="color:#e63946;">{missing_vals}</div>
          </div>
          <div class="kpi-metric">
            <div class="kpi-label">Data Types</div>
            <div class="kpi-value" style="color:#2a9d8f;">{data_types}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Column selections
cols = df.columns.tolist()
default_target = guess_target_name(cols)
default_id = guess_id_name(cols)

id_choice = sidebar.selectbox("Select Customer ID column (optional)", options=[None] + cols,
                              index=(cols.index(default_id) + 1) if default_id in cols else 0)
target_choice = sidebar.selectbox("Select Target column (required)", options=[None] + cols,
                                  index=(cols.index(default_target) + 1) if default_target in cols else 0)

if not target_choice:
    st.warning("⚠️ Please select the target column to continue.")
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

# Select row to inspect
if id_col:
    ids = df[id_col].astype(str).tolist()
    selected_id = sidebar.selectbox("Select Customer", ids)
    customer_row = df[df[id_col].astype(str) == str(selected_id)].iloc[0:1]
else:
    idx = sidebar.number_input("Row Index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
    customer_row = df.iloc[int(idx):int(idx)+1]

threshold = sidebar.slider("Prediction threshold", 0.0, 1.0, 0.50)
sidebar.caption(
    "🔎 **Threshold guide:** 0.50 = balanced | 0.35–0.30 = more sensitive to churn (detects more at-risk customers) | >0.50 = stricter (fewer false churn alerts)."
)

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

# ----------------------------- RUN AUTOML BUTTON -----------------------------
run_automl = st.button("⚙️ Run AutoML")

if run_automl:
    if not PYCARET_AVAILABLE:
        st.error("PyCaret is not available — please install pycaret first.")
        st.stop()

    with st.spinner("⚙️ Running AutoML... This may take a few minutes, please wait patiently while models are trained and optimized."):
        try:
            from sklearn.preprocessing import LabelEncoder

            # 1) Prepare data and drop ID column if any
            drop_cols = [id_col] if id_col else []
            modeling_df = df.drop(columns=drop_cols, errors="ignore").copy()

            # Explicit check and confirmation that ID column was dropped
            if id_col and id_col in modeling_df.columns:
                st.sidebar.error(f"❌ ID column '{id_col}' was not dropped properly.")
            else:
                if id_col:
                    st.sidebar.info(f"✅ ID column '{id_col}' dropped before training.")

            # 2) Clean column names
            modeling_df.columns = modeling_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
            modeling_df = modeling_df.loc[:, ~modeling_df.columns.duplicated()]

            # 3) Ensure target column exists and encode Yes/No or similar to 1/0
            if target_col not in modeling_df.columns:
                possible = [c for c in modeling_df.columns if c.lower() == target_col.lower()]
                if possible:
                    target_col = possible[0]
                else:
                    raise ValueError(f"Target column '{target_col}' not found after cleaning.")

            # Normalize and encode churn target robustly
            if modeling_df[target_col].dtype == 'object':
                modeling_df[target_col] = modeling_df[target_col].astype(str).str.strip().str.lower()
                modeling_df[target_col] = modeling_df[target_col].replace({
                    'yes': 1, 'y': 1, 'true': 1, 'churn': 1, '1': 1,
                    'no': 0, 'n': 0, 'false': 0, 'stay': 0, '0': 0
                })

            # Ensure binary integer type
            modeling_df[target_col] = modeling_df[target_col].astype(int)
            st.sidebar.write("Target distribution:", modeling_df[target_col].value_counts())

            # 4) Encode all string-type columns (except target) BEFORE imputation
            label_encoders = {}
            string_cols = modeling_df.select_dtypes(include=['object', 'category']).columns.tolist()
            string_cols = [c for c in string_cols if c != target_col]
            for c in string_cols:
                modeling_df[c] = modeling_df[c].astype(str)
                le = LabelEncoder()
                modeling_df[c] = le.fit_transform(modeling_df[c])
                label_encoders[c] = le

            # NEW: store simple mapping dicts instead of sklearn objects
            label_maps = {
                col: {cls: int(i) for i, cls in enumerate(le.classes_)}
                for col, le in label_encoders.items()
            }
            st.session_state.label_maps = label_maps

            # 5) Fill missing values (now all columns are numeric except possibly target)
            for c in modeling_df.select_dtypes(include=[np.number]).columns:
                modeling_df[c] = modeling_df[c].fillna(modeling_df[c].mean())
            # If target column is still object (shouldn't be), handle as string
            if modeling_df[target_col].dtype == 'object':
                modeling_df[target_col] = modeling_df[target_col].astype(str).fillna('Unknown')

            # 6) Enhanced AutoML setup with SMOTE and tuning for better churn sensitivity
            _ = clf.setup(
                data=modeling_df,
                target=target_col,
                session_id=123,
                normalize=True,
                transformation=True,
                fix_imbalance=True,
                fix_imbalance_method='smote',
                feature_selection=True,
                remove_multicollinearity=True,
                multicollinearity_threshold=0.9,
                fold_shuffle=True,
                train_size=0.75,
                fold_strategy='stratifiedkfold',
                fold=15,
                use_gpu=True,
                verbose=False
            )

            # Store reference data
            st.session_state.train_df = modeling_df.copy()

            # Focus on tree-based models only — balanced and robust
            tree_models = ['lightgbm', 'catboost', 'rf', 'et', 'xgboost', 'gbc']

            best = clf.compare_models(
                include=tree_models,
                sort='F1',
                fold=10,
                turbo=False,
                errors='ignore'
            )

            # Tune and calibrate with Optuna + Early Stopping for faster and smarter optimization
            tuned_best = clf.tune_model(
                best,
                optimize='F1',
                choose_better=True,
                early_stopping=True,
                search_library='optuna',
                search_algorithm='tpe',
                fold=10
            )

            # Apply Isotonic Calibration for more realistic churn probabilities
            best_cal = clf.calibrate_model(tuned_best, method='isotonic')

            # Automatically determine optimal threshold based on F1 score for balanced precision-recall
            try:
                metrics_df = clf.pull()
                if 'F1' in metrics_df.columns:
                    f1_idx = metrics_df['F1'].idxmax()
                    adaptive_threshold = 0.5
                    if isinstance(f1_idx, (int, float)) and not pd.isna(f1_idx):
                        adaptive_threshold = metrics_df.loc[f1_idx, 'F1']
                    st.session_state['adaptive_threshold'] = adaptive_threshold
                    st.sidebar.info(f"🔍 Auto-selected best threshold (F1-based): {adaptive_threshold:.2f}")
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

            import glob
            for f in glob.glob("automl_best_model*"):
                try:
                    os.remove(f)
                except Exception:
                    pass

            clf.save_model(best_cal, "automl_best_model")
            model = clf.load_model("automl_best_model")

            import joblib
            joblib.dump(clf.get_config("X_train").columns.tolist(), "automl_model_columns.pkl")

            st.session_state.model = model
            st.session_state.fitted = True

            st.sidebar.success("✅ AutoML training complete — calibrated with sigmoid scaling.")
            if drop_cols:
                st.sidebar.info(f"🧹 Dropped ID column(s): {', '.join(drop_cols)}")

        except Exception as e:
            st.error(f"AutoML failed: {e}")
            st.stop()


# ----------------------------- SHOW LEADERBOARD (IF ANY) + KPI METRICS -----------------------------
overall_churn_rate, best_model_acc, num_at_risk = None, None, None
leaderboard_df = None
if st.session_state.fitted:
    try:
        results_df = clf.pull()
        leaderboard_df = results_df
        # Only compute metrics, do not render leaderboard/KPI here (render in tab1)
        try:
            if "Accuracy" in results_df.columns and "AUC" in results_df.columns:
                best_row = results_df.iloc[0]
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
            preds_df = clf.predict_model(st.session_state.model, data=st.session_state.train_df.sample(frac=0.2, random_state=123), raw_score=True)
            score_col = next(
                (c for c in preds_df.columns if c.lower().endswith("1")
                 or "score" in c.lower()
                 or "prob" in c.lower()),
                None,
            )
            if score_col:
                at_risk = preds_df[score_col] > 0.5
                num_at_risk = int(at_risk.sum())

            # Calculate predicted churn share from model output
            predicted_churn_rate = None
            try:
                preds_df_full = clf.predict_model(st.session_state.model, data=align_to_model_columns(df.copy(), st.session_state.model), raw_score=True)
                score_col2 = next((c for c in preds_df_full.columns if c.lower().endswith("1") or "score" in c.lower() or "prob" in c.lower()), None)
                if score_col2:
                    predicted_churn_rate = (preds_df_full[score_col2] > threshold).mean()
            except Exception:
                predicted_churn_rate = None
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
    row_for_pred = customer_row.drop(columns=[c for c in [id_col, target_col] if c], errors="ignore")
    prediction_proba = predict_row_prob(st.session_state.model, row_for_pred)
    adaptive_thr = st.session_state.get('adaptive_threshold', threshold)
    prediction_label = "Likely to Churn" if prediction_proba > adaptive_thr else "Likely to Stay"
    confidence = (
        "High" if abs(prediction_proba - adaptive_thr) > 0.25
        else "Medium" if abs(prediction_proba - adaptive_thr) > 0.10
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
top_features, top_values = [], []

if st.session_state.fitted and st.session_state.model is not None and SHAP_AVAILABLE:
    try:
        # Use pipeline and align columns for SHAP background and row
        bg = df.drop(columns=[c for c in [id_col, target_col] if c], errors="ignore").copy()
        bg_aligned = align_to_model_columns(bg, st.session_state.model)
        try:
            prep_pipe = clf.get_config('prep_pipe')
        except Exception:
            prep_pipe = None
        if prep_pipe is not None:
            bg_sample = prep_pipe.transform(bg_aligned.sample(min(100, len(bg_aligned)), random_state=42))
            row_for_expl = customer_row.drop(columns=[c for c in [id_col, target_col] if c], errors="ignore")
            row_aligned = align_to_model_columns(row_for_expl, st.session_state.model)
            row_transformed = prep_pipe.transform(row_aligned)
        else:
            bg_sample = bg_aligned.sample(min(100, len(bg_aligned)), random_state=42)
            row_for_expl = customer_row.drop(columns=[c for c in [id_col, target_col] if c], errors="ignore")
            row_transformed = align_to_model_columns(row_for_expl, st.session_state.model)

        explainer = shap.Explainer(st.session_state.model.predict, bg_sample)
        shap_values = explainer(row_transformed)

        # capture top features
        order = np.argsort(np.abs(shap_values[0].values))[::-1][:8]
        top_features = [shap_values[0].feature_names[i] for i in order]
        top_values = [shap_values[0].data[i] for i in order]
    except Exception as e:
        st.info(f"ℹ️ SHAP explanation unavailable: {e}")

# ----------------------------- NAVIGATION TABS -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Model Overview", "🔍 Single Prediction", "💡 Insights"])

# --------- TAB 1: MODEL OVERVIEW ---------
with tab1:
    st.markdown('<div class="section-padding"></div>', unsafe_allow_html=True)
    st.header("📈 Model Overview")
    st.write("Review dataset, AutoML leaderboard, and overall churn statistics.")
    with st.expander("📋 Dataset Preview", expanded=True):
        st.dataframe(df.head(), use_container_width=True, height=260)
    if leaderboard_df is not None:
        st.markdown(
            """
            <div class="kpi-row">
              <div class="kpi-metric">
                <div class="kpi-label">Overall Churn Rate (Dataset)</div>
                <div class="kpi-value" style="color:#e63946;">{churn:.1f}%</div>
              </div>
              <div class="kpi-metric">
                <div class="kpi-label">Predicted Churn Rate (Model)</div>
                <div class="kpi-value" style="color:#ff006e;">{predicted:.1f}%</div>
              </div>
              <div class="kpi-metric">
                <div class="kpi-label">Best Model Accuracy</div>
                <div class="kpi-value" style="color:#0077b6;">{acc:.1f}%</div>
              </div>
              <div class="kpi-metric">
                <div class="kpi-label">At-Risk Customers</div>
                <div class="kpi-value" style="color:#ffb703;">{risk}</div>
              </div>
            </div>
            """.format(
                churn=overall_churn_rate*100 if overall_churn_rate is not None else 0,
                predicted=predicted_churn_rate*100 if 'predicted_churn_rate' in locals() and predicted_churn_rate is not None else 0,
                acc=best_model_acc*100 if best_model_acc is not None else 0,
                risk=num_at_risk if num_at_risk is not None else "—"
            ),
            unsafe_allow_html=True
        )
    # Put leaderboard in expander again for tab1
    if leaderboard_df is not None:
        with st.expander("📊 Show AutoML Leaderboard (click to expand)", expanded=False):
            model_name = st.session_state.get("best_model_name", "Unknown Model")
            st.markdown(f"**🧩 Selected Model:** `{model_name}`")
            st.dataframe(leaderboard_df, use_container_width=True)

# --------- TAB 2: SINGLE PREDICTION ---------
with tab2:
    st.markdown('<div class="section-padding"></div>', unsafe_allow_html=True)
    st.header("🔍 Single Prediction")
    colA, colB, colC = st.columns(3)
    # 1. Prediction Card
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Result")
        if prediction_proba is None:
            st.info("Run AutoML to see predictions.")
        else:
            color_class = "metric-red" if prediction_label == "Likely to Churn" else "metric-green"
            color_bar = "#e63946" if prediction_label == "Likely to Churn" else "#2a9d8f"
            bar_val = int(prediction_proba*100)
            # Progress bar with HTML
            st.markdown(
                f"""
                <div style="font-size:1.2rem;font-weight:700;margin-bottom:7px;color:{color_bar};">
                  {prediction_label}
                </div>
                <progress value="{bar_val}" max="100" style="width:100%;height:22px;background:#f1f3f5;border-radius:8px;">
                  {bar_val}%
                </progress>
                <div style="font-size:1.1rem;margin-top:8px;">
                  <span style="color:#adb5bd;">Churn Probability:</span> <b style="color:{color_bar};">{prediction_proba:.2%}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"<span style='color:#adb5bd;'>Confidence:</span> <b>{confidence}</b>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Customer Profile Card
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📄 Customer Profile")
        st.dataframe(customer_row.T, use_container_width=True, height=250)
        st.markdown('---')
        st.subheader("💡 Rule-Based Suggestion")
        if OPENAI_AVAILABLE and prediction_proba is not None:
            with st.spinner("🤖 AI analyzing profile..."):
                profile_summary = customer_row.drop(columns=[target_col], errors='ignore').to_dict(orient='records')[0]
                st.success("AI Rule-Based Suggestion")
                st.markdown(gpt_rule_suggestion(profile_summary))
        else:
            st.markdown(
                '<div class="gray-info">OpenAI not configured or model not trained yet.</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. SHAP Card
    with colC:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="font-size:1.25rem;font-weight:700;">🧠 SHAP Explanation</span>
              <span title="SHAP (SHapley Additive Explanations) explains which features most influenced the churn prediction for this customer. Bar length shows impact; color shows direction (red=increases churn, green=retains)."
                    style="color:#adb5bd;cursor:help;font-size:1.3rem;">&#9432;</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if shap_values is not None:
            try:
                # Bar plot instead of waterfall
                shap_bar_vals = shap_values[0].values
                shap_bar_names = shap_values[0].feature_names
                order = np.argsort(np.abs(shap_bar_vals))[::-1][:8]
                fig, ax = plt.subplots(figsize=(6, 3.2))
                bar_colors = ['#e63946' if v > 0 else '#2a9d8f' for v in shap_bar_vals[order]]
                ax.barh(
                    [shap_bar_names[i] for i in order][::-1],
                    shap_bar_vals[order][::-1],
                    color=bar_colors[::-1],
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
                    "🔎 <b>Customer retention drivers:</b> Bars show which features most increased/decreased this customer's churn risk. "
                    "<span style='color:#e63946;'>Red</span> = pushes toward churn, <span style='color:#2a9d8f;'>green</span> = retention.",
                    unsafe_allow_html=True
                )
            except Exception:
                st.info("SHAP local plot unavailable.")
        else:
            st.info("No SHAP values computed.")
        st.markdown("</div>", unsafe_allow_html=True)

# --------- TAB 3: AI INSIGHTS ---------
with tab3:
    st.markdown('<div class="section-padding"></div>', unsafe_allow_html=True)
    st.header("💡 Insights & AI Actions")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 AI Action Recommendation")
    if OPENAI_AVAILABLE:
        st.success("OpenAI API: Connected")
        if prediction_proba is not None:
            if st.button("Generate Recommendation", key="ai_action_btn"):
                with st.spinner("Generating AI insights..."):
                    profile_summary = customer_row.drop(columns=[target_col], errors='ignore').to_dict(orient='records')[0]
                    st.success("AI Recommendation:")
                    st.markdown(
                        gpt_action_recommendation(
                            prediction_proba,
                            top_features or [],
                            top_values or [],
                            profile_summary
                        )
                    )
    else:
        st.markdown(
            '<div class="gray-info">OpenAI is not configured. Connect your OpenAI API key in the environment to enable AI recommendations and retention actions.</div>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- FOOTER -----------------------------
st.markdown(
    """
    <div class="footer">
      &copy; {year} Rasel Mia &mdash; Powered by <span style="color:#0077b6;">Streamlit</span>, <span style="color:#ffb703;">PyCaret</span>, <span style="color:#e63946;">SHAP</span>, <span style="color:#2a9d8f;">OpenAI</span>
    </div>
    """.format(year=pd.Timestamp.today().year),
    unsafe_allow_html=True
)