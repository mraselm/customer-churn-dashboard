"""
Test FIX 1: SHAP personalisation in generate_counterfactuals.
Runs CF on 5 randomly-selected customers and checks that at least 3
of the 5 get different top-recommended features.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"

import pandas as pd, numpy as np

# ---------- load data ----------
df = pd.read_csv("processed_churn_dataset.csv")
target_col = "Churn"

# ---------- train a quick model ----------
from pycaret.classification import ClassificationExperiment
clf = ClassificationExperiment()
df_encoded = pd.get_dummies(df, columns=[c for c in df.columns if df[c].dtype == 'object'], drop_first=True)
clf.setup(data=df_encoded, target=target_col, session_id=42, verbose=False)
model = clf.create_model('lr', verbose=False)
model = clf.finalize_model(model)

# save the feature columns for the app helper
import streamlit as st
st.session_state = {}
st.session_state["fitted"] = True
st.session_state["model"] = model
st.session_state["model_feature_cols"] = list(
    df_encoded.drop(columns=[target_col]).columns
)
st.session_state["drop_id_cols"] = []

# ---------- import the function under test ----------
# Patch st.sidebar / st.progress so headless works
class _Noop:
    def __getattr__(self, _): return lambda *a, **kw: None
st.sidebar = _Noop()
st.progress = lambda *a, **kw: _Noop()
st.spinner = lambda *a, **kw: type("_ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s,*a: None})()
st.info = lambda *a, **kw: None
st.warning = lambda *a, **kw: None

# We need the function from app.py – exec the function definition
import importlib, types
# Read the helpers we need
exec_ns = {"pd": pd, "np": np, "st": st}
with open("app.py", "r", encoding="utf-8") as f:
    src = f.read()

# Extract align_to_model_columns and generate_counterfactuals
import re, textwrap

def extract_function(source, func_name):
    """Extract a top-level function from source."""
    pattern = rf'^(def {func_name}\(.+?)(?=\ndef [a-zA-Z_]|\Z)'
    m = re.search(pattern, source, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1)
    return None

for fn_name in ["align_to_model_columns", "generate_counterfactuals"]:
    fn_src = extract_function(src, fn_name)
    if fn_src:
        exec(fn_src, exec_ns)
    else:
        print(f"ERROR: could not extract {fn_name}")
        sys.exit(1)

generate_counterfactuals = exec_ns["generate_counterfactuals"]

# ---------- pick 5 random high-risk customers ----------
np.random.seed(123)
# Score all customers first to find high-risk ones
from pycaret.classification import predict_model
preds = clf.predict_model(model, data=df_encoded, raw_score=True)
score_col = [c for c in preds.columns if 'score' in c.lower() and '1' in c]
if not score_col:
    score_col = [c for c in preds.columns if 'score' in c.lower()]
score_col = score_col[0] if score_col else None

if score_col:
    high_risk = preds[preds[score_col] > 0.4].index.tolist()
else:
    high_risk = list(range(len(df)))

sample_indices = np.random.choice(high_risk, size=min(5, len(high_risk)), replace=False)

print(f"\nTesting {len(sample_indices)} customers...\n")
print(f"{'Customer':>10}  {'TopFeature':<35}  {'ChurnProb':>10}  {'NewProb':>10}")
print("-" * 75)

top_features = []
full_details = []
for idx in sample_indices:
    cust = df.iloc[[idx]]
    try:
        cfs = generate_counterfactuals(
            customer_data=cust,
            model=model,
            data_df=df,
            target_col=target_col,
            num_cfs=1
        )
        if cfs and len(cfs) > 0:
            top_feat = list(cfs[0]['changes'].keys())[0]
            new_prob = cfs[0]['predicted_churn_prob']
            churn_prob = "?"
            top_features.append(top_feat)
            full_details.append((idx, top_feat, cfs[0]))
            print(f"{idx:>10}  {top_feat:<35}  {'?':>10}  {new_prob:>10.4f}")
        else:
            top_features.append("NONE")
            print(f"{idx:>10}  {'(no recommendations)':<35}")
    except Exception as e:
        top_features.append("ERROR")
        print(f"{idx:>10}  ERROR: {str(e)[:50]}")

unique_features = set(f for f in top_features if f not in ("NONE", "ERROR"))
print(f"\n{'='*75}")
print(f"Unique top features across 5 customers: {len(unique_features)}")
print(f"Features seen: {unique_features}")

if len(unique_features) >= 3:
    print("\n✅ PASS: At least 3 different top features — personalisation works!")
else:
    print(f"\n❌ NEEDS REVIEW: Only {len(unique_features)} unique feature(s).")
    # Print full importance table for first customer
    if full_details:
        idx0, _, cf0 = full_details[0]
        print(f"\nFull SHAP detail for customer {idx0}: see counterfactual output")
        print(cf0)
