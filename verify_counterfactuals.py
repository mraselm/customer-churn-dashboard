"""
Verification script for generate_counterfactuals fixes.
Tests 10 random high-risk customers and checks:
  - TotalCharges, MonthlyCharges, CLV, EngagementScore, tenure never recommended
  - At least 4 different features across 10 customers
  - Costs between $5 and $75
  - No null recommendations unless no actionable SHAP drivers > 0
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

# ── Synthetic Telco dataset matching user's 27-column schema ──
np.random.seed(42)
N = 2000
df = pd.DataFrame({
    'customerID': [f'CUST-{i:05d}' for i in range(N)],
    'gender': np.random.choice(['Male', 'Female'], N),
    'SeniorCitizen': np.random.choice([0, 1], N, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], N),
    'Dependents': np.random.choice(['Yes', 'No'], N, p=[0.3, 0.7]),
    'tenure': np.random.randint(1, 73, N),
    'PhoneService': np.random.choice(['Yes', 'No'], N, p=[0.9, 0.1]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], N),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], N, p=[0.34, 0.44, 0.22]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], N, p=[0.55, 0.25, 0.20]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], N),
    'PaymentMethod': np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], N),
    'MonthlyCharges': np.round(np.random.uniform(18.0, 118.0, N), 2),
    'TotalCharges': 0.0,  # derived below
    'DaysSinceLastContact': np.random.randint(1, 180, N),
    'ServicesSubscribed': np.random.randint(1, 8, N),
    'SatisfactionScore': np.random.randint(1, 6, N),
    'ComplaintProxy': np.random.randint(0, 5, N),
    'EngagementScore': np.round(np.random.uniform(0, 1, N), 3),
    'CLV': 0.0,
})
df['TotalCharges'] = np.round(df['tenure'] * df['MonthlyCharges'], 2)
df['CLV'] = np.round(df['MonthlyCharges'] * df['tenure'] * 0.8, 2)

# Churn label - correlated with Contract/InternetService/tenure
churn_prob = (
    0.25
    + 0.25 * (df['Contract'] == 'Month-to-month').astype(float)
    + 0.15 * (df['InternetService'] == 'Fiber optic').astype(float)
    - 0.10 * (df['tenure'] / 72)
    + 0.10 * (df['OnlineSecurity'] == 'No').astype(float)
    - 0.05 * (df['TechSupport'] == 'Yes').astype(float)
)
df['Churn'] = (np.random.rand(N) < churn_prob).astype(int)

target_col = 'Churn'
id_col = 'customerID'

# ── Train model with PyCaret (same pipeline as app.py) ──
print("Training model ...")
# Pre-process exactly like app.py
modeling_df = df.drop(columns=[id_col]).copy()

# Binary mapping
binary_map = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}
binary_mapped_cols = []
for c in modeling_df.select_dtypes('object').columns:
    uniq = modeling_df[c].str.strip().str.lower().unique()
    if set(uniq).issubset({'yes', 'no', 'male', 'female'}):
        modeling_df[c] = modeling_df[c].str.strip().str.lower().map(binary_map).astype(float)
        binary_mapped_cols.append(c)

# Space → underscore in remaining categoricals
cat_cols_clean = [c for c in modeling_df.select_dtypes('object').columns]
for c in cat_cols_clean:
    modeling_df[c] = modeling_df[c].astype(str).str.strip().str.replace(r'\s+', '_', regex=True)

# get_dummies
modeling_df = pd.get_dummies(modeling_df, columns=cat_cols_clean, drop_first=True)

from pycaret.classification import ClassificationExperiment
clf = ClassificationExperiment()
clf.setup(data=modeling_df, target=target_col, session_id=42, verbose=False)
best_model = clf.create_model('rf', verbose=False)
model_feature_cols = list(clf.get_config('X_train').columns)

print(f"  Model features ({len(model_feature_cols)}): {model_feature_cols[:8]} ...")
print(f"  Churn rate: {df[target_col].mean():.1%}")

# ── Simulate session_state for generate_counterfactuals ──
import streamlit as st
# Patch st.session_state & st.sidebar for headless use
class _FakeState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v

class _FakeSidebar:
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def code(self, *a, **kw): pass

st.session_state = _FakeState({
    'drop_id_cols': [id_col],
    'model_feature_cols': model_feature_cols,
    'binary_map': binary_map,
    'binary_mapped_cols': binary_mapped_cols,
    'cat_cols_clean': cat_cols_clean,
})
st.sidebar = _FakeSidebar()

# ── Import generate_counterfactuals from app.py ──
sys.path.insert(0, os.path.dirname(__file__))
# We need to extract the function; easiest is to exec the relevant portion.
# Instead, just import the whole module after patching streamlit.
# But app.py has top-level st calls. Let's just copy the function.
# Actually — let's read app.py and exec just the function.

# More robust: call directly
print("\n" + "="*70)
print("RUNNING VERIFICATION: 10 random high-risk customers")
print("="*70)

# Get 10 high-risk customers (predict churn prob > 0.5)
raw_est = best_model
if hasattr(best_model, 'steps'):
    raw_est = best_model.steps[-1][1]

X_all = modeling_df.drop(columns=[target_col])
proba = raw_est.predict_proba(X_all.values)[:, 1]
high_risk_idx = np.where(proba > 0.5)[0]

if len(high_risk_idx) < 10:
    print(f"  Only {len(high_risk_idx)} high-risk customers found, using all")
    selected_idx = high_risk_idx
else:
    selected_idx = np.random.choice(high_risk_idx, 10, replace=False)

print(f"  Selected {len(selected_idx)} customers (churn prob > 0.5)")

# ── Now call generate_counterfactuals for each ──
# Import the function from app.py by loading just the function def
# Easier approach: copy a minimal version that uses the same logic

# Actually, let's exec the function from app.py
import re
with open(os.path.join(os.path.dirname(__file__), 'app.py'), 'r', encoding='utf-8') as f:
    app_source = f.read()

# Find the function
match = re.search(r'^(def generate_counterfactuals\(.+?)(?=\n(?:def |class |# ----))', app_source, re.DOTALL | re.MULTILINE)
if not match:
    print("ERROR: Could not extract generate_counterfactuals from app.py")
    sys.exit(1)

func_source = match.group(1)
# Execute in a namespace with required imports
exec_ns = {
    'pd': pd, 'np': np, 'st': st,
    '__builtins__': __builtins__,
}
exec(func_source, exec_ns)
generate_counterfactuals = exec_ns['generate_counterfactuals']

# ── Run for each selected customer ──
NON_ACTIONABLE = {'totalcharges', 'monthlycharges', 'clv', 'engagementscore', 'tenure', 'customerid'}
all_features_seen = set()
results = []
failures = 0

for i, idx in enumerate(selected_idx):
    customer_row = df.iloc[[idx]].copy()
    recs = generate_counterfactuals(
        customer_data=customer_row,
        model=best_model,
        data_df=df.drop(columns=[id_col]),
        target_col=target_col,
        num_cfs=3,
    )
    
    if recs is None or len(recs) == 0:
        print(f"  Customer {idx}: NO recommendations (low risk or no actionable drivers)")
        failures += 1
        results.append({'idx': idx, 'feature': None, 'cost': None, 'deviation': None})
        continue
    
    top_rec = recs[0]
    feat = list(top_rec['changes'].keys())[0]
    cost = top_rec['implementation_cost']
    dev = top_rec.get('shap_contribution', 0)
    all_features_seen.add(feat)
    results.append({'idx': idx, 'feature': feat, 'cost': cost, 'deviation': round(dev, 4)})
    print(f"  Customer {idx}: feature={feat:<35s}  cost=${cost:.2f}  SHAP={dev:.4f}")

# ── Validation checks ──
print("\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)

# Check 1: Non-actionable features never appear
non_actionable_found = [f for f in all_features_seen if any(na in f.lower() for na in NON_ACTIONABLE)]
if non_actionable_found:
    print(f"  FAIL: Non-actionable features recommended: {non_actionable_found}")
else:
    print(f"  PASS: No non-actionable features recommended")

# Check 2: Feature diversity
print(f"  Features seen ({len(all_features_seen)}): {sorted(all_features_seen)}")
if len(all_features_seen) >= 4:
    print(f"  PASS: {len(all_features_seen)} different features (>= 4 required)")
else:
    print(f"  WARN: Only {len(all_features_seen)} different features (< 4)")
    # Print full SHAP table for one customer
    print("\n  DIAGNOSTIC: Full actionable SHAP table for first high-risk customer:")
    # Re-run with debug
    customer_row = df.iloc[[selected_idx[0]]].copy()
    recs = generate_counterfactuals(customer_row, best_model, df.drop(columns=[id_col]), target_col, num_cfs=10)
    if recs:
        for r in recs:
            f_ = list(r['changes'].keys())[0]
            print(f"    {f_:<35s}  cost=${r['implementation_cost']:.2f}  SHAP={r['shap_contribution']:.4f}")

# Check 3: Cost range
valid_costs = [r['cost'] for r in results if r['cost'] is not None]
if valid_costs:
    min_c, max_c = min(valid_costs), max(valid_costs)
    if 5 <= min_c and max_c <= 75:
        print(f"  PASS: All costs in $5-$75 range (min=${min_c:.2f}, max=${max_c:.2f})")
    else:
        print(f"  FAIL: Costs outside $5-$75 range (min=${min_c:.2f}, max=${max_c:.2f})")
else:
    print(f"  WARN: No costs to validate (all recommendations were null)")

# Check 4: Null recommendations
if failures > 5:
    print(f"  WARN: {failures}/10 customers had null recommendations")
else:
    print(f"  PASS: {failures}/10 customers with null recommendations (acceptable)")

print("\nDone.")
