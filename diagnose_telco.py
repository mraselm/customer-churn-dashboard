"""
Full diagnostic with the user's ACTUAL Telco dataset schema (27 columns).
Creates a synthetic dataset matching the exact column names and types,
then traces the entire app.py pipeline.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, '.')
os.chdir(r'e:\GitHub Repo\customer-churn-dashboard')

np.random.seed(42)
N = 2000

# Create synthetic Telco-style dataset matching user's 27 columns
df = pd.DataFrame({
    'customerID': [f'C{i:04d}' for i in range(N)],
    'gender': np.random.choice(['Male', 'Female'], N),
    'SeniorCitizen': np.random.choice([0, 1], N, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], N),
    'Dependents': np.random.choice(['Yes', 'No'], N, p=[0.3, 0.7]),
    'tenure': np.random.randint(0, 72, N),
    'PhoneService': np.random.choice(['Yes', 'No'], N, p=[0.9, 0.1]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], N),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], N),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], N),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], N),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], N),
    'PaymentMethod': np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], N),
    'MonthlyCharges': np.round(np.random.uniform(18, 118, N), 2),
    'TotalCharges': np.round(np.random.uniform(18, 8600, N), 2),
    'Churn': np.random.choice([0, 1], N, p=[0.73, 0.27]),
    'DaysSinceLastContact': np.random.randint(1, 365, N),
    'ServicesSubscribed': np.random.randint(1, 7, N),
    'SatisfactionScore': np.random.randint(1, 6, N),
    'ComplaintProxy': np.round(np.random.uniform(0, 1, N), 3),
    'EngagementScore': np.round(np.random.uniform(0, 100, N), 1),
    'CLV': np.round(np.random.uniform(100, 15000, N), 2),
})

print(f"[1] Synthetic Telco dataset: {df.shape}")
print(f"    Columns: {list(df.columns)}")
print(f"    Dtypes: object={sum(df.dtypes=='object')}, numeric={sum(df.dtypes!='object')}")

target_col = 'Churn'
id_col = 'customerID'

# ========== STEP 2: Simulate app.py modeling_df ==========
drop_cols = [id_col]
modeling_df = df.drop(columns=drop_cols, errors="ignore").copy()
print(f"\n[2] modeling_df after dropping ID: {modeling_df.shape}")

# ========== STEP 3: Infer feature types (like app.py) ==========
inferred_num = []
inferred_cat = []
for c in modeling_df.columns:
    if c == target_col:
        continue
    if modeling_df[c].dtype in ['int64', 'float64', 'int32', 'float32']:
        inferred_num.append(c)
    elif modeling_df[c].dtype == 'object':
        inferred_cat.append(c)
print(f"\n[3] Numeric: {len(inferred_num)} = {inferred_num}")
print(f"    Categorical: {len(inferred_cat)} = {inferred_cat}")

# ========== STEP 4: RFM Analysis ==========
from universal_rfm import UniversalRFMAnalyzer
rfm_analyzer = UniversalRFMAnalyzer(verbose=True)
modeling_df = rfm_analyzer.analyze_and_engineer(modeling_df, target_col=target_col)
rfm_features = rfm_analyzer.rfm_features_created

print(f"\n[4] RFM features: {rfm_features}")
rfm_cols_in_df = [c for c in modeling_df.columns if 'RFM' in c]
print(f"    RFM cols in modeling_df: {rfm_cols_in_df}")

# ========== STEP 5: Drop RFM (THE FIX) ==========
rfm_cols_to_drop = [c for c in modeling_df.columns if 'RFM' in c]
if rfm_cols_to_drop:
    modeling_df = modeling_df.drop(columns=rfm_cols_to_drop)
    inferred_num = [c for c in inferred_num if 'RFM' not in c]
    inferred_cat = [c for c in inferred_cat if 'RFM' not in c]
    print(f"\n[5] Dropped {len(rfm_cols_to_drop)} RFM cols. modeling_df: {modeling_df.shape}")
else:
    print(f"\n[5] No RFM cols to drop")
print(f"    Remaining cols: {list(modeling_df.columns)}")

# ========== STEP 6: Binary mapping (like app.py) ==========
binary_mapped_cols = []
for c in list(inferred_cat):
    lower_vals = modeling_df[c].astype(str).str.strip().str.lower()
    unique_lower = set(lower_vals.unique())
    if unique_lower <= {'yes', 'no'}:
        binary_map = {'yes': 1, 'no': 0}
        modeling_df[c] = lower_vals.map(binary_map).astype(int)
        binary_mapped_cols.append(c)
        inferred_cat.remove(c)
        inferred_num.append(c)
print(f"\n[6] Binary mapped: {binary_mapped_cols}")
print(f"    Remaining categorical: {inferred_cat}")

# ========== STEP 7: Train/Test Split ==========
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=123)
train_idx, test_idx = next(sss.split(modeling_df, modeling_df[target_col]))
train_df = modeling_df.iloc[train_idx].reset_index(drop=True)
test_df = modeling_df.iloc[test_idx].reset_index(drop=True)
print(f"\n[7] Split: train={train_df.shape}, test={test_df.shape}")

# ========== STEP 8: One-hot encode objects ==========
obj_cols = [c for c in train_df.columns if c != target_col and train_df[c].dtype == 'object']
print(f"\n[8] Object cols to encode: {obj_cols}")
if obj_cols:
    train_df = pd.get_dummies(train_df, columns=obj_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=obj_cols, drop_first=True)
    for c in train_df.columns:
        if c not in test_df.columns:
            test_df[c] = 0
    test_df = test_df[[c for c in train_df.columns if c in test_df.columns]]
train_df[target_col] = train_df[target_col].astype(int)
test_df[target_col] = test_df[target_col].astype(int)
print(f"    After encoding: train={train_df.shape}")
print(f"    Columns: {list(train_df.columns)}")
print(f"    Any RFM: {[c for c in train_df.columns if 'RFM' in c]}")

# ========== STEP 9: PyCaret ==========
from pycaret import classification as clf

print(f"\n[9] PyCaret setup...")
_ = clf.setup(
    data=train_df,
    target=target_col,
    session_id=123,
    fold=5,
    fold_shuffle=True,
    fix_imbalance=True,
    use_gpu=False,
    verbose=False
)

X_train = clf.get_config("X_train")
print(f"    X_train: {X_train.shape}")
print(f"    RFM in X_train: {[c for c in X_train.columns if 'RFM' in c]}")

print(f"\n    Comparing models...")
best = clf.compare_models(
    include=['rf', 'xgboost', 'lightgbm'],
    sort='AUC',
    fold=5,
    n_select=1
)
lb = clf.pull()
print(f"\n    Leaderboard:")
print(lb.to_string())

# ========== STEP 10: Calibrate + Save ==========
try:
    best_cal = clf.calibrate_model(best, method='isotonic')
except:
    best_cal = best

import glob, joblib
for f in glob.glob("test_telco_model*"):
    try: os.remove(f)
    except: pass

clf.save_model(best_cal, "test_telco_model")
model = clf.load_model("test_telco_model")
model_feature_cols = clf.get_config("X_train").columns.tolist()
print(f"\n[10] Model features ({len(model_feature_cols)}): {model_feature_cols}")

# ========== STEP 11: Predict on full data ==========
full_X = df.drop(columns=[target_col, id_col], errors='ignore')
# Encode categoricals same way
obj_predict = [c for c in full_X.columns if full_X[c].dtype == 'object']
# Binary map first
for c in binary_mapped_cols:
    if c in full_X.columns:
        full_X[c] = full_X[c].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).astype(float)
obj_predict = [c for c in full_X.columns if full_X[c].dtype == 'object']
if obj_predict:
    full_X = pd.get_dummies(full_X, columns=obj_predict, drop_first=True)
for c in model_feature_cols:
    if c not in full_X.columns:
        full_X[c] = 0
full_X = full_X[model_feature_cols]
full_X = full_X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"\n[11] Predicting on full data ({full_X.shape})...")
proba = model.predict_proba(full_X)
if hasattr(proba, 'iloc'):
    scores = proba.iloc[:, 1].values
else:
    scores = proba[:, 1]
scores = np.array(scores, dtype=float)

print(f"    Score dist: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
print(f"    Churn (>0.5): {(scores > 0.5).sum()}/{len(scores)}")
print(f"    Churn (>0.3): {(scores > 0.3).sum()}/{len(scores)}")
print(f"    All near-zero (<0.01): {(scores < 0.01).sum()}/{len(scores)}")

# ========== STEP 12: Feature Importance ==========
estimator = model
if hasattr(model, 'steps'):
    estimator = model.steps[-1][1]
if hasattr(estimator, 'calibrated_classifiers_'):
    estimator = estimator.calibrated_classifiers_[0].estimator
if hasattr(estimator, 'feature_importances_'):
    imp = pd.Series(estimator.feature_importances_, index=model_feature_cols).sort_values(ascending=False)
    print(f"\n[12] Top 10 features:")
    for feat, val in imp.head(10).items():
        print(f"    {feat}: {val:.4f}")

# Cleanup
for f in glob.glob("test_telco_model*"):
    try: os.remove(f)
    except: pass

print(f"\n{'='*60}")
print("TELCO PIPELINE DIAGNOSTIC COMPLETE")
print(f"{'='*60}")
