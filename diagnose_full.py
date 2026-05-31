"""
Full end-to-end test: Simulate app.py's EXACT pipeline with PyCaret.
This tests everything including RFM exclusion, pre-encoding, and PyCaret setup.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, '.')
os.chdir(r'e:\GitHub Repo\customer-churn-dashboard')

# ========================== STEP 1: Load Dataset ==========================
df = pd.read_csv('processed_churn_dataset.csv')
target_col = 'Churn'
print(f"[STEP 1] Dataset loaded: {df.shape}")

# ========================== STEP 2: Create modeling_df ==========================
modeling_df = df.copy()
print(f"[STEP 2] modeling_df: {modeling_df.shape}")

# ========================== STEP 3: Infer feature types ==========================
inferred_num = []
inferred_cat = []
for c in modeling_df.columns:
    if c == target_col:
        continue
    if modeling_df[c].dtype in ['int64', 'float64']:
        inferred_num.append(c)
    else:
        inferred_cat.append(c)
print(f"[STEP 3] Numeric: {len(inferred_num)}, Categorical: {len(inferred_cat)}")
print(f"  Categorical cols: {inferred_cat}")

# ========================== STEP 4: RFM Analysis ==========================
from universal_rfm import UniversalRFMAnalyzer
rfm_analyzer = UniversalRFMAnalyzer(verbose=False)
modeling_df = rfm_analyzer.analyze_and_engineer(modeling_df, target_col=target_col)
rfm_features = rfm_analyzer.rfm_features_created
print(f"\n[STEP 4] RFM features created: {len(rfm_features)}")
print(f"  Features: {rfm_features}")

# ========================== STEP 5: Drop RFM (THE FIX) ==========================
rfm_cols_to_drop = [c for c in modeling_df.columns if 'RFM' in c]
print(f"\n[STEP 5] Dropping {len(rfm_cols_to_drop)} RFM columns from modeling_df")
modeling_df = modeling_df.drop(columns=rfm_cols_to_drop)
inferred_num = [c for c in inferred_num if 'RFM' not in c]
inferred_cat = [c for c in inferred_cat if 'RFM' not in c]
print(f"  modeling_df now: {modeling_df.shape}")
print(f"  Any RFM remaining: {[c for c in modeling_df.columns if 'RFM' in c]}")

# ========================== STEP 6: Train/Test Split ==========================
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=123)
train_idx, test_idx = next(sss.split(modeling_df, modeling_df[target_col]))
train_df = modeling_df.iloc[train_idx].reset_index(drop=True)
test_df = modeling_df.iloc[test_idx].reset_index(drop=True)
print(f"\n[STEP 6] Split: train={train_df.shape}, test={test_df.shape}")

# ========================== STEP 7: Pre-encode object cols ==========================
obj_cols = [c for c in train_df.columns if c != target_col and train_df[c].dtype == 'object']
print(f"\n[STEP 7] Object cols to encode: {obj_cols}")
if obj_cols:
    train_df = pd.get_dummies(train_df, columns=obj_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=obj_cols, drop_first=True)
    for c in train_df.columns:
        if c not in test_df.columns:
            test_df[c] = 0
    test_df = test_df[[c for c in train_df.columns if c in test_df.columns]]
train_df[target_col] = train_df[target_col].astype(int)
test_df[target_col] = test_df[target_col].astype(int)
print(f"  After encoding: train={train_df.shape}, test={test_df.shape}")
print(f"  Any RFM in train: {[c for c in train_df.columns if 'RFM' in c]}")

# ========================== STEP 8: PyCaret Setup + Train ==========================
from pycaret import classification as clf

print(f"\n[STEP 8] Running PyCaret setup...")
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

# Check what PyCaret sees
X_train_pycaret = clf.get_config("X_train")
print(f"  PyCaret X_train shape: {X_train_pycaret.shape}")
print(f"  PyCaret X_train columns: {list(X_train_pycaret.columns)}")
rfm_in_pycaret = [c for c in X_train_pycaret.columns if 'RFM' in c]
print(f"  RFM in PyCaret: {rfm_in_pycaret}")

print(f"\n  Comparing models...")
best = clf.compare_models(
    include=['rf', 'xgboost', 'lightgbm'],
    sort='AUC',
    fold=5,
    n_select=1
)
lb = clf.pull()
print(f"\n  Leaderboard:")
print(lb.to_string())

# ========================== STEP 9: Calibrate ==========================
try:
    best_cal = clf.calibrate_model(best, method='isotonic')
    print(f"\n[STEP 9] Calibrated model: {type(best_cal).__name__}")
except Exception as e:
    print(f"\n[STEP 9] Calibration failed: {e}, using uncalibrated")
    best_cal = best

# ========================== STEP 10: Save/Load + Predict ==========================
import glob, joblib
for f in glob.glob("test_model*"):
    try: os.remove(f)
    except: pass

clf.save_model(best_cal, "test_model")
model = clf.load_model("test_model")

model_feature_cols = clf.get_config("X_train").columns.tolist()
print(f"\n[STEP 10] Model feature columns ({len(model_feature_cols)}):")
print(f"  {model_feature_cols}")
print(f"  RFM in model features: {[c for c in model_feature_cols if 'RFM' in c]}")

# ========================== STEP 11: Predict on full dataset ==========================
# Simulate align_to_model_columns
full_df = df.copy()
# Drop target
full_X = full_df.drop(columns=[target_col], errors='ignore')
# One-hot encode
obj_predict = [c for c in full_X.columns if full_X[c].dtype == 'object']
if obj_predict:
    full_X = pd.get_dummies(full_X, columns=obj_predict, drop_first=True)
# Align to model columns
for c in model_feature_cols:
    if c not in full_X.columns:
        full_X[c] = 0
full_X = full_X[model_feature_cols]
full_X = full_X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"\n[STEP 11] Predicting on full dataset ({full_X.shape})...")
try:
    proba = model.predict_proba(full_X)
    if hasattr(proba, 'iloc'):
        scores = proba.iloc[:, 1] if proba.shape[1] > 1 else proba.iloc[:, 0]
    else:
        scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    scores = np.array(scores, dtype=float)
    
    print(f"  Score distribution: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    print(f"  Churn (>0.5): {(scores > 0.5).sum()}/{len(scores)}")
    print(f"  Churn (>0.3): {(scores > 0.3).sum()}/{len(scores)}")
    print(f"  All zero: {(scores == 0).all()}")
    print(f"  Near-zero (<0.01): {(scores < 0.01).sum()}/{len(scores)}")
except Exception as e:
    print(f"  PREDICTION FAILED: {e}")
    import traceback
    traceback.print_exc()

# ========================== STEP 12: Top feature importances ==========================
print(f"\n[STEP 12] Feature importances (from PyCaret model):")
try:
    estimator = model
    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]
    if hasattr(estimator, 'calibrated_classifiers_'):
        # CalibratedClassifierCV - get base estimator
        estimator = estimator.calibrated_classifiers_[0].estimator
    if hasattr(estimator, 'feature_importances_'):
        imp = pd.Series(estimator.feature_importances_, index=model_feature_cols).sort_values(ascending=False)
        print(f"  Top 10:")
        for feat, val in imp.head(10).items():
            print(f"    {feat}: {val:.4f}")
        rfm_imp = [(f, v) for f, v in imp.items() if 'RFM' in f]
        if rfm_imp:
            print(f"  WARNING: RFM features in importance: {rfm_imp}")
        else:
            print(f"  OK: No RFM features in importance")
    else:
        print(f"  (no feature_importances_ attribute)")
except Exception as e:
    print(f"  Error getting importances: {e}")

# Cleanup
for f in glob.glob("test_model*"):
    try: os.remove(f)
    except: pass

print(f"\n{'='*60}")
print(f"DIAGNOSIS COMPLETE")
print(f"{'='*60}")
