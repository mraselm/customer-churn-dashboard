"""
Deep diagnostic: Trace the EXACT data flow from CSV to model training.
Checks if RFM features leak into training despite the fix.
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

# 1. Load dataset (same as app.py)
df = pd.read_csv('processed_churn_dataset.csv')
target_col = 'Churn'
print(f"[1] Original CSV: {df.shape[1]} cols, {df.shape[0]} rows")
print(f"    Columns: {list(df.columns)}")

# 2. Simulate modeling_df creation (matches app.py ~line 2883)
modeling_df = df.copy()
print(f"\n[2] modeling_df before RFM: {modeling_df.shape[1]} cols")
print(f"    RFM-prefixed cols: {[c for c in modeling_df.columns if 'RFM' in c]}")

# 3. Apply RFM analysis (same as app.py ~line 3072-3075)
from universal_rfm import UniversalRFMAnalyzer
rfm_analyzer = UniversalRFMAnalyzer(verbose=True)
modeling_df = rfm_analyzer.analyze_and_engineer(modeling_df, target_col=target_col)
rfm_features = rfm_analyzer.rfm_features_created

print(f"\n[3] modeling_df AFTER RFM: {modeling_df.shape[1]} cols")
print(f"    RFM features created: {rfm_features}")
print(f"    All RFM-prefixed cols: {[c for c in modeling_df.columns if 'RFM' in c]}")

# 4. Apply the FIX (same as app.py ~line 3121-3123)
rfm_cols_to_drop = [c for c in modeling_df.columns if 'RFM' in c]
print(f"\n[4] RFM cols to drop: {rfm_cols_to_drop}")
if rfm_cols_to_drop:
    modeling_df = modeling_df.drop(columns=rfm_cols_to_drop)
print(f"    modeling_df AFTER drop: {modeling_df.shape[1]} cols")
print(f"    Remaining RFM cols: {[c for c in modeling_df.columns if 'RFM' in c]}")
print(f"    Final columns: {list(modeling_df.columns)}")

# 5. Simulate train/test split (same as app.py ~line 3246)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=123)
train_idx, test_idx = next(sss.split(modeling_df, modeling_df[target_col]))
train_df = modeling_df.iloc[train_idx].reset_index(drop=True)
test_df = modeling_df.iloc[test_idx].reset_index(drop=True)
print(f"\n[5] train_df: {train_df.shape}, test_df: {test_df.shape}")
print(f"    train_df RFM cols: {[c for c in train_df.columns if 'RFM' in c]}")
print(f"    train_df columns: {list(train_df.columns)}")

# 6. Quick model test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

feat_cols = [c for c in train_df.columns if c != target_col]
# One-hot encode object columns
obj_cols = [c for c in feat_cols if train_df[c].dtype == 'object']
if obj_cols:
    train_enc = pd.get_dummies(train_df, columns=obj_cols, drop_first=True)
    test_enc = pd.get_dummies(test_df, columns=obj_cols, drop_first=True)
    # Align
    for c in train_enc.columns:
        if c not in test_enc.columns:
            test_enc[c] = 0
    test_enc = test_enc[[c for c in train_enc.columns if c in test_enc.columns]]
else:
    train_enc = train_df.copy()
    test_enc = test_df.copy()

feat_cols_enc = [c for c in train_enc.columns if c != target_col]
print(f"\n[6] Training features ({len(feat_cols_enc)}): {feat_cols_enc}")

X_train = train_enc[feat_cols_enc].apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = train_enc[target_col].astype(int)
X_test = test_enc[feat_cols_enc].apply(pd.to_numeric, errors='coerce').fillna(0)
y_test = test_enc[target_col].astype(int)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print(f"\n[7] Model Performance:")
print(f"    AUC = {auc:.4f}")
print(f"    Churn proba distribution: min={y_proba.min():.4f}, max={y_proba.max():.4f}, mean={y_proba.mean():.4f}")
print(f"    Predicted churn (>0.5): {(y_proba > 0.5).sum()}/{len(y_proba)}")
print(f"    Predicted churn (>0.3): {(y_proba > 0.3).sum()}/{len(y_proba)}")

# Feature importance
imp = pd.Series(rf.feature_importances_, index=feat_cols_enc).sort_values(ascending=False)
print(f"\n[8] Top 10 Feature Importances:")
for feat, val in imp.head(10).items():
    print(f"    {feat}: {val:.4f}")

# 9. Check for leakage-like features
print(f"\n[9] Correlation with target:")
for c in feat_cols_enc:
    corr = X_train[c].corr(y_train.astype(float))
    if abs(corr) > 0.3:
        print(f"    HIGH CORR: {c} = {corr:.4f}")
