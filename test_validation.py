"""
Test script to validate the professional validation engine
"""

import pandas as pd
from validation_engine import quick_validate

# Load dataset
print("="*80)
print("TESTING PROFESSIONAL VALIDATION ENGINE")
print("="*80)

df = pd.read_csv("processed_churn_dataset.csv")
print(f"\n✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"✅ Columns: {', '.join(df.columns.tolist())}")

# Run quick validation
print("\n" + "="*80)
print("RUNNING VALIDATION...")
print("="*80)

clean_df, leaked_features, engine = quick_validate(df, "Churn", leakage_threshold=0.90)

# Show results
print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)

if leaked_features:
    print(f"\n❌ Found {len(leaked_features)} leaked features:")
    for f in leaked_features:
        print(f"   • {f}")
    print(f"\n✅ Clean dataset: {clean_df.shape[0]} rows × {clean_df.shape[1]} columns")
    print(f"✅ Removed {len(leaked_features)} features")
else:
    print("\n✅ No leaked features detected")
    print(f"✅ Dataset is clean: {clean_df.shape[0]} rows × {clean_df.shape[1]} columns")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
