import pandas as pd
import numpy as np

df = pd.read_csv('processed_churn_dataset.csv')

print('='*80)
print('DATASET ANALYSIS')
print('='*80)
print(f'Total rows: {len(df)}')
print(f'\nChurn distribution:')
print(df['Churn'].value_counts())
print()
print('Feature correlations with Churn (top 10):')
corrs = {}
for col in df.columns:
    if col != 'Churn' and df[col].dtype.kind in 'bifc':
        try:
            corrs[col] = abs(df[col].corr(df['Churn']))
        except:
            pass

for col, corr in sorted(corrs.items(), key=lambda x: -x[1])[:10]:
    print(f'  {col:30s}: {corr:.4f}')

print()
print('Suspicious features (RFM and derived metrics):')
print(f'  Recency - unique values: {df["Recency"].nunique()}')
print(f'  Frequency - unique values: {df["Frequency"].nunique()}')
print(f'  Monetary - unique values: {df["Monetary"].nunique()}')
print(f'  EngagementScore - unique values: {df["EngagementScore"].nunique()}')
print(f'  ComplaintRatio - unique values: {df["ComplaintRatio"].nunique()}')
print(f'  Complain - unique values: {df["Complain"].nunique()}')
print('='*80)
