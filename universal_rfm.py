"""
Universal RFM (Recency, Frequency, Monetary) Analyzer
Automatically detects and creates RFM features from any dataset structure.
Works with any column names - no rigid requirements.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


class UniversalRFMAnalyzer:
    """
    Intelligent RFM feature engineering that works with ANY dataset.
    Auto-detects monetary, frequency, and recency indicators.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.rfm_features_created = []
        self.detection_log = []
        self.rfm_sources = {
            'monetary': [],
            'frequency': [],
            'recency': [],
        }
        
        # Universal keyword mappings for column detection
        self.monetary_keywords = [
            'monetary', 'revenue', 'amount', 'spending', 'spend', 'price',
            'sales', 'value', 'ltv', 'clv', 'lifetime', 'total', 'payment',
            'charge', 'cost', 'cashback', 'balance', 'income',
            'wallet', 'refund', 'coupon', 'hike',
        ]

        self.frequency_keywords = [
            'frequency', 'order', 'purchase', 'transaction', 'visit', 'count',
            'number', 'quantity', 'times', 'occurrences', 'sessions', 'trips',
            'bookings', 'renewals', 'interactions',
            'ordercount', 'numorder', 'num_order',
        ]

        self.recency_keywords = [
            'recency', 'tenure', 'age', 'duration', 'last', 'recent', 'days',
            'months', 'years', 'since', 'member', 'subscription', 'account',
            'dayslastorder', 'dayssincelast', 'daysincelast', 'vintage',
            'seniority', 'lastorder', 'lastpurchase', 'lasttransaction',
        ]

        # Keywords indicating "days since" semantics (lower = more recent = better)
        self._recency_days_since_keywords = [
            'since', 'last', 'ago', 'dayslastorder', 'dayssincelast',
            'daysincelast', 'lastorder', 'lastpurchase', 'lasttransaction',
            'recency',
        ]
        self._id_like_patterns = [
            r'(^|_)(id)($|_)',
            r'(^|_)(customer_id|customerid|cust_id)($|_)',
            r'(^|_)(account_id|accountid)($|_)',
            r'(^|_)(user_id|userid)($|_)',
            r'(^|_)(client_id|clientid)($|_)',
        ]
    
    def _log(self, message: str):
        """Log detection progress"""
        self.detection_log.append(message)
        if self.verbose:
            print(f"[RFM] {message}")

    def _is_id_like(self, col_name: str) -> bool:
        key = re.sub(r"\s+", "_", str(col_name).strip().lower())
        return any(re.search(p, key) for p in self._id_like_patterns)

    def _numeric_quality(self, series: pd.Series) -> tuple[pd.Series, float, int]:
        num = pd.to_numeric(series, errors='coerce')
        conv_ratio = float(num.notna().mean())
        unique_n = int(num.nunique(dropna=True))
        return num, conv_ratio, unique_n

    def _keyword_score(self, col_name: str, keywords: List[str]) -> int:
        name = str(col_name).lower()
        return sum(1 for kw in keywords if kw in name)

    def _is_demographic_age_col(self, col_name: str) -> bool:
        """
        Exclude demographic age fields from recency detection.
        Keep account-age style fields (e.g., account_age_months) as valid recency.
        """
        key = re.sub(r"\s+", "_", str(col_name).strip().lower())
        if "age" not in key:
            return False
        recency_context = ["account", "tenure", "member", "subscription", "vintage", "seniority"]
        if any(k in key for k in recency_context):
            return False
        return True

    def _pick_best_columns(
        self,
        df: pd.DataFrame,
        keywords: List[str],
        k_max: int = 1,
        dimension: Optional[str] = None,
    ) -> List[str]:
        """
        Select top candidate numeric columns for a dimension using:
        - keyword match strength
        - numeric conversion quality
        - non-trivial variance
        """
        candidates = []
        for col in df.columns:
            if self._is_id_like(col):
                continue
            if dimension == "recency" and self._is_demographic_age_col(col):
                continue
            kscore = self._keyword_score(col, keywords)
            if kscore <= 0:
                continue

            num, conv_ratio, unique_n = self._numeric_quality(df[col])
            if conv_ratio < 0.60:
                continue
            if unique_n <= 2:
                continue
            std = float(num.std(skipna=True)) if num.notna().any() else 0.0
            if std <= 1e-9:
                continue

            candidates.append((kscore, conv_ratio, std, col))

        if not candidates:
            return []

        # Highest keyword relevance first, then conversion quality, then variability.
        candidates = sorted(candidates, key=lambda x: (x[0], x[1], x[2]), reverse=True)
        picked = [c[-1] for c in candidates[:k_max]]
        return picked
    
    def _calculate_rfm_score(self, series: pd.Series, reverse: bool = False) -> pd.Series:
        """
        Calculate RFM score (1-5) using quintile ranking.
        
        Args:
            series: Column to score
            reverse: If True, lower values get higher scores (for recency)
        """
        try:
            # Handle missing values
            series_clean = series.fillna(series.median())
            
            # Use qcut for quintile-based scoring (1-5)
            try:
                scores = pd.qcut(
                    series_clean, 
                    q=5, 
                    labels=[1, 2, 3, 4, 5] if not reverse else [5, 4, 3, 2, 1],
                    duplicates='drop'
                )
            except ValueError:
                # If qcut fails (not enough unique values), use cut
                scores = pd.cut(
                    series_clean,
                    bins=5,
                    labels=[1, 2, 3, 4, 5] if not reverse else [5, 4, 3, 2, 1],
                    duplicates='drop'
                )
            
            # Convert to numeric
            scores = pd.to_numeric(scores, errors='coerce').fillna(3)  # Default to middle score
            return scores
            
        except Exception as e:
            self._log(f"Scoring failed: {e}, using raw normalized values")
            # Fallback: normalize to 1-5 range
            normalized = (series - series.min()) / (series.max() - series.min()) * 4 + 1
            if reverse:
                normalized = 6 - normalized
            return normalized.fillna(3)
    
    def analyze_and_engineer(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Main method: Analyze dataset and create RFM features.
        
        Args:
            df: Input dataframe
            target_col: Target column to exclude from RFM calculation
            
        Returns:
            Enhanced dataframe with RFM features
        """
        # Reset state on every run so repeated calls don't accumulate stale features/logs.
        self.rfm_features_created = []
        self.detection_log = []
        self.rfm_sources = {'monetary': [], 'frequency': [], 'recency': []}

        df_enhanced = df.copy()
        self._log("Starting Universal RFM Analysis...")
        
        # Exclude target column from analysis
        analysis_cols = [c for c in df.columns if c != target_col]
        df_analysis = df[analysis_cols]
        
        # ========== MONETARY DETECTION ==========
        monetary_cols = self._pick_best_columns(
            df_analysis,
            self.monetary_keywords,
            k_max=1,
            dimension="monetary",
        )
        if monetary_cols:
            self._log(f"Detected {len(monetary_cols)} monetary column(s): {', '.join(monetary_cols[:3])}")
            self.rfm_sources['monetary'] = monetary_cols.copy()
            
            # Aggregate if multiple monetary columns
            if len(monetary_cols) == 1:
                monetary_value = pd.to_numeric(df[monetary_cols[0]], errors='coerce')
            else:
                # Median-stable average across selected monetary columns
                monetary_value = df[monetary_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            
            # Create Monetary score (higher spend = higher score)
            df_enhanced['RFM_Monetary_Score'] = self._calculate_rfm_score(monetary_value, reverse=False)
            df_enhanced['RFM_Monetary_Value'] = monetary_value  # Raw value for reference
            self.rfm_features_created.extend(['RFM_Monetary_Score', 'RFM_Monetary_Value'])
        else:
            self._log("No monetary columns detected - skipping M component")
        
        # ========== FREQUENCY DETECTION ==========
        frequency_cols = self._pick_best_columns(
            df_analysis,
            self.frequency_keywords,
            k_max=1,
            dimension="frequency",
        )
        if frequency_cols:
            self._log(f"Detected {len(frequency_cols)} frequency column(s): {', '.join(frequency_cols[:3])}")
            self.rfm_sources['frequency'] = frequency_cols.copy()
            
            # Aggregate if multiple frequency columns
            if len(frequency_cols) == 1:
                frequency_value = pd.to_numeric(df[frequency_cols[0]], errors='coerce')
            else:
                # Sum counts/interactions to approximate interaction intensity
                frequency_value = df[frequency_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            
            # Create Frequency score (more orders = higher score)
            df_enhanced['RFM_Frequency_Score'] = self._calculate_rfm_score(frequency_value, reverse=False)
            df_enhanced['RFM_Frequency_Value'] = frequency_value
            self.rfm_features_created.extend(['RFM_Frequency_Score', 'RFM_Frequency_Value'])
        else:
            self._log("No frequency columns detected - skipping F component")
        
        # ========== RECENCY DETECTION ==========
        recency_cols = self._pick_best_columns(
            df_analysis,
            self.recency_keywords,
            k_max=1,
            dimension="recency",
        )
        if recency_cols:
            self._log(f"Detected {len(recency_cols)} recency column(s): {', '.join(recency_cols[:3])}")
            self.rfm_sources['recency'] = recency_cols.copy()

            # Use first recency column (or average if multiple)
            if len(recency_cols) == 1:
                recency_value = pd.to_numeric(df[recency_cols[0]], errors='coerce')
            else:
                # Average recency indicators
                recency_value = df[recency_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

            # Direction correction:
            # "days since last" -> lower is better (reverse=True)
            # tenure/membership age -> higher is better (reverse=False)
            _recency_names = " ".join([str(c).lower() for c in recency_cols])
            _is_days_since = any(kw in _recency_names for kw in self._recency_days_since_keywords)
            _is_tenure = any(kw in _recency_names for kw in ['tenure', 'member', 'vintage', 'seniority', 'subscription'])

            if _is_days_since and not _is_tenure:
                reverse = True
                self._log(f"  Recency direction: reverse ('{recency_cols[0]}' = days-since; lower is better)")
            elif _is_tenure:
                reverse = False
                self._log(f"  Recency direction: normal ('{recency_cols[0]}' = tenure; higher is better)")
            else:
                # Heuristic fallback:
                # very large medians often imply "elapsed since event" scales.
                reverse = recency_value.median() <= 90
                self._log(f"  Recency direction: {'reverse' if reverse else 'normal'} (median={recency_value.median():.1f}, heuristic)")

            df_enhanced['RFM_Recency_Score'] = self._calculate_rfm_score(
                recency_value,
                reverse=reverse,
            )
            df_enhanced['RFM_Recency_Value'] = recency_value
            self.rfm_features_created.extend(['RFM_Recency_Score', 'RFM_Recency_Value'])
        else:
            self._log("No recency columns detected - skipping R component")
        
        # ========== COMPOSITE RFM SCORE ==========
        score_cols = [c for c in self.rfm_features_created if 'Score' in c]
        if len(score_cols) >= 2:
            # Create weighted composite (Monetary gets 40%, Frequency 30%, Recency 30%)
            weights = {
                'RFM_Monetary_Score': 0.4,
                'RFM_Frequency_Score': 0.3,
                'RFM_Recency_Score': 0.3
            }
            
            composite = pd.Series(0.0, index=df_enhanced.index)
            total_weight = 0.0
            
            for col, weight in weights.items():
                if col in df_enhanced.columns:
                    composite += df_enhanced[col] * weight
                    total_weight += weight
            
            if total_weight > 0:
                composite = composite / total_weight  # Normalize
            
            df_enhanced['RFM_Composite_Score'] = composite
            self.rfm_features_created.append('RFM_Composite_Score')
            self._log(f"Created composite RFM score from {len(score_cols)} components")
            
            # ========== RFM SEGMENTS ==========
            # Use empirical quantiles to avoid hardcoded/manipulated segment boundaries.
            try:
                _r = pd.qcut(
                    df_enhanced['RFM_Composite_Score'].rank(method='first'),
                    q=4,
                    labels=['At_Risk', 'Developing', 'Established', 'Champions'],
                    duplicates='drop'
                )
                df_enhanced['RFM_Segment'] = _r.astype(str)
            except Exception:
                # Fallback when too few unique values
                _r2 = pd.cut(
                    df_enhanced['RFM_Composite_Score'],
                    bins=4,
                    labels=['At_Risk', 'Developing', 'Established', 'Champions'],
                    include_lowest=True
                )
                df_enhanced['RFM_Segment'] = _r2.astype(str)
            
            # One-hot encode segments for model
            segment_dummies = pd.get_dummies(df_enhanced['RFM_Segment'], prefix='RFM_Seg', dtype=int)
            df_enhanced = pd.concat([df_enhanced, segment_dummies], axis=1)
            
            self.rfm_features_created.append('RFM_Segment')
            self.rfm_features_created.extend(segment_dummies.columns.tolist())
            
            self._log(f"Created {len(segment_dummies.columns)} RFM segment features")
        else:
            self._log("Insufficient RFM components for composite score")
        
        # ========== VALIDATION ==========
        _found = []
        _missing = []
        for dim, cols in [('Monetary', monetary_cols), ('Frequency', frequency_cols), ('Recency', recency_cols)]:
            if cols:
                _found.append(f"{dim}={cols}")
            else:
                _missing.append(dim)
        if _found:
            self._log(f"RFM dimensions found: {'; '.join(_found)}")
        if _missing:
            self._log(f"RFM dimensions MISSING: {', '.join(_missing)}")

        # ========== SUMMARY ==========
        total_features = len(self.rfm_features_created)
        if total_features > 0:
            self._log(f"RFM Analysis Complete: {total_features} features created")
        else:
            self._log("No RFM features could be created from this dataset")
        
        return df_enhanced
    
    def get_feature_summary(self) -> Dict:
        """Get summary of created RFM features"""
        dimensions_detected = {
            'monetary': len(self.rfm_sources.get('monetary', [])) > 0,
            'frequency': len(self.rfm_sources.get('frequency', [])) > 0,
            'recency': len(self.rfm_sources.get('recency', [])) > 0,
        }
        quality_score = round(sum(dimensions_detected.values()) / 3, 3)
        return {
            'total_features': len(self.rfm_features_created),
            'feature_names': self.rfm_features_created,
            'detection_log': self.detection_log,
            'rfm_sources': self.rfm_sources,
            'dimensions_detected': dimensions_detected,
            'quality_score': quality_score,
        }


def quick_rfm_analysis(df: pd.DataFrame, target_col: Optional[str] = None, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for quick RFM analysis.
    
    Usage:
        enhanced_df, summary = quick_rfm_analysis(df, target_col='Churn')
    
    Returns:
        enhanced_df: DataFrame with RFM features
        summary: Dictionary with feature summary
    """
    analyzer = UniversalRFMAnalyzer(verbose=verbose)
    enhanced_df = analyzer.analyze_and_engineer(df, target_col=target_col)
    summary = analyzer.get_feature_summary()
    return enhanced_df, summary


if __name__ == "__main__":
    # Example usage
    print("Universal RFM Analyzer - Test Mode")
    print("=" * 60)
    
    # Create synthetic dataset
    np.random.seed(42)
    test_df = pd.DataFrame({
        'CustomerID': range(1, 101),
        'TotalRevenue': np.random.uniform(100, 5000, 100),
        'OrderCount': np.random.randint(1, 50, 100),
        'Tenure': np.random.randint(1, 60, 100),
        'Churn': np.random.choice([0, 1], 100)
    })
    
    print("\nTest Dataset:")
    print(test_df.head())
    
    # Run RFM analysis
    enhanced_df, summary = quick_rfm_analysis(test_df, target_col='Churn')
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"Original features: {len(test_df.columns)}")
    print(f"Enhanced features: {len(enhanced_df.columns)}")
    print(f"RFM features added: {summary['total_features']}")
    print("\nNew RFM features:")
    for feat in summary['feature_names']:
        print(f"  • {feat}")
    
    print("\nSample enhanced data:")
    rfm_cols = [c for c in enhanced_df.columns if 'RFM' in c]
    print(enhanced_df[['CustomerID'] + rfm_cols[:5]].head())
