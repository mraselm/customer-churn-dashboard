"""
Universal RFM (Recency, Frequency, Monetary) Analyzer
Automatically detects and creates RFM features from any dataset structure.
Works with any column names - no rigid requirements.
"""

import pandas as pd
import numpy as np
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
        
        # Universal keyword mappings for column detection
        self.monetary_keywords = [
            'monetary', 'revenue', 'amount', 'spending', 'spend', 'price', 
            'sales', 'value', 'ltv', 'clv', 'lifetime', 'total', 'payment',
            'charge', 'cost', 'cashback', 'balance', 'income'
        ]
        
        self.frequency_keywords = [
            'frequency', 'order', 'purchase', 'transaction', 'visit', 'count',
            'number', 'quantity', 'times', 'occurrences', 'sessions', 'trips',
            'bookings', 'renewals', 'interactions'
        ]
        
        self.recency_keywords = [
            'recency', 'tenure', 'age', 'duration', 'last', 'recent', 'days',
            'months', 'years', 'since', 'member', 'subscription', 'account',
            'dayslastorder', 'vintage', 'seniority'
        ]
    
    def _log(self, message: str):
        """Log detection progress"""
        self.detection_log.append(message)
        if self.verbose:
            print(f"[RFM] {message}")
    
    def _detect_columns(self, df: pd.DataFrame, keywords: List[str]) -> List[str]:
        """
        Detect columns matching keyword patterns.
        Returns list of column names.
        """
        detected = []
        for col in df.columns:
            col_lower = col.lower()
            for keyword in keywords:
                if keyword in col_lower:
                    # Must be numeric or convertible to numeric
                    try:
                        pd.to_numeric(df[col], errors='coerce')
                        detected.append(col)
                        break
                    except:
                        continue
        return detected
    
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
        df_enhanced = df.copy()
        self._log("Starting Universal RFM Analysis...")
        
        # Exclude target column from analysis
        analysis_cols = [c for c in df.columns if c != target_col]
        df_analysis = df[analysis_cols]
        
        # ========== MONETARY DETECTION ==========
        monetary_cols = self._detect_columns(df_analysis, self.monetary_keywords)
        if monetary_cols:
            self._log(f"✓ Detected {len(monetary_cols)} monetary column(s): {', '.join(monetary_cols[:3])}")
            
            # Aggregate if multiple monetary columns
            if len(monetary_cols) == 1:
                monetary_value = pd.to_numeric(df[monetary_cols[0]], errors='coerce')
            else:
                # Sum all monetary columns (total customer spend)
                monetary_value = df[monetary_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            
            # Create Monetary score (higher spend = higher score)
            df_enhanced['RFM_Monetary_Score'] = self._calculate_rfm_score(monetary_value, reverse=False)
            df_enhanced['RFM_Monetary_Value'] = monetary_value  # Raw value for reference
            self.rfm_features_created.extend(['RFM_Monetary_Score', 'RFM_Monetary_Value'])
        else:
            self._log("⚠ No monetary columns detected - skipping M component")
        
        # ========== FREQUENCY DETECTION ==========
        frequency_cols = self._detect_columns(df_analysis, self.frequency_keywords)
        if frequency_cols:
            self._log(f"✓ Detected {len(frequency_cols)} frequency column(s): {', '.join(frequency_cols[:3])}")
            
            # Aggregate if multiple frequency columns
            if len(frequency_cols) == 1:
                frequency_value = pd.to_numeric(df[frequency_cols[0]], errors='coerce')
            else:
                # Sum all frequency columns (total interactions)
                frequency_value = df[frequency_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            
            # Create Frequency score (more orders = higher score)
            df_enhanced['RFM_Frequency_Score'] = self._calculate_rfm_score(frequency_value, reverse=False)
            df_enhanced['RFM_Frequency_Value'] = frequency_value
            self.rfm_features_created.extend(['RFM_Frequency_Score', 'RFM_Frequency_Value'])
        else:
            self._log("⚠ No frequency columns detected - skipping F component")
        
        # ========== RECENCY DETECTION ==========
        recency_cols = self._detect_columns(df_analysis, self.recency_keywords)
        if recency_cols:
            self._log(f"✓ Detected {len(recency_cols)} recency column(s): {', '.join(recency_cols[:3])}")
            
            # Use first recency column (or average if multiple)
            if len(recency_cols) == 1:
                recency_value = pd.to_numeric(df[recency_cols[0]], errors='coerce')
            else:
                # Average recency indicators
                recency_value = df[recency_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            
            # Create Recency score (LOWER recency/longer tenure = HIGHER score)
            # Note: For tenure, high values are good; for days_since_last, low values are good
            # We'll use a heuristic: if median > 100, assume it's tenure (reverse=False)
            median_val = recency_value.median()
            is_tenure = median_val > 100 or any(kw in recency_cols[0].lower() for kw in ['tenure', 'age', 'member', 'vintage'])
            
            df_enhanced['RFM_Recency_Score'] = self._calculate_rfm_score(
                recency_value, 
                reverse=not is_tenure  # If tenure, don't reverse; if days_since, reverse
            )
            df_enhanced['RFM_Recency_Value'] = recency_value
            self.rfm_features_created.extend(['RFM_Recency_Score', 'RFM_Recency_Value'])
        else:
            self._log("⚠ No recency columns detected - skipping R component")
        
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
            self._log(f"✓ Created composite RFM score from {len(score_cols)} components")
            
            # ========== RFM SEGMENTS ==========
            # Create customer segments based on composite score
            df_enhanced['RFM_Segment'] = pd.cut(
                df_enhanced['RFM_Composite_Score'],
                bins=[0, 2, 3, 4, 6],
                labels=['At_Risk', 'Developing', 'Established', 'Champions'],
                include_lowest=True
            )
            
            # One-hot encode segments for model
            segment_dummies = pd.get_dummies(df_enhanced['RFM_Segment'], prefix='RFM_Seg', dtype=int)
            df_enhanced = pd.concat([df_enhanced, segment_dummies], axis=1)
            
            self.rfm_features_created.append('RFM_Segment')
            self.rfm_features_created.extend(segment_dummies.columns.tolist())
            
            self._log(f"✓ Created {len(segment_dummies.columns)} RFM segment features")
        else:
            self._log("⚠ Insufficient RFM components for composite score")
        
        # ========== SUMMARY ==========
        total_features = len(self.rfm_features_created)
        if total_features > 0:
            self._log(f"✅ RFM Analysis Complete: {total_features} features created")
        else:
            self._log("⚠ No RFM features could be created from this dataset")
        
        return df_enhanced
    
    def get_feature_summary(self) -> Dict:
        """Get summary of created RFM features"""
        return {
            'total_features': len(self.rfm_features_created),
            'feature_names': self.rfm_features_created,
            'detection_log': self.detection_log
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
