"""
Professional Validation Engine for Churn Prediction
====================================================
Detects data leakage, ensures proper validation, and provides reliable diagnostics.
No UI changes - pure backend validation.

Author: Enhanced ML System
Date: February 2026
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class ChurnValidationEngine:
    """
    Professional validation engine that ensures model reliability.
    """
    
    def __init__(self, df, target_col, verbose=True):
        """
        Initialize validation engine.
        
        Parameters:
        -----------
        df : DataFrame
            Full dataset
        target_col : str
            Target column name
        verbose : bool
            Print diagnostic messages
        """
        self.df = df.copy()
        self.target_col = target_col
        self.verbose = verbose
        self.leakage_features = []
        self.validation_report = {}
        
    def detect_leakage(self, threshold=0.95):
        """
        Detect features with suspiciously high correlation to target.
        Universal method - works with any dataset structure.
        
        Returns:
        --------
        list : Features to remove
        """
        if self.verbose:
            print("\n" + "="*70)
            print("DATA LEAKAGE DETECTION")
            print("="*70)
        
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col].copy()
        
        # Universal target conversion - handles ANY format
        try:
            # First try: already numeric
            y = pd.to_numeric(y, errors='coerce')
            
            # Second try: common text mappings
            if y.isna().any() or y.dtype == 'object':
                y_str = self.df[self.target_col].astype(str).str.strip().str.lower()
                
                # Flexible mapping for common variations
                positive_values = ['yes', 'y', 'true', 't', '1', '1.0', 'churn', 'churned', 'left', 'exit', 'exited', 'positive', 'pos']
                negative_values = ['no', 'n', 'false', 'f', '0', '0.0', 'stay', 'stayed', 'retain', 'retained', 'negative', 'neg']
                
                y = y_str.apply(lambda x: 1 if x in positive_values else (0 if x in negative_values else np.nan))
            
            # Remove any remaining NaNs
            y = y.fillna(y.mode()[0] if not y.mode().empty else 0).astype(int)
            
            # Ensure binary
            if y.nunique() > 2:
                # Convert to binary by threshold (> median = 1)
                y = (y > y.median()).astype(int)
                
        except Exception as e:
            if self.verbose:
                print(f"WARNING: Could not convert target to binary: {e}")
                print("   Using original values...")
            y = pd.to_numeric(self.df[self.target_col], errors='coerce').fillna(0).astype(int)
        
        leakage_features = []
        
        for col in X.columns:
            try:
                series = X[col].copy()
                
                # Skip if constant or all missing
                if series.nunique(dropna=True) <= 1:
                    continue
                
                # Universal approach: Try to calculate predictive power regardless of type
                score = None
                corr_value = 0
                
                # Attempt 1: Numeric correlation (handles int, float, bool)
                if series.dtype.kind in 'bifcu':  # bool, int, float, complex, unicode
                    try:
                        numeric_series = pd.to_numeric(series, errors='coerce')
                        if numeric_series.notna().sum() > 10:  # Need enough values
                            # Handle edge cases
                            if numeric_series.std() > 0:
                                corr_value = abs(np.corrcoef(numeric_series.fillna(numeric_series.mean()), y)[0, 1])
                                if not np.isnan(corr_value) and corr_value > threshold:
                                    leakage_features.append({
                                        'feature': col,
                                        'correlation': corr_value,
                                        'reason': f'High correlation ({corr_value:.4f}) with target'
                                    })
                                    if self.verbose:
                                        print(f"WARNING: {col:30s} | Correlation: {corr_value:.4f} - LEAKAGE DETECTED")
                                    continue
                    except Exception:
                        pass
                
                # Attempt 2: Categorical predictive power (handles object, category, datetime)
                try:
                    # Convert to string categories
                    cat_series = series.astype(str).fillna('_MISSING_')
                    
                    # Skip if too many categories (likely an ID)
                    if cat_series.nunique() > min(100, len(cat_series) * 0.5):
                        continue
                    
                    # Calculate mean target per category
                    cat_df = pd.DataFrame({'cat': cat_series, 'target': y})
                    cat_means = cat_df.groupby('cat')['target'].agg(['mean', 'count'])
                    
                    # Filter categories with enough samples
                    cat_means = cat_means[cat_means['count'] >= 5]
                    
                    if len(cat_means) > 0:
                        separation = cat_means['mean'].max() - cat_means['mean'].min()
                        
                        # Check if categories perfectly separate target
                        if separation > 0.9 and cat_means['mean'].nunique() <= 3:
                            leakage_features.append({
                                'feature': col,
                                'correlation': separation,
                                'reason': f'Categories perfectly separate target (separation: {separation:.4f})'
                            })
                            if self.verbose:
                                print(f"WARNING: {col:30s} | Separation: {separation:.4f} - LEAKAGE DETECTED")
                
                except Exception:
                    pass
                    
            except Exception as e:
                if self.verbose:
                    print(f"INFO: {col:30s} | Skipped: {str(e)[:50]}")
                continue
        
        self.leakage_features = leakage_features
        
        if self.verbose:
            print("\n" + "-"*70)
            if leakage_features:
                print(f"ALERT: Found {len(leakage_features)} leaked features")
                print("\nRecommendation: Remove these features before training")
            else:
                print("SUCCESS: No obvious data leakage detected")
            print("="*70 + "\n")
        
        return [f['feature'] for f in leakage_features]
    
    def create_clean_dataset(self):
        """
        Create dataset with leakage features removed.
        
        Returns:
        --------
        DataFrame : Clean dataset
        """
        leaked_cols = [f['feature'] for f in self.leakage_features]
        
        if leaked_cols:
            if self.verbose:
                print(f"\nRemoving {len(leaked_cols)} leaked features...")
                for col in leaked_cols:
                    print(f"   - {col}")
            
            clean_df = self.df.drop(columns=leaked_cols, errors='ignore')
            
            if self.verbose:
                print(f"\nClean dataset: {clean_df.shape[0]} rows × {clean_df.shape[1]} columns")
            
            return clean_df
        else:
            if self.verbose:
                print("\nNo features to remove - dataset is clean")
            return self.df.copy()
    
    def validate_model_performance(self, model, X_train, X_test, y_train, y_test):
        """
        Universal model validation with proper metrics.
        Works with any model type and dataset structure.
        
        Parameters:
        -----------
        model : trained model
            The model to validate
        X_train, X_test : DataFrames or arrays
            Feature sets
        y_train, y_test : Series or arrays
            Target variables
        
        Returns:
        --------
        dict : Validation report
        """
        if self.verbose:
            print("\n" + "="*70)
            print("MODEL PERFORMANCE VALIDATION")
            print("="*70)
        
        # Ensure targets are numeric arrays
        try:
            y_train = np.array(y_train).ravel()
            y_test = np.array(y_test).ravel()
            
            # Handle any non-numeric values
            y_train = pd.to_numeric(pd.Series(y_train), errors='coerce').fillna(0).astype(int).values
            y_test = pd.to_numeric(pd.Series(y_test), errors='coerce').fillna(0).astype(int).values
        except Exception as e:
            if self.verbose:
                print(f"WARNING: Could not convert targets: {e}")
        
        # Get predictions - handle different model types
        try:
            if hasattr(model, 'predict_proba'):
                y_train_prob = model.predict_proba(X_train)
                y_test_prob = model.predict_proba(X_test)
                
                # Handle multi-class vs binary
                if y_train_prob.ndim > 1 and y_train_prob.shape[1] > 1:
                    y_train_prob = y_train_prob[:, 1]
                    y_test_prob = y_test_prob[:, 1]
                else:
                    y_train_prob = y_train_prob.ravel()
                    y_test_prob = y_test_prob.ravel()
                
                y_train_pred = (y_train_prob > 0.5).astype(int)
                y_test_pred = (y_test_prob > 0.5).astype(int)
            else:
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                y_train_prob = y_train_pred.astype(float)
                y_test_prob = y_test_pred.astype(float)
        except Exception as e:
            if self.verbose:
                print(f"WARNING: Prediction failed: {e}")
            # Fallback to random predictions
            y_train_pred = np.zeros(len(y_train))
            y_test_pred = np.zeros(len(y_test))
            y_train_prob = y_train_pred.astype(float)
            y_test_prob = y_test_pred.astype(float)
        
        # Calculate metrics with robust error handling
        def safe_metric(metric_func, y_true, y_pred, **kwargs):
            try:
                return metric_func(y_true, y_pred, **kwargs)
            except Exception:
                return 0.0
        
        train_metrics = {
            'accuracy': safe_metric(accuracy_score, y_train, y_train_pred),
            'precision': safe_metric(precision_score, y_train, y_train_pred, zero_division=0),
            'recall': safe_metric(recall_score, y_train, y_train_pred, zero_division=0),
            'f1': safe_metric(f1_score, y_train, y_train_pred, zero_division=0),
            'auc': safe_metric(roc_auc_score, y_train, y_train_prob) if len(np.unique(y_train)) == 2 else 0.5
        }
        
        test_metrics = {
            'accuracy': safe_metric(accuracy_score, y_test, y_test_pred),
            'precision': safe_metric(precision_score, y_test, y_test_pred, zero_division=0),
            'recall': safe_metric(recall_score, y_test, y_test_pred, zero_division=0),
            'f1': safe_metric(f1_score, y_test, y_test_pred, zero_division=0),
            'auc': safe_metric(roc_auc_score, y_test, y_test_prob) if len(np.unique(y_test)) == 2 else 0.5
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_test_prob) if hasattr(model, 'predict_proba') else 0
        }
        
        # Calculate gaps
        gaps = {
            'accuracy_gap': train_metrics['accuracy'] - test_metrics['accuracy'],
            'f1_gap': train_metrics['f1'] - test_metrics['f1'],
            'auc_gap': train_metrics['auc'] - test_metrics['auc']
        }
        
        # Assess quality
        issues = []
        warnings_list = []
        
        # Check for remaining leakage (universal threshold)
        if train_metrics['auc'] > 0.95 or test_metrics['auc'] > 0.95:
            issues.append("⚠️ CRITICAL: Near-perfect AUC suggests remaining data leakage")
            issues.append("   → Some features may have <0.85 correlation but still predict the target perfectly")
            issues.append("   → Review your data: Are any features measured AFTER the target event?")
        
        # Check for overfitting
        if gaps['f1_gap'] > 0.15 or gaps['auc_gap'] > 0.15:
            issues.append(f"⚠️ CRITICAL: Severe overfitting detected (F1 gap: {gaps['f1_gap']:.3f}, AUC gap: {gaps['auc_gap']:.3f})")
        elif gaps['f1_gap'] > 0.08 or gaps['auc_gap'] > 0.08:
            warnings_list.append(f"⚠️ WARNING: Moderate overfitting (F1 gap: {gaps['f1_gap']:.3f}, AUC gap: {gaps['auc_gap']:.3f})")
        
        # Check for realistic performance
        if test_metrics['auc'] < 0.55:
            issues.append("⚠️ CRITICAL: Model performs no better than random (AUC < 0.55)")
        elif test_metrics['f1'] < 0.3:
            warnings_list.append("⚠️ WARNING: Low F1 score suggests poor balance between precision and recall")
        
        # Print report
        if self.verbose:
            print("\n📈 TRAINING SET PERFORMANCE:")
            print(f"   Accuracy:  {train_metrics['accuracy']:.4f}")
            print(f"   Precision: {train_metrics['precision']:.4f}")
            print(f"   Recall:    {train_metrics['recall']:.4f}")
            print(f"   F1 Score:  {train_metrics['f1']:.4f}")
            print(f"   AUC:       {train_metrics['auc']:.4f}")
            
            print("\n📉 TEST SET PERFORMANCE:")
            print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"   Precision: {test_metrics['precision']:.4f}")
            print(f"   Recall:    {test_metrics['recall']:.4f}")
            print(f"   F1 Score:  {test_metrics['f1']:.4f}")
            print(f"   AUC:       {test_metrics['auc']:.4f}")
            
            print("\n📊 PERFORMANCE GAPS:")
            print(f"   Accuracy Gap: {gaps['accuracy_gap']:+.4f}")
            print(f"   F1 Gap:       {gaps['f1_gap']:+.4f}")
            print(f"   AUC Gap:      {gaps['auc_gap']:+.4f}")
            
            if issues:
                print("\n❌ CRITICAL ISSUES:")
                for issue in issues:
                    print(f"   {issue}")
            
            if warnings_list:
                print("\n⚠️  WARNINGS:")
                for warning in warnings_list:
                    print(f"   {warning}")
            
            if not issues and not warnings_list:
                print("\n✅ Model validation passed - performance looks realistic")
            
            print("\n" + "="*70)
        
        # Store report
        self.validation_report = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'gaps': gaps,
            'issues': issues,
            'warnings': warnings_list,
            'status': 'CRITICAL' if issues else ('WARNING' if warnings_list else 'HEALTHY')
        }
        
        return self.validation_report
    
    def get_expected_performance(self):
        """
        Provide realistic performance expectations for churn prediction.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("🎯 REALISTIC PERFORMANCE EXPECTATIONS FOR CHURN PREDICTION")
            print("="*70)
            print("\nExcellent Performance:")
            print("   AUC:       0.75 - 0.85")
            print("   F1 Score:  0.65 - 0.80")
            print("   Recall:    0.70 - 0.85  (Most important for churn!)")
            print("   Precision: 0.60 - 0.75")
            
            print("\nGood Performance:")
            print("   AUC:       0.65 - 0.75")
            print("   F1 Score:  0.55 - 0.65")
            print("   Recall:    0.60 - 0.70")
            print("   Precision: 0.50 - 0.60")
            
            print("\nAcceptable Performance:")
            print("   AUC:       0.60 - 0.65")
            print("   F1 Score:  0.45 - 0.55")
            print("   Recall:    0.50 - 0.60")
            print("   Precision: 0.40 - 0.50")
            
            print("\n💡 Key Points:")
            print("   • AUC > 0.90 almost always indicates data leakage")
            print("   • For churn, prioritize RECALL over precision")
            print("   • Train/test gap should be < 0.10 for F1 and AUC")
            print("   • Class imbalance is normal - use SMOTE")
            print("="*70 + "\n")
    
    def generate_report(self, save_path=None):
        """
        Generate comprehensive validation report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save report
        
        Returns:
        --------
        str : Report text
        """
        lines = []
        lines.append("="*80)
        lines.append("CHURN MODEL VALIDATION REPORT")
        lines.append("="*80)
        lines.append("")
        
        # Dataset info
        lines.append(f"Dataset: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        lines.append(f"Target: {self.target_col}")
        lines.append("")
        
        # Leakage detection
        lines.append("-"*80)
        lines.append("DATA LEAKAGE DETECTION")
        lines.append("-"*80)
        
        if self.leakage_features:
            lines.append(f"Status: ❌ {len(self.leakage_features)} leaked features detected")
            lines.append("")
            lines.append("Leaked Features:")
            for f in self.leakage_features:
                lines.append(f"  • {f['feature']}: {f['reason']}")
            lines.append("")
            lines.append("Action Required: Remove these features and retrain")
        else:
            lines.append("Status: ✅ No obvious leakage detected")
        
        lines.append("")
        
        # Model performance
        if self.validation_report:
            lines.append("-"*80)
            lines.append("MODEL PERFORMANCE")
            lines.append("-"*80)
            
            train = self.validation_report['train_metrics']
            test = self.validation_report['test_metrics']
            gaps = self.validation_report['gaps']
            
            lines.append("")
            lines.append("Training Set:")
            lines.append(f"  Accuracy:  {train['accuracy']:.4f}")
            lines.append(f"  Precision: {train['precision']:.4f}")
            lines.append(f"  Recall:    {train['recall']:.4f}")
            lines.append(f"  F1 Score:  {train['f1']:.4f}")
            lines.append(f"  AUC:       {train['auc']:.4f}")
            
            lines.append("")
            lines.append("Test Set:")
            lines.append(f"  Accuracy:  {test['accuracy']:.4f}")
            lines.append(f"  Precision: {test['precision']:.4f}")
            lines.append(f"  Recall:    {test['recall']:.4f}")
            lines.append(f"  F1 Score:  {test['f1']:.4f}")
            lines.append(f"  AUC:       {test['auc']:.4f}")
            
            lines.append("")
            lines.append("Performance Gaps:")
            lines.append(f"  F1 Gap:  {gaps['f1_gap']:+.4f}")
            lines.append(f"  AUC Gap: {gaps['auc_gap']:+.4f}")
            
            lines.append("")
            lines.append(f"Overall Status: {self.validation_report['status']}")
            
            if self.validation_report['issues']:
                lines.append("")
                lines.append("Critical Issues:")
                for issue in self.validation_report['issues']:
                    lines.append(f"  {issue}")
            
            if self.validation_report['warnings']:
                lines.append("")
                lines.append("Warnings:")
                for warning in self.validation_report['warnings']:
                    lines.append(f"  {warning}")
        
        lines.append("")
        lines.append("="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            if self.verbose:
                print(f"\n💾 Report saved to: {save_path}")
        
        return report


def quick_validate(df, target_col, leakage_threshold=0.95):
    """
    Quick validation function for easy use.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    target_col : str
        Target column name
    leakage_threshold : float
        Correlation threshold for leakage detection
    
    Returns:
    --------
    tuple : (clean_df, leaked_features, engine)
    """
    engine = ChurnValidationEngine(df, target_col, verbose=True)
    
    # Detect leakage
    leaked_features = engine.detect_leakage(threshold=leakage_threshold)
    
    # Get clean dataset
    clean_df = engine.create_clean_dataset()
    
    # Show expectations
    engine.get_expected_performance()
    
    return clean_df, leaked_features, engine
