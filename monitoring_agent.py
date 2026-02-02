"""
Autonomous Monitoring Agent for AutoML Training
Detects problems and automatically applies fixes until resolved
"""

import traceback
import time
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


class MonitoringAgent:
    """
    Autonomous agent that monitors training, detects problems, and applies fixes automatically.
    Retries with different strategies until problem is resolved or max attempts reached.
    """
    
    def __init__(self, max_retries: int = 3, verbose: bool = True):
        self.max_retries = max_retries
        self.verbose = verbose
        self.fix_log = []
        self.problem_history = []
        
    def detect_problem_type(self, error: Exception, context: str = "") -> str:
        """Classify error type to apply appropriate fix"""
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        # Data-related problems
        if any(x in error_msg for x in ['nan', 'inf', 'missing', 'null', 'empty']):
            return "DATA_QUALITY"
        if any(x in error_msg for x in ['dtype', 'type', 'convert', 'cast']):
            return "DATA_TYPE"
        if any(x in error_msg for x in ['shape', 'dimension', 'size mismatch']):
            return "DATA_SHAPE"
        if any(x in error_msg for x in ['duplicate', 'index']):
            return "DATA_INDEX"
            
        # Model training problems
        if any(x in error_msg for x in ['converge', 'iteration', 'max_iter']):
            return "CONVERGENCE"
        if any(x in error_msg for x in ['memory', 'allocation', 'ram']):
            return "MEMORY"
        if any(x in error_msg for x in ['feature', 'column not found']):
            return "FEATURE_MISMATCH"
        if any(x in error_msg for x in ['imbalance', 'class', 'label']):
            return "CLASS_IMBALANCE"
            
        # Model performance problems
        if any(x in error_msg for x in ['overfit', 'variance']):
            return "OVERFITTING"
        if any(x in error_msg for x in ['underfit', 'bias']):
            return "UNDERFITTING"
            
        # Technical problems
        if any(x in error_msg for x in ['gpu', 'cuda', 'device']):
            return "GPU_ISSUE"
        if any(x in error_msg for x in ['timeout', 'time', 'deadline']):
            return "TIMEOUT"
        if 'KeyError' in error_type:
            return "KEY_ERROR"
        if 'ValueError' in error_type:
            return "VALUE_ERROR"
            
        return "UNKNOWN"
    
    def apply_fix(self, problem_type: str, data: pd.DataFrame, 
                  config: Dict[str, Any], attempt: int) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """
        Apply appropriate fix based on problem type.
        Returns: (fixed_data, updated_config, fix_description)
        """
        fix_applied = ""
        
        if problem_type == "DATA_QUALITY":
            # Fix NaN, Inf, missing values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Drop columns with >50% missing
            missing_ratio = data.isnull().sum() / len(data)
            drop_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
            if drop_cols:
                data = data.drop(columns=drop_cols)
                fix_applied = f"Dropped {len(drop_cols)} columns with >50% missing values"
            
            # Fill numeric NaN with median
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].isnull().any():
                    data[col] = data[col].fillna(data[col].median())
            
            # Fill categorical NaN with mode or 'missing'
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if data[col].isnull().any():
                    mode_val = data[col].mode()
                    if len(mode_val) > 0:
                        data[col] = data[col].fillna(mode_val[0])
                    else:
                        data[col] = data[col].fillna('missing')
            
            fix_applied += " | Imputed all missing values"
            
        elif problem_type == "DATA_TYPE":
            # Fix dtype issues
            for col in data.columns:
                try:
                    # Try converting to numeric if it looks like numbers
                    if data[col].dtype == 'object':
                        data[col] = pd.to_numeric(data[col], errors='ignore')
                except Exception:
                    pass
            fix_applied = "Fixed data type inconsistencies"
            
        elif problem_type == "DATA_SHAPE":
            # Remove empty rows/columns
            data = data.dropna(how='all', axis=0)  # Drop empty rows
            data = data.dropna(how='all', axis=1)  # Drop empty columns
            data = data.reset_index(drop=True)
            fix_applied = "Removed empty rows/columns and reset index"
            
        elif problem_type == "DATA_INDEX":
            # Fix duplicate index
            data = data.reset_index(drop=True)
            data = data.drop_duplicates()
            fix_applied = "Fixed index issues and removed duplicates"
            
        elif problem_type == "CONVERGENCE":
            # Increase iterations and relax convergence criteria
            if 'max_iter' in config:
                config['max_iter'] = config.get('max_iter', 100) * 2
            config['early_stopping'] = True
            config['n_iter'] = min(config.get('n_iter', 10) + 10, 50)
            fix_applied = f"Increased iterations (attempt {attempt})"
            
        elif problem_type == "MEMORY":
            # Reduce memory footprint
            # Disable polynomial features if enabled
            if config.get('polynomial_features', False):
                config['polynomial_features'] = False
                fix_applied = "Disabled polynomial features to reduce memory"
            
            # Reduce number of models
            if 'n_select' in config:
                config['n_select'] = max(1, config['n_select'] - 1)
                fix_applied += " | Reduced model count"
            
            # Sample data if too large
            if len(data) > 50000:
                data = data.sample(n=50000, random_state=123)
                fix_applied += " | Sampled to 50k rows"
                
        elif problem_type == "FEATURE_MISMATCH":
            # Clean column names
            data.columns = [str(col).strip().replace(' ', '_') for col in data.columns]
            fix_applied = "Cleaned feature names"
            
        elif problem_type == "CLASS_IMBALANCE":
            # Adjust SMOTE settings
            config['fix_imbalance'] = True
            config['fix_imbalance_method'] = 'smote'
            fix_applied = "Enabled SMOTE for class imbalance"
            
        elif problem_type == "OVERFITTING":
            # Add regularization
            config['remove_multicollinearity'] = True
            config['multicollinearity_threshold'] = 0.8
            config['feature_selection'] = True
            config['n_features_to_select'] = 0.7  # More aggressive
            fix_applied = "Increased regularization to reduce overfitting"
            
        elif problem_type == "UNDERFITTING":
            # Increase model complexity
            if config.get('feature_selection', False):
                config['n_features_to_select'] = 0.9  # Keep more features
            config['polynomial_features'] = True
            config['polynomial_degree'] = 2
            fix_applied = "Increased model complexity"
            
        elif problem_type == "GPU_ISSUE":
            # Force CPU mode
            config['use_gpu'] = False
            fix_applied = "Switched to CPU mode"
            
        elif problem_type == "TIMEOUT":
            # Reduce computational load
            config['fold'] = max(3, config.get('fold', 10) - 2)
            config['n_iter'] = max(5, config.get('n_iter', 10) - 5)
            config['turbo'] = True
            fix_applied = "Reduced CV folds and iterations for speed"
            
        elif problem_type == "KEY_ERROR" or problem_type == "VALUE_ERROR":
            # Generic fixes for common errors
            data = data.dropna()
            data = data.reset_index(drop=True)
            config['verbose'] = False
            config['errors'] = 'ignore'
            fix_applied = "Applied generic error fixes"
            
        else:  # UNKNOWN
            # Apply conservative fallback strategy
            data = data.dropna()
            data = data.drop_duplicates()
            config['use_gpu'] = False
            config['verbose'] = False
            config['errors'] = 'ignore'
            config['turbo'] = True
            fix_applied = "Applied fallback strategy (conservative mode)"
        
        return data, config, fix_applied
    
    def monitor_and_fix(self, train_function, data: pd.DataFrame, 
                       config: Dict[str, Any], target_col: str) -> Tuple[Any, List[str]]:
        """
        Monitor training execution and automatically fix problems.
        
        Args:
            train_function: Function to execute (should accept data, config, target_col)
            data: Training data
            config: Configuration dictionary
            target_col: Target column name
            
        Returns:
            (result, fix_log): Training result and list of fixes applied
        """
        self.fix_log = []
        self.problem_history = []
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.verbose and attempt > 0:
                    print(f"\n🔄 Retry attempt {attempt}/{self.max_retries}")
                
                # Execute training function
                result = train_function(data, config, target_col)
                
                # Success!
                if attempt == 0:
                    self.fix_log.append("✓ Training completed successfully on first attempt")
                else:
                    self.fix_log.append(f"✓ Training succeeded after {attempt} fix(es)")
                
                return result, self.fix_log
                
            except Exception as e:
                # Detect problem type
                problem_type = self.detect_problem_type(e)
                self.problem_history.append({
                    'attempt': attempt,
                    'problem_type': problem_type,
                    'error': str(e)[:200],
                    'traceback': traceback.format_exc()[:500]
                })
                
                if self.verbose:
                    print(f"\n⚠️  Problem detected: {problem_type}")
                    print(f"Error: {str(e)[:150]}")
                
                # Check if we have retries left
                if attempt >= self.max_retries:
                    self.fix_log.append(f"❌ Failed after {self.max_retries} attempts")
                    self.fix_log.append(f"Last error: {problem_type} - {str(e)[:100]}")
                    raise e  # Re-raise if all retries exhausted
                
                # Apply fix
                data, config, fix_description = self.apply_fix(
                    problem_type, data, config, attempt + 1
                )
                
                self.fix_log.append(f"🔧 Attempt {attempt + 1}: {problem_type} → {fix_description}")
                
                if self.verbose:
                    print(f"Applied fix: {fix_description}")
                
                # Wait before retry (exponential backoff)
                time.sleep(0.5 * (attempt + 1))
        
        # Should never reach here, but just in case
        raise RuntimeError("Training failed after all retry attempts")
    
    def get_fix_summary(self) -> str:
        """Get human-readable summary of fixes applied"""
        if not self.fix_log:
            return "No issues detected"
        
        summary = "\n".join(f"• {log}" for log in self.fix_log)
        return summary
    
    def get_problem_report(self) -> Dict[str, Any]:
        """Get detailed report of all problems encountered"""
        return {
            'total_problems': len(self.problem_history),
            'problems': self.problem_history,
            'fixes_applied': self.fix_log,
            'success': any('succeeded' in log or 'successfully' in log for log in self.fix_log)
        }


def create_monitored_training_wrapper(clf, monitoring_agent: MonitoringAgent):
    """
    Create a wrapper function that can be used with monitoring_agent.monitor_and_fix()
    """
    def training_function(data: pd.DataFrame, config: Dict[str, Any], target_col: str):
        """Wrapper function for PyCaret training pipeline"""
        
        # Setup with monitoring
        _ = clf.setup(
            data=data,
            target=target_col,
            **config
        )
        
        # Compare models
        tree_models = ['rf', 'et', 'xgboost', 'gbc', 'lightgbm']
        best = clf.compare_models(
            include=tree_models,
            sort='AUC',
            fold=config.get('fold', 10),
            turbo=config.get('turbo', False),
            errors='ignore',
            n_select=config.get('n_select', 5)
        )
        
        return best
    
    return training_function
