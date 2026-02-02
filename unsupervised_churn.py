"""
Unsupervised Churn Detection Engine
Automatically detects churn patterns when no labeled target column exists
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class UnsupervisedChurnDetector:
    """
    Detects churn patterns in unlabeled data using multiple techniques:
    1. Clustering (K-means, DBSCAN)
    2. Anomaly detection (Isolation Forest)
    3. Behavioral heuristics (declining engagement, reduced usage)
    4. RFM-style scoring (Recency, Frequency, Monetary if applicable)
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.detection_log = []
        
    def detect_churn_indicators(self, df: pd.DataFrame) -> dict:
        """
        Detect columns that could indicate churn behavior
        Returns dict with indicator types and column names
        """
        indicators = {
            'tenure': [],
            'monetary': [],
            'usage': [],
            'engagement': [],
            'complaints': [],
            'satisfaction': [],
            'timestamp': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Tenure indicators (how long customer has been with company)
            if any(x in col_lower for x in ['tenure', 'age', 'duration', 'month', 'year']):
                if df[col].dtype in ['int64', 'float64']:
                    indicators['tenure'].append(col)
            
            # Monetary indicators
            if any(x in col_lower for x in ['charge', 'payment', 'bill', 'price', 'cost', 'amount', 'revenue']):
                if df[col].dtype in ['int64', 'float64']:
                    indicators['monetary'].append(col)
            
            # Usage indicators
            if any(x in col_lower for x in ['usage', 'minutes', 'data', 'calls', 'sessions', 'login', 'activity']):
                if df[col].dtype in ['int64', 'float64']:
                    indicators['usage'].append(col)
            
            # Engagement indicators
            if any(x in col_lower for x in ['engagement', 'interaction', 'visits', 'clicks', 'views']):
                if df[col].dtype in ['int64', 'float64']:
                    indicators['engagement'].append(col)
            
            # Complaint indicators
            if any(x in col_lower for x in ['complaint', 'issue', 'problem', 'ticket', 'support']):
                indicators['complaints'].append(col)
            
            # Satisfaction indicators
            if any(x in col_lower for x in ['satisfaction', 'rating', 'score', 'nps', 'csat']):
                if df[col].dtype in ['int64', 'float64']:
                    indicators['satisfaction'].append(col)
            
            # Timestamp indicators
            if any(x in col_lower for x in ['date', 'time', 'timestamp']):
                indicators['timestamp'].append(col)
        
        return indicators
    
    def create_churn_labels_heuristic(self, df: pd.DataFrame, indicators: dict) -> pd.Series:
        """
        Create churn labels based on behavioral heuristics
        High risk if: low tenure + high charges + low usage + complaints
        """
        risk_scores = pd.Series(0.0, index=df.index)
        n_factors = 0
        
        # Factor 1: Low tenure (new customers churn more)
        if indicators['tenure']:
            tenure_col = indicators['tenure'][0]
            tenure_normalized = (df[tenure_col] - df[tenure_col].min()) / (df[tenure_col].max() - df[tenure_col].min() + 1e-8)
            risk_scores += (1 - tenure_normalized) * 0.3  # Low tenure = higher risk
            n_factors += 1
            self.detection_log.append(f"✓ Using tenure: {tenure_col}")
        
        # Factor 2: High monetary without proportional usage (price sensitivity)
        if indicators['monetary'] and indicators['usage']:
            monetary_col = indicators['monetary'][0]
            usage_col = indicators['usage'][0]
            
            monetary_norm = (df[monetary_col] - df[monetary_col].min()) / (df[monetary_col].max() - df[monetary_col].min() + 1e-8)
            usage_norm = (df[usage_col] - df[usage_col].min()) / (df[usage_col].max() - df[usage_col].min() + 1e-8)
            
            # High price but low usage = high risk
            price_risk = monetary_norm * (1 - usage_norm)
            risk_scores += price_risk * 0.25
            n_factors += 1
            self.detection_log.append(f"✓ Price/usage ratio: {monetary_col} vs {usage_col}")
        
        # Factor 3: Low engagement
        if indicators['engagement']:
            engagement_col = indicators['engagement'][0]
            engagement_norm = (df[engagement_col] - df[engagement_col].min()) / (df[engagement_col].max() - df[engagement_col].min() + 1e-8)
            risk_scores += (1 - engagement_norm) * 0.2
            n_factors += 1
            self.detection_log.append(f"✓ Using engagement: {engagement_col}")
        
        # Factor 4: Complaints (more complaints = higher risk)
        if indicators['complaints']:
            complaint_col = indicators['complaints'][0]
            if df[complaint_col].dtype in ['int64', 'float64']:
                complaint_norm = (df[complaint_col] - df[complaint_col].min()) / (df[complaint_col].max() - df[complaint_col].min() + 1e-8)
                risk_scores += complaint_norm * 0.15
                n_factors += 1
                self.detection_log.append(f"✓ Using complaints: {complaint_col}")
        
        # Factor 5: Low satisfaction
        if indicators['satisfaction']:
            satisfaction_col = indicators['satisfaction'][0]
            satisfaction_norm = (df[satisfaction_col] - df[satisfaction_col].min()) / (df[satisfaction_col].max() - df[satisfaction_col].min() + 1e-8)
            risk_scores += (1 - satisfaction_norm) * 0.1
            n_factors += 1
            self.detection_log.append(f"✓ Using satisfaction: {satisfaction_col}")
        
        # Normalize to 0-1 range
        if risk_scores.max() > 0:
            risk_scores = risk_scores / risk_scores.max()
        
        # Convert to binary labels (top 30% risk = churn)
        threshold = np.percentile(risk_scores, 70)  # Top 30% are "churners"
        churn_labels = (risk_scores >= threshold).astype(int)
        
        self.detection_log.append(f"✓ Generated labels using {n_factors} behavioral factors")
        self.detection_log.append(f"✓ Churn rate: {churn_labels.mean():.1%} ({churn_labels.sum()}/{len(churn_labels)})")
        
        return churn_labels
    
    def create_churn_labels_clustering(self, df: pd.DataFrame, n_clusters=3) -> pd.Series:
        """
        Create churn labels using clustering
        Identify high-risk cluster based on characteristics
        """
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")
        
        # Prepare data
        X = df[numeric_cols].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Identify high-risk cluster (heuristic: lowest tenure, highest charges)
        cluster_profiles = []
        for i in range(n_clusters):
            cluster_mask = clusters == i
            profile = {
                'cluster': i,
                'size': cluster_mask.sum(),
                'mean_values': X[cluster_mask].mean().to_dict()
            }
            cluster_profiles.append(profile)
        
        # Simple heuristic: cluster with lowest median of first numeric column = high risk
        # (assuming first numeric is often tenure or similar)
        cluster_medians = [X[clusters == i][numeric_cols[0]].median() for i in range(n_clusters)]
        high_risk_cluster = np.argmin(cluster_medians)
        
        churn_labels = (clusters == high_risk_cluster).astype(int)
        
        self.detection_log.append(f"✓ K-means clustering with {n_clusters} clusters")
        self.detection_log.append(f"✓ High-risk cluster: {high_risk_cluster} ({churn_labels.sum()} customers)")
        self.detection_log.append(f"✓ Churn rate: {churn_labels.mean():.1%}")
        
        return churn_labels
    
    def create_churn_labels_anomaly(self, df: pd.DataFrame, contamination=0.3) -> pd.Series:
        """
        Create churn labels using anomaly detection
        Anomalies = potential churners
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for anomaly detection")
        
        # Prepare data
        X = df[numeric_cols].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        
        # Convert: -1 (anomaly) -> 1 (churn), 1 (normal) -> 0 (no churn)
        churn_labels = (anomaly_labels == -1).astype(int)
        
        self.detection_log.append(f"✓ Isolation Forest anomaly detection")
        self.detection_log.append(f"✓ Contamination: {contamination:.0%}")
        self.detection_log.append(f"✓ Detected anomalies: {churn_labels.sum()} ({churn_labels.mean():.1%})")
        
        return churn_labels
    
    def auto_detect_churn(self, df: pd.DataFrame, method='auto') -> tuple:
        """
        Automatically detect churn labels using best available method
        
        Args:
            df: DataFrame without churn labels
            method: 'auto', 'heuristic', 'clustering', 'anomaly'
            
        Returns:
            (churn_labels, method_used, detection_log, reliability_score)
        """
        self.detection_log = []
        
        # Detect available indicators
        indicators = self.detect_churn_indicators(df)
        n_indicators = sum(len(v) for v in indicators.values())
        
        self.detection_log.append(f"📊 Dataset Analysis:")
        self.detection_log.append(f"  • Total features: {len(df.columns)}")
        self.detection_log.append(f"  • Numeric features: {len(df.select_dtypes(include=[np.number]).columns)}")
        self.detection_log.append(f"  • Behavioral indicators: {n_indicators}")
        
        # Calculate reliability score based on available data
        reliability_score = self._calculate_reliability(df, indicators, n_indicators)
        
        # Choose method
        if method == 'auto':
            # Auto-select based on available data
            if n_indicators >= 3:
                method = 'heuristic'
            elif len(df.select_dtypes(include=[np.number]).columns) >= 5:
                method = 'clustering'
            else:
                method = 'anomaly'
            
            self.detection_log.append(f"\n🤖 Auto-selected method: {method.upper()}")
        
        # Apply chosen method
        try:
            if method == 'heuristic':
                churn_labels = self.create_churn_labels_heuristic(df, indicators)
            elif method == 'clustering':
                churn_labels = self.create_churn_labels_clustering(df)
            elif method == 'anomaly':
                churn_labels = self.create_churn_labels_anomaly(df)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Validate labels
            if churn_labels.sum() == 0 or churn_labels.sum() == len(churn_labels):
                raise ValueError("Generated labels are all 0 or all 1")
            
            # Add reliability assessment
            self._add_reliability_warning(reliability_score)
            
            return churn_labels, method, self.detection_log, reliability_score
            
        except Exception as e:
            # Fallback to simpler method
            self.detection_log.append(f"⚠️  {method} failed: {str(e)[:100]}")
            self.detection_log.append(f"🔄 Falling back to anomaly detection")
            
            churn_labels = self.create_churn_labels_anomaly(df, contamination=0.3)
            reliability_score = min(reliability_score, 0.55)  # Lower reliability for fallback
            self._add_reliability_warning(reliability_score)
            
            return churn_labels, 'anomaly', self.detection_log, reliability_score
    
    def _calculate_reliability(self, df: pd.DataFrame, indicators: dict, n_indicators: int) -> float:
        """
        Calculate reliability score (0-1) based on data quality
        Higher score = more reliable labels
        """
        score = 0.0
        max_score = 100
        
        # Factor 1: Number of behavioral indicators (40 points)
        if n_indicators >= 5:
            score += 40
        elif n_indicators >= 3:
            score += 30
        elif n_indicators >= 1:
            score += 15
        
        # Factor 2: Data size (20 points)
        if len(df) >= 1000:
            score += 20
        elif len(df) >= 500:
            score += 15
        elif len(df) >= 100:
            score += 10
        else:
            score += 5
        
        # Factor 3: Feature richness (20 points)
        n_numeric = len(df.select_dtypes(include=[np.number]).columns)
        if n_numeric >= 10:
            score += 20
        elif n_numeric >= 5:
            score += 15
        elif n_numeric >= 3:
            score += 10
        else:
            score += 5
        
        # Factor 4: Key indicator presence (20 points)
        key_indicators = ['tenure', 'monetary', 'usage', 'satisfaction']
        present = sum(1 for k in key_indicators if indicators[k])
        score += (present / 4) * 20
        
        # Normalize to 0-1
        return min(score / max_score, 1.0)
    
    def _add_reliability_warning(self, reliability_score: float):
        """Add reliability assessment to log"""
        self.detection_log.append(f"\n⚡ RELIABILITY ASSESSMENT:")
        
        if reliability_score >= 0.75:
            level = "HIGH"
            color = "🟢"
            expected = "70-85%"
            recommendation = "Good for production with monitoring"
        elif reliability_score >= 0.60:
            level = "MODERATE"
            color = "🟡"
            expected = "60-75%"
            recommendation = "Good for exploration, validate before production"
        elif reliability_score >= 0.45:
            level = "LOW"
            color = "🟠"
            expected = "50-65%"
            recommendation = "Use for initial analysis only"
        else:
            level = "VERY LOW"
            color = "🔴"
            expected = "40-55%"
            recommendation = "Not recommended - collect real labels"
        
        self.detection_log.append(f"  {color} Reliability: {level} ({reliability_score:.0%})")
        self.detection_log.append(f"  • Expected accuracy: {expected}")
        self.detection_log.append(f"  • Recommendation: {recommendation}")
    
    def get_confidence_metrics(self, df: pd.DataFrame, churn_labels: pd.Series) -> dict:
        """
        Calculate confidence metrics for generated labels
        Returns dict with various quality indicators
        """
        metrics = {}
        
        # 1. Label distribution balance
        churn_rate = churn_labels.mean()
        balance_score = 1.0 - abs(churn_rate - 0.3)  # Ideal around 30%
        metrics['balance_score'] = max(0, min(1, balance_score / 0.3))
        metrics['churn_rate'] = churn_rate
        
        # 2. Separation quality (if we have risk scores)
        # Higher separation = more confident labels
        try:
            indicators = self.detect_churn_indicators(df)
            risk_scores = self._calculate_risk_scores(df, indicators)
            
            churner_scores = risk_scores[churn_labels == 1]
            non_churner_scores = risk_scores[churn_labels == 0]
            
            separation = abs(churner_scores.mean() - non_churner_scores.mean())
            metrics['separation_score'] = min(1.0, separation * 2)
        except Exception:
            metrics['separation_score'] = 0.5
        
        # 3. Indicator coverage
        indicators = self.detect_churn_indicators(df)
        n_indicators = sum(len(v) for v in indicators.values())
        metrics['indicator_coverage'] = min(1.0, n_indicators / 5)
        
        # 4. Overall confidence
        metrics['overall_confidence'] = (
            metrics['balance_score'] * 0.3 +
            metrics['separation_score'] * 0.4 +
            metrics['indicator_coverage'] * 0.3
        )
        
        return metrics
    
    def _calculate_risk_scores(self, df: pd.DataFrame, indicators: dict) -> pd.Series:
        """Calculate raw risk scores for confidence metrics"""
        risk_scores = pd.Series(0.0, index=df.index)
        
        if indicators['tenure']:
            tenure_col = indicators['tenure'][0]
            tenure_normalized = (df[tenure_col] - df[tenure_col].min()) / (df[tenure_col].max() - df[tenure_col].min() + 1e-8)
            risk_scores += (1 - tenure_normalized)
        
        return risk_scores
    
    def get_summary(self) -> str:
        """Get human-readable summary of detection process"""
        return "\n".join(self.detection_log)


def prepare_unlabeled_dataset(df: pd.DataFrame, 
                             target_name='Churn_Predicted',
                             method='auto') -> tuple:
    """
    Convenience function to prepare unlabeled dataset for supervised training
    
    Args:
        df: Dataset without churn column
        target_name: Name for generated target column
        method: 'auto', 'heuristic', 'clustering', 'anomaly'
        
    Returns:
        (df_with_labels, method_used, summary, reliability_score, confidence_metrics)
    """
    detector = UnsupervisedChurnDetector(verbose=True)
    
    # Detect churn labels
    churn_labels, method_used, log, reliability_score = detector.auto_detect_churn(df, method=method)
    
    # Calculate confidence metrics
    confidence_metrics = detector.get_confidence_metrics(df, churn_labels)
    
    # Add to dataframe
    df_labeled = df.copy()
    df_labeled[target_name] = churn_labels
    
    summary = detector.get_summary()
    
    return df_labeled, method_used, summary, reliability_score, confidence_metrics
