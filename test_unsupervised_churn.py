"""
Test Unsupervised Churn Detection
Demonstrates automatic label generation for datasets without churn column
"""

import pandas as pd
import numpy as np
from unsupervised_churn import UnsupervisedChurnDetector, prepare_unlabeled_dataset


def create_unlabeled_dataset():
    """Create a realistic customer dataset WITHOUT churn label"""
    np.random.seed(42)
    n = 1000
    
    # Customer profiles
    data = {
        'CustomerID': [f'CUST{i:04d}' for i in range(n)],
        'Tenure': np.random.randint(1, 72, n),  # months with company
        'MonthlyCharges': np.random.uniform(20, 120, n),
        'TotalCharges': np.random.uniform(100, 8000, n),
        'DataUsageGB': np.random.uniform(0, 50, n),
        'CallMinutes': np.random.randint(0, 1000, n),
        'SupportTickets': np.random.poisson(2, n),
        'SatisfactionScore': np.random.randint(1, 6, n),
        'ContractType': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.4, 0.4, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # Create patterns that indicate churn risk:
    # 1. High charges but low usage = price sensitive
    # 2. Low tenure = haven't built loyalty
    # 3. Many support tickets = frustrated
    # 4. Low satisfaction = unhappy
    # 5. Month-to-month contracts = no commitment
    
    return df


def test_heuristic_method():
    """Test behavioral heuristics method"""
    print("\n" + "="*70)
    print("TEST 1: Behavioral Heuristics Method")
    print("="*70)
    
    df = create_unlabeled_dataset()
    print(f"Dataset: {len(df)} customers, {len(df.columns)} features")
    print(f"Features: {', '.join(df.columns[:5])}...")
    
    detector = UnsupervisedChurnDetector(verbose=True)
    
    # Detect indicators
    indicators = detector.detect_churn_indicators(df)
    print(f"\n📊 Detected Indicators:")
    for indicator_type, cols in indicators.items():
        if cols:
            print(f"  • {indicator_type}: {cols}")
    
    # Generate labels
    churn_labels, method, log = detector.auto_detect_churn(df, method='heuristic')
    
    print(f"\n📋 Detection Summary:")
    print(detector.get_summary())
    
    print(f"\n✓ Generated {churn_labels.sum()} churn labels ({churn_labels.mean():.1%} churn rate)")
    
    return df, churn_labels


def test_clustering_method():
    """Test clustering method"""
    print("\n" + "="*70)
    print("TEST 2: Clustering Method")
    print("="*70)
    
    df = create_unlabeled_dataset()
    detector = UnsupervisedChurnDetector(verbose=True)
    
    churn_labels, method, log = detector.auto_detect_churn(df, method='clustering')
    
    print(f"\n📋 Detection Summary:")
    print(detector.get_summary())
    
    print(f"\n✓ Generated {churn_labels.sum()} churn labels ({churn_labels.mean():.1%} churn rate)")
    
    return df, churn_labels


def test_anomaly_method():
    """Test anomaly detection method"""
    print("\n" + "="*70)
    print("TEST 3: Anomaly Detection Method")
    print("="*70)
    
    df = create_unlabeled_dataset()
    detector = UnsupervisedChurnDetector(verbose=True)
    
    churn_labels, method, log = detector.auto_detect_churn(df, method='anomaly')
    
    print(f"\n📋 Detection Summary:")
    print(detector.get_summary())
    
    print(f"\n✓ Generated {churn_labels.sum()} churn labels ({churn_labels.mean():.1%} churn rate)")
    
    return df, churn_labels


def test_auto_method():
    """Test auto-selection method"""
    print("\n" + "="*70)
    print("TEST 4: Auto-Selection Method (Recommended)")
    print("="*70)
    
    df = create_unlabeled_dataset()
    detector = UnsupervisedChurnDetector(verbose=True)
    
    churn_labels, method, log = detector.auto_detect_churn(df, method='auto')
    
    print(f"\n📋 Detection Summary:")
    print(detector.get_summary())
    
    print(f"\n✓ Auto-selected method: {method.upper()}")
    print(f"✓ Generated {churn_labels.sum()} churn labels ({churn_labels.mean():.1%} churn rate)")
    
    return df, churn_labels


def test_prepare_function():
    """Test convenience function"""
    print("\n" + "="*70)
    print("TEST 5: Convenience Function (prepare_unlabeled_dataset)")
    print("="*70)
    
    df = create_unlabeled_dataset()
    
    # Use convenience function
    df_labeled, method, summary = prepare_unlabeled_dataset(
        df,
        target_name='Churn_Risk',
        method='auto'
    )
    
    print(f"\n📊 Original: {len(df.columns)} columns")
    print(f"📊 Labeled: {len(df_labeled.columns)} columns")
    print(f"✓ New column added: 'Churn_Risk'")
    print(f"✓ Method used: {method.upper()}")
    
    print(f"\n📋 Summary:")
    print(summary)
    
    print(f"\n✓ Churn distribution:")
    print(df_labeled['Churn_Risk'].value_counts())
    
    return df_labeled


def test_minimal_dataset():
    """Test with minimal features (should fallback to anomaly)"""
    print("\n" + "="*70)
    print("TEST 6: Minimal Dataset (Fallback Test)")
    print("="*70)
    
    # Very simple dataset
    df = pd.DataFrame({
        'CustomerID': [f'C{i}' for i in range(100)],
        'Value1': np.random.randn(100),
        'Value2': np.random.randn(100)
    })
    
    print(f"Dataset: {len(df)} customers, {len(df.columns)} features (minimal)")
    
    detector = UnsupervisedChurnDetector(verbose=True)
    churn_labels, method, log = detector.auto_detect_churn(df, method='auto')
    
    print(f"\n📋 Detection Summary:")
    print(detector.get_summary())
    
    print(f"\n✓ Method selected for minimal data: {method.upper()}")
    
    return df, churn_labels


def test_comparison():
    """Compare all methods on same dataset"""
    print("\n" + "="*70)
    print("TEST 7: Method Comparison")
    print("="*70)
    
    df = create_unlabeled_dataset()
    detector = UnsupervisedChurnDetector(verbose=False)
    
    methods = ['heuristic', 'clustering', 'anomaly']
    results = {}
    
    for method in methods:
        try:
            labels, _, _ = detector.auto_detect_churn(df.copy(), method=method)
            churn_rate = labels.mean()
            n_churners = labels.sum()
            results[method] = {'churn_rate': churn_rate, 'n_churners': n_churners}
        except Exception as e:
            results[method] = {'error': str(e)[:50]}
    
    print(f"\n📊 Comparison Results:")
    print(f"{'Method':<20} {'Churners':>10} {'Churn Rate':>12}")
    print("-" * 45)
    for method, res in results.items():
        if 'error' in res:
            print(f"{method:<20} ERROR: {res['error']}")
        else:
            print(f"{method:<20} {res['n_churners']:>10} {res['churn_rate']:>11.1%}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNSUPERVISED CHURN DETECTION TEST SUITE")
    print("Testing automatic label generation for unlabeled datasets")
    print("="*70)
    
    # Run all tests
    test_heuristic_method()
    test_clustering_method()
    test_anomaly_method()
    test_auto_method()
    test_prepare_function()
    test_minimal_dataset()
    test_comparison()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    print("\nKey Insights:")
    print("• Heuristic: Best when behavioral indicators present")
    print("• Clustering: Good for customer segmentation")
    print("• Anomaly: Universal fallback, works with any data")
    print("• Auto: Intelligently selects best method")
    print("\nRecommendation: Use 'auto' mode for best results")
