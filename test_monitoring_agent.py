"""
Test the Monitoring Agent's problem detection and automatic fixing capabilities
"""

from monitoring_agent import MonitoringAgent
import pandas as pd
import numpy as np


def test_data_quality_fix():
    """Test automatic fixing of data quality issues"""
    print("\n" + "="*60)
    print("TEST 1: Data Quality Issues (NaN, Inf)")
    print("="*60)
    
    # Create problematic data
    data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [1.0, np.inf, 3.0, 4.0, 5.0],
        'feature3': ['a', 'b', None, 'd', 'e'],
        'target': [0, 1, 0, 1, 0]
    })
    
    config = {}
    agent = MonitoringAgent(max_retries=2)
    
    # Simulate error
    error = ValueError("Data contains NaN or Inf values")
    problem_type = agent.detect_problem_type(error)
    print(f"✓ Detected: {problem_type}")
    
    # Apply fix
    fixed_data, fixed_config, fix_desc = agent.apply_fix(problem_type, data, config, 1)
    print(f"✓ Applied: {fix_desc}")
    print(f"✓ NaN count before: {data.isnull().sum().sum()}, after: {fixed_data.isnull().sum().sum()}")
    print(f"✓ Inf count before: {np.isinf(data.select_dtypes(include=[np.number]).values).sum()}, after: {np.isinf(fixed_data.select_dtypes(include=[np.number]).values).sum()}")
    

def test_memory_fix():
    """Test automatic fixing of memory issues"""
    print("\n" + "="*60)
    print("TEST 2: Memory Issues")
    print("="*60)
    
    # Create large data
    data = pd.DataFrame({
        f'feature_{i}': np.random.randn(100000) for i in range(20)
    })
    data['target'] = np.random.randint(0, 2, 100000)
    
    config = {
        'polynomial_features': True,
        'n_select': 10,
        'polynomial_degree': 3
    }
    
    agent = MonitoringAgent(max_retries=2)
    
    # Simulate memory error
    error = MemoryError("Cannot allocate memory for polynomial features")
    problem_type = agent.detect_problem_type(error)
    print(f"✓ Detected: {problem_type}")
    
    # Apply fix
    fixed_data, fixed_config, fix_desc = agent.apply_fix(problem_type, data, config, 1)
    print(f"✓ Applied: {fix_desc}")
    print(f"✓ Polynomial features: {config.get('polynomial_features')} → {fixed_config.get('polynomial_features')}")
    print(f"✓ Data shape: {data.shape} → {fixed_data.shape}")


def test_convergence_fix():
    """Test automatic fixing of convergence issues"""
    print("\n" + "="*60)
    print("TEST 3: Convergence Issues")
    print("="*60)
    
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    
    config = {
        'max_iter': 10,
        'n_iter': 5
    }
    
    agent = MonitoringAgent(max_retries=2)
    
    # Simulate convergence error
    error = ValueError("Model failed to converge after 10 iterations")
    problem_type = agent.detect_problem_type(error)
    print(f"✓ Detected: {problem_type}")
    
    # Apply fix
    fixed_data, fixed_config, fix_desc = agent.apply_fix(problem_type, data, config, 1)
    print(f"✓ Applied: {fix_desc}")
    print(f"✓ Max iterations: {config.get('max_iter')} → {fixed_config.get('max_iter')}")
    print(f"✓ Tuning iterations: {config.get('n_iter')} → {fixed_config.get('n_iter')}")


def test_full_workflow():
    """Test complete monitoring workflow with retry"""
    print("\n" + "="*60)
    print("TEST 4: Full Monitoring Workflow (Simulated)")
    print("="*60)
    
    agent = MonitoringAgent(max_retries=3, verbose=True)
    
    attempt_count = [0]  # Use list to modify in nested function
    
    def faulty_training(data, config, target):
        """Simulates a training function that fails twice then succeeds"""
        attempt_count[0] += 1
        
        if attempt_count[0] == 1:
            raise ValueError("Data contains NaN values")
        elif attempt_count[0] == 2:
            raise MemoryError("Cannot allocate memory")
        else:
            return "SUCCESS: Model trained!"
    
    # Create test data
    data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [1.0, np.inf, 3.0, 4.0, 5.0],
        'target': [0, 1, 0, 1, 0]
    })
    
    config = {'polynomial_features': True}
    
    # Run monitored training
    result, fix_log = agent.monitor_and_fix(
        faulty_training,
        data,
        config,
        'target'
    )
    
    print(f"\n✓ Final result: {result}")
    print(f"\n📋 Fix Log:")
    for log in fix_log:
        print(f"  {log}")
    
    print(f"\n📊 Problem Report:")
    report = agent.get_problem_report()
    print(f"  Total problems: {report['total_problems']}")
    print(f"  Success: {report['success']}")


def test_problem_detection():
    """Test problem type detection"""
    print("\n" + "="*60)
    print("TEST 5: Problem Type Detection")
    print("="*60)
    
    agent = MonitoringAgent()
    
    test_errors = [
        (ValueError("Data contains NaN values"), "DATA_QUALITY"),
        (TypeError("Cannot convert string to float"), "DATA_TYPE"),
        (MemoryError("Allocation failed"), "MEMORY"),
        (KeyError("Column 'xyz' not found"), "KEY_ERROR"),
        (ValueError("Model failed to converge"), "CONVERGENCE"),
        (RuntimeError("CUDA out of memory"), "GPU_ISSUE"),
        (ValueError("Feature mismatch"), "FEATURE_MISMATCH"),
    ]
    
    for error, expected in test_errors:
        detected = agent.detect_problem_type(error)
        status = "✓" if detected == expected else "✗"
        print(f"{status} {str(error)[:40]:45} → {detected:20} (expected: {expected})")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MONITORING AGENT TEST SUITE")
    print("="*60)
    
    test_problem_detection()
    test_data_quality_fix()
    test_memory_fix()
    test_convergence_fix()
    test_full_workflow()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
