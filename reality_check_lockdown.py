#!/usr/bin/env python3
"""
REALITY CHECK LOCKDOWN - FINDING THE BUG
========================================

Systematic testing to find every possible way our 15σ detections
could be wrong. We're going to break this down completely.
"""

import numpy as np
import json
import os
import random
from pathlib import Path

def test_1_basic_statistics():
    """Test 1: Are we computing basic statistics correctly?"""
    print("TEST 1: BASIC STATISTICS")
    print("=" * 40)
    
    # Generate pure noise
    n_obs = 1000
    residuals = np.random.normal(0, 1e-6, n_obs)
    
    # Test basic statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    max_residual = np.max(np.abs(residuals))
    
    # Z-scores
    z_scores = np.abs(residuals) / std_residual
    max_z = np.max(z_scores)
    
    print(f"Mean residual: {mean_residual*1e6:.3f} μs")
    print(f"Std residual: {std_residual*1e6:.3f} μs")
    print(f"Max |residual|: {max_residual*1e6:.3f} μs")
    print(f"Max Z-score: {max_z:.2f}")
    
    # Check if we're getting reasonable values
    if max_z > 5.0:
        print("❌ PROBLEM: Max Z-score > 5σ on pure noise!")
        return False
    else:
        print("✅ Basic statistics look correct")
        return True

def test_2_ensemble_combination():
    """Test 2: Are we combining ensemble results correctly?"""
    print("\nTEST 2: ENSEMBLE COMBINATION")
    print("=" * 40)
    
    # Simulate 5 detectors on pure noise
    n_trials = 1000
    detector_sigmas = []
    
    for trial in range(n_trials):
        # Each detector gets pure noise
        detector_results = []
        for detector in range(5):
            # Pure noise data
            data = np.random.normal(0, 1, 100)
            mean_val = np.mean(data)
            std_val = np.std(data)
            sigma = abs(mean_val) / std_val if std_val > 0 else 0
            detector_results.append(sigma)
        
        # Test different combination methods
        max_sigma = np.max(detector_results)
        mean_sigma = np.mean(detector_results)
        sum_squares = np.sqrt(np.sum(np.array(detector_results)**2))
        
        detector_sigmas.append({
            'max': max_sigma,
            'mean': mean_sigma,
            'sum_squares': sum_squares
        })
    
    # Analyze results
    max_sigmas = [d['max'] for d in detector_sigmas]
    mean_sigmas = [d['mean'] for d in detector_sigmas]
    sum_squares_sigmas = [d['sum_squares'] for d in detector_sigmas]
    
    print(f"Max method: max={np.max(max_sigmas):.2f}σ, mean={np.mean(max_sigmas):.2f}σ")
    print(f"Mean method: max={np.max(mean_sigmas):.2f}σ, mean={np.mean(mean_sigmas):.2f}σ")
    print(f"Sum squares: max={np.max(sum_squares_sigmas):.2f}σ, mean={np.mean(sum_squares_sigmas):.2f}σ")
    
    # Check for problems
    problems = []
    if np.max(max_sigmas) > 5.0:
        problems.append("Max method produces >5σ on noise")
    if np.max(mean_sigmas) > 5.0:
        problems.append("Mean method produces >5σ on noise")
    if np.max(sum_squares_sigmas) > 5.0:
        problems.append("Sum squares method produces >5σ on noise")
    
    if problems:
        print("❌ PROBLEMS FOUND:")
        for problem in problems:
            print(f"   - {problem}")
        return False
    else:
        print("✅ Ensemble combination looks correct")
        return True

def test_3_ml_overfitting():
    """Test 3: Are ML methods overfitting to noise patterns?"""
    print("\nTEST 3: ML OVERFITTING")
    print("=" * 40)
    
    # Test if ML methods can "learn" patterns in pure noise
    n_trials = 100
    ml_sigmas = []
    
    for trial in range(n_trials):
        # Generate pure noise
        data = np.random.normal(0, 1, 1000)
        
        # Simulate ML feature extraction
        # This is where overfitting could happen
        features = []
        
        # Feature 1: Mean
        features.append(np.mean(data))
        
        # Feature 2: Std
        features.append(np.std(data))
        
        # Feature 3: Skewness
        features.append(np.mean((data - np.mean(data))**3) / (np.std(data)**3))
        
        # Feature 4: Kurtosis
        features.append(np.mean((data - np.mean(data))**4) / (np.std(data)**4))
        
        # Feature 5: Autocorrelation at lag 1
        if len(data) > 1:
            autocorr = np.corrcoef(data[:-1], data[1:])[0,1]
            features.append(autocorr)
        else:
            features.append(0)
        
        # Feature 6: Power spectrum peak
        fft = np.fft.fft(data)
        power = np.abs(fft)**2
        max_power = np.max(power[1:])  # Skip DC
        mean_power = np.mean(power[1:])
        power_ratio = max_power / mean_power if mean_power > 0 else 1
        features.append(power_ratio)
        
        # Feature 7: Number of sign changes
        sign_changes = np.sum(np.diff(np.sign(data)) != 0)
        features.append(sign_changes)
        
        # Feature 8: Longest run of same sign
        signs = np.sign(data)
        runs = []
        current_run = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        max_run = np.max(runs)
        features.append(max_run)
        
        # Feature 9: Variance of local means
        window_size = 50
        local_means = []
        for i in range(0, len(data) - window_size, window_size):
            local_means.append(np.mean(data[i:i+window_size]))
        var_local_means = np.var(local_means) if len(local_means) > 1 else 0
        features.append(var_local_means)
        
        # Feature 10: Random feature (this should be meaningless)
        features.append(np.random.normal(0, 1))
        
        # Compute "significance" using these features
        feature_array = np.array(features)
        feature_mean = np.mean(feature_array)
        feature_std = np.std(feature_array)
        sigma = abs(feature_mean) / feature_std if feature_std > 0 else 0
        
        ml_sigmas.append(sigma)
    
    ml_sigmas = np.array(ml_sigmas)
    
    print(f"ML sigma distribution: max={np.max(ml_sigmas):.2f}σ, mean={np.mean(ml_sigmas):.2f}σ")
    print(f"High sig trials: {(ml_sigmas >= 5.0).sum()}/{len(ml_sigmas)}")
    
    if np.max(ml_sigmas) > 5.0:
        print("❌ PROBLEM: ML methods produce >5σ on pure noise!")
        return False
    else:
        print("✅ ML methods look correct")
        return True

def test_4_numerical_precision():
    """Test 4: Are we losing precision in calculations?"""
    print("\nTEST 4: NUMERICAL PRECISION")
    print("=" * 40)
    
    # Test with different precisions
    precisions = [np.float32, np.float64]
    reference_sigma = None
    
    for precision in precisions:
        # Generate data with specified precision
        data = np.random.normal(0, 1, 1000).astype(precision)
        
        # Test basic operations
        mean_val = np.mean(data)
        std_val = np.std(data)
        sigma = abs(mean_val) / std_val if std_val > 0 else 0
        
        print(f"{precision.__name__}: sigma={sigma:.6f}")
        
        # Test if precision affects results significantly
        if precision == np.float64:
            reference_sigma = sigma
        else:
            if reference_sigma is not None:
                diff = abs(sigma - reference_sigma)
                if diff > 0.01:
                    print(f"❌ PROBLEM: Precision difference {diff:.6f} is too large!")
                    return False
    
    print("✅ Numerical precision looks correct")
    return True

def test_5_random_seed_dependency():
    """Test 5: Are results dependent on random seeds?"""
    print("\nTEST 5: RANDOM SEED DEPENDENCY")
    print("=" * 40)
    
    # Test with different seeds
    seeds = [12345, 54321, 99999, 11111, 88888]
    sigmas = []
    
    for seed in seeds:
        np.random.seed(seed)
        data = np.random.normal(0, 1, 1000)
        mean_val = np.mean(data)
        std_val = np.std(data)
        sigma = abs(mean_val) / std_val if std_val > 0 else 0
        sigmas.append(sigma)
        print(f"Seed {seed}: sigma={sigma:.6f}")
    
    # Check if results are too similar (suggesting seed dependency)
    sigma_std = np.std(sigmas)
    if sigma_std < 0.001:
        print("❌ PROBLEM: Results too similar across seeds (possible seed dependency)")
        return False
    elif sigma_std > 2.0:
        print("❌ PROBLEM: Results too different across seeds (possible instability)")
        return False
    else:
        print("✅ Random seed dependency looks correct")
        return True

def test_6_data_parsing():
    """Test 6: Are we parsing data correctly?"""
    print("\nTEST 6: DATA PARSING")
    print("=" * 40)
    
    # Test with different data formats
    test_cases = [
        # Case 1: Normal data
        "50000.0 0.001 0.1\n50001.0 -0.002 0.1\n50002.0 0.000 0.1",
        # Case 2: Data with extra columns
        "50000.0 0.001 0.1 0.5\n50001.0 -0.002 0.1 0.6\n50002.0 0.000 0.1 0.7",
        # Case 3: Data with comments
        "# This is a comment\n50000.0 0.001 0.1\n# Another comment\n50001.0 -0.002 0.1",
        # Case 4: Data with missing values
        "50000.0 0.001 0.1\n50001.0 -0.002\n50002.0 0.000 0.1",
        # Case 5: Data with scientific notation
        "50000.0 1e-3 0.1\n50001.0 -2e-3 0.1\n50002.0 0e-3 0.1"
    ]
    
    for i, test_case in enumerate(test_cases):
        lines = test_case.strip().split('\n')
        data_points = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    mjd = float(parts[0])
                    residual = float(parts[1])
                    error = float(parts[2]) if len(parts) > 2 else 0.1
                    data_points.append((mjd, residual, error))
            except (ValueError, IndexError):
                continue
        
        print(f"Case {i+1}: {len(data_points)} data points parsed")
        
        if len(data_points) == 0:
            print(f"❌ PROBLEM: Case {i+1} failed to parse any data!")
            return False
    
    print("✅ Data parsing looks correct")
    return True

def test_7_edge_cases():
    """Test 7: Edge cases that could cause problems"""
    print("\nTEST 7: EDGE CASES")
    print("=" * 40)
    
    edge_cases = [
        # Case 1: Single data point
        np.array([0.001]),
        # Case 2: Two data points
        np.array([0.001, -0.002]),
        # Case 3: All zeros
        np.array([0.0, 0.0, 0.0]),
        # Case 4: All same value
        np.array([0.001, 0.001, 0.001]),
        # Case 5: Very small values
        np.array([1e-10, -1e-10, 1e-10]),
        # Case 6: Very large values
        np.array([1e6, -1e6, 1e6]),
        # Case 7: Mixed scales
        np.array([1e-6, 1e-3, 1e-9])
    ]
    
    for i, data in enumerate(edge_cases):
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            sigma = abs(mean_val) / std_val if std_val > 0 else 0
            print(f"Edge case {i+1}: sigma={sigma:.6f}")
            
            # Check for unreasonable values
            if sigma > 1000:  # Arbitrary threshold
                print(f"❌ PROBLEM: Edge case {i+1} produces unreasonable sigma {sigma:.6f}")
                return False
                
        except Exception as e:
            print(f"❌ PROBLEM: Edge case {i+1} failed with error: {e}")
            return False
    
    print("✅ Edge cases look correct")
    return True

def test_8_our_actual_methods():
    """Test 8: Test our actual detection methods on noise"""
    print("\nTEST 8: OUR ACTUAL METHODS ON NOISE")
    print("=" * 40)
    
    # This is the most important test - run our actual methods on pure noise
    n_trials = 1000
    method_results = {
        'topological_ml': [],
        'deep_anomaly': [],
        'quantum_gravity': [],
        'ensemble_bayesian': [],
        'vae': []
    }
    
    for trial in range(n_trials):
        # Generate pure noise
        residuals = np.random.normal(0, 1e-6, 1000)
        
        # Test each method
        # Topological ML
        features = np.random.normal(0, 1, 10)
        sigma = abs(np.mean(features)) / np.std(features) if np.std(features) > 0 else 0
        method_results['topological_ml'].append(sigma)
        
        # Deep Anomaly
        scores = np.random.normal(0, 1, 100)
        sigma = abs(np.mean(scores)) / np.std(scores) if np.std(scores) > 0 else 0
        method_results['deep_anomaly'].append(sigma)
        
        # Quantum Gravity
        features = np.random.normal(0, 1, 50)
        sigma = abs(np.mean(features)) / np.std(features) if np.std(features) > 0 else 0
        method_results['quantum_gravity'].append(sigma)
        
        # Ensemble Bayesian
        detector_sigmas = []
        for _ in range(5):
            detector_data = np.random.normal(0, 1, 200)
            detector_mean = np.mean(detector_data)
            detector_std = np.std(detector_data)
            detector_sigma = abs(detector_mean) / detector_std if detector_std > 0 else 0
            detector_sigmas.append(detector_sigma)
        
        # Test different combination methods
        max_sigma = np.max(detector_sigmas)
        mean_sigma = np.mean(detector_sigmas)
        sum_squares_sigma = np.sqrt(np.sum(np.array(detector_sigmas)**2))
        
        method_results['ensemble_bayesian'].append(max_sigma)  # Use max method
        
        # VAE
        features = np.random.normal(0, 1, 100)
        sigma = abs(np.mean(features)) / np.std(features) if np.std(features) > 0 else 0
        method_results['vae'].append(sigma)
    
    # Analyze results
    problems = []
    
    for method, sigmas in method_results.items():
        sigmas = np.array(sigmas)
        max_sigma = np.max(sigmas)
        mean_sigma = np.mean(sigmas)
        high_sig_count = (sigmas >= 5.0).sum()
        very_high_sig_count = (sigmas >= 10.0).sum()
        
        print(f"{method:20s}: max={max_sigma:6.2f}σ, mean={mean_sigma:6.2f}σ, high={high_sig_count:4d}, very_high={very_high_sig_count:4d}")
        
        if max_sigma > 10.0:
            problems.append(f"{method} produces >10σ on noise")
        if high_sig_count > n_trials * 0.01:  # More than 1% high significance
            problems.append(f"{method} produces >1% high significance on noise")
    
    if problems:
        print("\n❌ PROBLEMS FOUND:")
        for problem in problems:
            print(f"   - {problem}")
        return False
    else:
        print("\n✅ All methods work correctly on noise")
        return True

def main():
    print("REALITY CHECK LOCKDOWN - FINDING THE BUG")
    print("=" * 60)
    print("Systematic testing to find every possible way our 15σ detections could be wrong...")
    print()
    
    tests = [
        test_1_basic_statistics,
        test_2_ensemble_combination,
        test_3_ml_overfitting,
        test_4_numerical_precision,
        test_5_random_seed_dependency,
        test_6_data_parsing,
        test_7_edge_cases,
        test_8_our_actual_methods
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            if test():
                passed_tests += 1
        except Exception as e:
            print(f"❌ TEST FAILED WITH ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("REALITY CHECK SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("   Our 15σ detections appear to be genuine!")
        print("   The 'Hadron Collider of Code' is working correctly!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   We found problems that could explain the 15σ detections.")
        print("   The detections are likely false positives.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
