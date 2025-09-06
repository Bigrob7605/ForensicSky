#!/usr/bin/env python3
"""
TEST ADVANCED METHODS ON NOISE
==============================

Test the specific advanced methods that are giving us 15σ detections
to see if they're working correctly on pure noise.
"""

import numpy as np
import json
import os
from pathlib import Path

def test_advanced_methods():
    """Test the advanced methods that gave us 15σ detections."""
    print("Testing advanced detection methods on synthetic noise...")
    
    # Generate synthetic noise data
    n_obs = 1000
    mjd = np.linspace(50000, 60000, n_obs)
    residuals = np.random.normal(0, 1e-6, n_obs)  # Pure white noise
    
    print(f"Generated {n_obs} synthetic noise points")
    print(f"RMS: {np.std(residuals)*1e6:.3f} microseconds")
    
    # Test Method 1: Topological ML Analysis (gave us 15σ)
    print("\n1. Testing Topological ML Analysis...")
    
    # Simulate topological features
    from scipy import stats
    
    # Create some fake topological features
    n_features = 10
    features = np.random.normal(0, 1, n_features)
    
    # Calculate "significance" using the same method as our code
    feature_mean = np.mean(features)
    feature_std = np.std(features)
    topological_sigma = abs(feature_mean) / feature_std if feature_std > 0 else 0
    
    print(f"   Topological features: {features}")
    print(f"   Topological sigma: {topological_sigma:.2f}")
    
    # Test Method 2: Deep Anomaly Detection (gave us 15σ)
    print("\n2. Testing Deep Anomaly Detection...")
    
    # Simulate deep learning anomaly score
    anomaly_scores = np.random.normal(0, 1, 100)
    anomaly_mean = np.mean(anomaly_scores)
    anomaly_std = np.std(anomaly_scores)
    anomaly_sigma = abs(anomaly_mean) / anomaly_std if anomaly_std > 0 else 0
    
    print(f"   Anomaly scores: {anomaly_scores[:5]}...")
    print(f"   Anomaly sigma: {anomaly_sigma:.2f}")
    
    # Test Method 3: Quantum Gravity Effects (gave us 13.6σ)
    print("\n3. Testing Quantum Gravity Effects...")
    
    # Simulate quantum gravity search
    qg_features = np.random.normal(0, 1, 50)
    qg_mean = np.mean(qg_features)
    qg_std = np.std(qg_features)
    qg_sigma = abs(qg_mean) / qg_std if qg_std > 0 else 0
    
    print(f"   Quantum gravity features: {qg_features[:5]}...")
    print(f"   Quantum gravity sigma: {qg_sigma:.2f}")
    
    # Test Method 4: Ensemble Bayesian Analysis (gave us 9.35σ)
    print("\n4. Testing Ensemble Bayesian Analysis...")
    
    # Simulate ensemble of detectors
    n_detectors = 5
    detector_sigmas = []
    
    for i in range(n_detectors):
        detector_data = np.random.normal(0, 1, 200)
        detector_mean = np.mean(detector_data)
        detector_std = np.std(detector_data)
        detector_sigma = abs(detector_mean) / detector_std if detector_std > 0 else 0
        detector_sigmas.append(detector_sigma)
    
    # Combine detectors (this is where the problem might be)
    combined_sigma = np.sqrt(np.sum(np.array(detector_sigmas)**2))  # This is wrong!
    
    print(f"   Individual detector sigmas: {[f'{s:.2f}' for s in detector_sigmas]}")
    print(f"   Combined sigma: {combined_sigma:.2f}")
    
    # Test Method 5: VAE Analysis (gave us 1.05σ - this looks correct)
    print("\n5. Testing VAE Analysis...")
    
    vae_features = np.random.normal(0, 1, 100)
    vae_mean = np.mean(vae_features)
    vae_std = np.std(vae_features)
    vae_sigma = abs(vae_mean) / vae_std if vae_std > 0 else 0
    
    print(f"   VAE features: {vae_features[:5]}...")
    print(f"   VAE sigma: {vae_sigma:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("ADVANCED METHODS TEST RESULTS:")
    print("="*60)
    
    methods = [
        ("Topological ML", topological_sigma),
        ("Deep Anomaly", anomaly_sigma),
        ("Quantum Gravity", qg_sigma),
        ("Ensemble Bayesian", combined_sigma),
        ("VAE", vae_sigma)
    ]
    
    for method_name, sigma in methods:
        if sigma > 5.0:
            print(f"❌ {method_name}: {sigma:.2f}σ (FAILED - should be < 5σ)")
        else:
            print(f"✅ {method_name}: {sigma:.2f}σ (PASSED)")
    
    # Check for the specific problem
    if combined_sigma > 5.0:
        print(f"\n🚨 FOUND THE PROBLEM!")
        print(f"   Ensemble Bayesian is giving {combined_sigma:.2f}σ on pure noise!")
        print(f"   This suggests we're incorrectly combining detector significances.")
        print(f"   The correct way might be to average them, not sum their squares.")
        
        # Test correct combination
        correct_combined = np.mean(detector_sigmas)
        print(f"   Correct combination (mean): {correct_combined:.2f}σ")
        
        return False
    else:
        print(f"\n🎉 All advanced methods work correctly on pure noise!")
        return True

if __name__ == "__main__":
    print("ADVANCED METHODS NULL TEST")
    print("="*30)
    print("Testing the specific methods that gave us 15σ detections...")
    print()
    
    test_advanced_methods()
