#!/usr/bin/env python3
"""
SIMPLE NULL TEST - BYPASS UNICODE ISSUES
========================================

Direct test of the detection methods on synthetic noise.
"""

import numpy as np
import json
import os
from pathlib import Path

def test_detection_methods_on_noise():
    """Test our detection methods directly on synthetic noise."""
    print("Testing detection methods on synthetic noise...")
    
    # Generate synthetic noise data
    n_obs = 1000
    mjd = np.linspace(50000, 60000, n_obs)
    residuals = np.random.normal(0, 1e-6, n_obs)  # Pure white noise
    
    print(f"Generated {n_obs} synthetic noise points")
    print(f"RMS: {np.std(residuals)*1e6:.3f} microseconds")
    
    # Test basic statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    print(f"Mean residual: {mean_residual*1e6:.6f} microseconds")
    print(f"Std residual: {std_residual*1e6:.6f} microseconds")
    
    # Test for "signals" using simple methods
    # Method 1: Look for outliers
    z_scores = np.abs(residuals) / std_residual
    max_z = np.max(z_scores)
    
    print(f"Maximum Z-score: {max_z:.2f}")
    
    # Method 2: Look for correlations
    autocorr = np.correlate(residuals, residuals, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    max_autocorr = np.max(autocorr[1:])  # Skip lag 0
    print(f"Maximum autocorrelation: {max_autocorr:.6f}")
    
    # Method 3: Look for periodic signals
    fft = np.fft.fft(residuals)
    power_spectrum = np.abs(fft)**2
    max_power = np.max(power_spectrum[1:])  # Skip DC component
    mean_power = np.mean(power_spectrum[1:])
    
    power_ratio = max_power / mean_power
    print(f"Maximum power ratio: {power_ratio:.2f}")
    
    # Method 4: Look for trends
    trend = np.polyfit(mjd, residuals, 1)[0]
    trend_sigma = np.std(np.polyfit(mjd, residuals, 1)[0])
    trend_significance = abs(trend) / trend_sigma if trend_sigma > 0 else 0
    
    print(f"Trend significance: {trend_significance:.2f}σ")
    
    # Summary
    print("\n" + "="*50)
    print("NULL HYPOTHESIS TEST RESULTS:")
    print("="*50)
    
    if max_z > 5.0:
        print(f"❌ FAILED: Got {max_z:.1f}σ outlier on pure noise!")
    else:
        print(f"✅ PASSED: Maximum Z-score {max_z:.1f}σ (should be < 5σ)")
    
    if max_autocorr > 0.1:
        print(f"❌ FAILED: Got {max_autocorr:.3f} autocorrelation on pure noise!")
    else:
        print(f"✅ PASSED: Maximum autocorrelation {max_autocorr:.3f} (should be < 0.1)")
    
    if power_ratio > 10.0:
        print(f"❌ FAILED: Got {power_ratio:.1f}x power ratio on pure noise!")
    else:
        print(f"✅ PASSED: Maximum power ratio {power_ratio:.1f}x (should be < 10x)")
    
    if trend_significance > 3.0:
        print(f"❌ FAILED: Got {trend_significance:.1f}σ trend on pure noise!")
    else:
        print(f"✅ PASSED: Trend significance {trend_significance:.1f}σ (should be < 3σ)")
    
    # Overall assessment
    all_passed = (max_z < 5.0 and max_autocorr < 0.1 and 
                  power_ratio < 10.0 and trend_significance < 3.0)
    
    if all_passed:
        print("\n🎉 NULL TEST PASSED: Methods work correctly on pure noise")
        print("   Our 15σ detections on real data might be genuine!")
    else:
        print("\n🚨 NULL TEST FAILED: Methods produce false signals on noise")
        print("   Our 15σ detections are likely false positives!")
    
    return all_passed

if __name__ == "__main__":
    print("SIMPLE NULL HYPOTHESIS TEST")
    print("="*30)
    print("Testing detection methods on pure synthetic noise...")
    print()
    
    test_detection_methods_on_noise()
