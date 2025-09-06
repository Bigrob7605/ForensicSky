#!/usr/bin/env python3
"""
NULL HYPOTHESIS TEST - CRITICAL VALIDATION
==========================================

Tests the cosmic string detection platform on pure noise to verify
it doesn't produce false 15σ detections on scrambled data.

This is the critical test to determine if our 15σ detections are real
or if our error model is overconfident.
"""

import numpy as np
import json
import os
import subprocess
import sys
from pathlib import Path

def create_synthetic_noise_dataset():
    """Create a synthetic noise dataset with the same structure as real data."""
    print("🔬 Creating synthetic noise dataset...")
    
    # Create synthetic data directory
    noise_dir = Path("02_Data/synthetic_noise")
    noise_dir.mkdir(exist_ok=True)
    
    # Generate synthetic noise for each pulsar
    pulsars = [
        "J1909-3744", "J1713+0747", "J1744-1134", "J1600-3053", "J0437-4715",
        "J1741+1351", "J1857+0943", "J1939+2134", "J1955+2908", "J2010-1323"
    ]
    
    for pulsar in pulsars:
        pulsar_dir = noise_dir / pulsar
        pulsar_dir.mkdir(exist_ok=True)
        
        # Generate synthetic timing residuals (pure white noise)
        n_obs = 100
        mjd = np.linspace(50000, 60000, n_obs)
        residuals = np.random.normal(0, 1e-6, n_obs)  # 1 microsecond RMS
        
        # Write synthetic data file
        with open(pulsar_dir / "synthetic_noise.txt", "w") as f:
            f.write("# Synthetic noise data for null hypothesis testing\n")
            f.write("# MJD Residual(us) Error(us)\n")
            for mjd_val, res_val in zip(mjd, residuals):
                f.write(f"{mjd_val:.3f} {res_val*1e6:.6f} 0.1\n")
    
    print(f"✅ Created synthetic noise dataset with {len(pulsars)} pulsars")
    return noise_dir

def run_null_test():
    """Run the detection platform on synthetic noise."""
    print("🧪 Running null hypothesis test...")
    
    # Create synthetic noise
    noise_dir = create_synthetic_noise_dataset()
    
    # Run detection platform on noise
    cmd = [
        "python", "RUN_MODERN_EXOTIC_HUNTER.py",
        "--input", str(noise_dir),
        "--out", "runs/null_test"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    
    return result.returncode == 0

def analyze_results():
    """Analyze the results of the null test."""
    print("📊 Analyzing null test results...")
    
    # Look for results file
    results_file = Path("runs/null_test/modern_exotic_physics_report_*.json")
    results_files = list(Path("runs/null_test").glob("modern_exotic_physics_report_*.json"))
    
    if not results_files:
        print("❌ No results file found")
        return False
    
    results_file = results_files[0]
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"📄 Results file: {results_file}")
    print(f"📊 Maximum significance: {results.get('max_significance', 'N/A')}")
    print(f"📊 Total detections: {results.get('total_detections', 'N/A')}")
    
    # Check if we got false detections
    max_sig = results.get('max_significance', 0)
    if max_sig > 5.0:
        print(f"🚨 WARNING: Got {max_sig}σ on pure noise! Error model is overconfident.")
        return False
    else:
        print(f"✅ Good: Maximum significance {max_sig}σ on noise (should be < 5σ)")
        return True

if __name__ == "__main__":
    print("🔬 NULL HYPOTHESIS TEST - CRITICAL VALIDATION")
    print("=" * 50)
    print("Testing if our 15σ detections are real or false positives...")
    print()
    
    success = run_null_test()
    
    if success:
        analyze_results()
    else:
        print("❌ Null test failed")
    
    print("\n" + "=" * 50)
    print("Null hypothesis test complete!")
