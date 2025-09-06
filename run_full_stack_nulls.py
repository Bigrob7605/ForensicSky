#!/usr/bin/env python3
"""
FULL-STACK NULL MONTE CARLO DRIVER
=================================

Runs full-stack null Monte Carlo to compute empirical global p-values
for the combined detection statistic. This is the most critical test
for validating our 15σ detections.

Usage:
    python run_full_stack_nulls.py --input 02_Data/real_ipta_dr2 --out runs/full_null_mc --n-trials 1000
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

def run_full_stack_nulls(input_data, output_root, n_trials=1000, n_workers=4):
    """Run full-stack null Monte Carlo."""
    print(f"Running full-stack null Monte Carlo:")
    print(f"  Input data: {input_data}")
    print(f"  Output root: {output_root}")
    print(f"  Number of trials: {n_trials}")
    print(f"  Workers: {n_workers}")
    print()
    
    # Create output directory
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    # First, get the observed values from real data
    print("Getting observed values from real data...")
    observed_cmd = [
        "python", "RUN_MODERN_EXOTIC_HUNTER.py",
        "--input", input_data,
        "--out", str(Path(output_root) / "observed"),
        "--seed", "999999"
    ]
    
    try:
        result = subprocess.run(observed_cmd, capture_output=True, text=True, check=True)
        print("Observed values obtained")
    except subprocess.CalledProcessError as e:
        print(f"Failed to get observed values: {e}")
        return
    
    # Load observed values
    observed_report = None
    for report_file in Path(output_root).glob("observed/**/modern_exotic_physics_report_*.json"):
        with open(report_file, 'r') as f:
            observed_report = json.load(f)
        break
    
    if observed_report is None:
        print("Error: Could not find observed report")
        return
    
    observed_max = observed_report.get('summary', {}).get('max_significance', 0.0)
    print(f"Observed maximum significance: {observed_max:.2f}σ")
    
    # Run null trials
    print(f"\nRunning {n_trials} null trials...")
    start_time = time.time()
    null_maxima = []
    completed_trials = 0
    failed_trials = 0
    
    for i in range(n_trials):
        trial_dir = Path(output_root) / f"trial_{i:04d}"
        trial_dir.mkdir(exist_ok=True)
        
        # Create scrambled dataset
        scrambled_dir = trial_dir / "scrambled"
        scramble_cmd = [
            "python", "tools/scramble_preserve_intervals.py",
            "--input", input_data,
            "--out", str(scrambled_dir),
            "--seed", str(12345 + i * 1000)
        ]
        
        try:
            subprocess.run(scramble_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Trial {i}: Scrambling failed")
            failed_trials += 1
            continue
        
        # Run detection pipeline
        detection_cmd = [
            "python", "RUN_MODERN_EXOTIC_HUNTER.py",
            "--input", str(scrambled_dir),
            "--out", str(trial_dir),
            "--seed", str(12345 + i * 1000 + 10000)
        ]
        
        try:
            subprocess.run(detection_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Trial {i}: Detection failed")
            failed_trials += 1
            continue
        
        # Extract maximum significance
        trial_max = 0.0
        for report_file in trial_dir.glob("**/modern_exotic_physics_report_*.json"):
            with open(report_file, 'r') as f:
                report = json.load(f)
                trial_max = max(trial_max, report.get('summary', {}).get('max_significance', 0.0))
                break
        
        null_maxima.append(trial_max)
        completed_trials += 1
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"Progress: {i + 1}/{n_trials} trials, "
                  f"{completed_trials} completed, {failed_trials} failed, "
                  f"{rate:.1f} trials/min")
    
    elapsed = time.time() - start_time
    print(f"\nFull-stack null MC complete!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Completed: {completed_trials}")
    print(f"  Failed: {failed_trials}")
    print(f"  Success rate: {completed_trials/(completed_trials+failed_trials)*100:.1f}%")
    
    # Analyze results
    if null_maxima:
        null_maxima = np.array(null_maxima)
        
        # Compute empirical global p-value
        p_value = (null_maxima >= observed_max).sum() / len(null_maxima)
        
        # Compute statistics
        max_null = np.max(null_maxima)
        mean_null = np.mean(null_maxima)
        std_null = np.std(null_maxima)
        
        # Find quantiles
        quantiles = np.percentile(null_maxima, [50, 90, 95, 99, 99.9, 99.99])
        
        print(f"\n" + "="*60)
        print("FULL-STACK NULL MC RESULTS")
        print("="*60)
        print(f"Observed maximum significance: {observed_max:.2f}σ")
        print(f"Null distribution statistics:")
        print(f"  Maximum: {max_null:.2f}σ")
        print(f"  Mean: {mean_null:.2f}σ")
        print(f"  Std: {std_null:.2f}σ")
        print(f"  Quantiles:")
        print(f"    50%: {quantiles[0]:.2f}σ")
        print(f"    90%: {quantiles[1]:.2f}σ")
        print(f"    95%: {quantiles[2]:.2f}σ")
        print(f"    99%: {quantiles[3]:.2f}σ")
        print(f"    99.9%: {quantiles[4]:.2f}σ")
        print(f"    99.99%: {quantiles[5]:.2f}σ")
        print(f"\nEmpirical global p-value: {p_value:.2e}")
        
        # Decision
        if p_value < 1e-6:
            print(f"🎉 EXCELLENT: p < 1e-6 - detection is highly significant!")
        elif p_value < 1e-4:
            print(f"✅ GOOD: p < 1e-4 - detection is significant")
        elif p_value < 1e-3:
            print(f"⚠️  MARGINAL: p < 1e-3 - detection is marginal")
        else:
            print(f"❌ FAILED: p >= 1e-3 - detection is not significant")
        
        # Save results
        results = {
            'observed_max': observed_max,
            'null_maxima': null_maxima.tolist(),
            'p_value': p_value,
            'max_null': max_null,
            'mean_null': mean_null,
            'std_null': std_null,
            'quantiles': quantiles.tolist(),
            'n_trials': len(null_maxima)
        }
        
        with open(Path(output_root) / "full_stack_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_root}/full_stack_results.json")
        
    else:
        print("No valid null trials completed!")

def main():
    parser = argparse.ArgumentParser(description="Run full-stack null Monte Carlo")
    parser.add_argument("--input", required=True, help="Input real dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for null trials")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of null trials")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        sys.exit(1)
    
    run_full_stack_nulls(args.input, args.out, args.n_trials, args.n_workers)

if __name__ == "__main__":
    main()
