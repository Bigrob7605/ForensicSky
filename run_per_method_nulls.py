#!/usr/bin/env python3
"""
PER-METHOD NULL CALIBRATION DRIVER
==================================

Runs per-method null calibration by scrambling real data and measuring
each method's null distribution. This is critical for validating that
our 15σ detections aren't due to overconfident error models.

Usage:
    python run_per_method_nulls.py --input 02_Data/real_ipta_dr2 --out runs/nulls_per_method --n-trials 500
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def run_single_trial(trial_num, input_data, output_root, seed):
    """Run a single null trial."""
    trial_dir = Path(output_root) / f"trial_{trial_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scrambled dataset
    scrambled_dir = trial_dir / "scrambled"
    scramble_cmd = [
        "python", "tools/scramble_preserve_intervals.py",
        "--input", input_data,
        "--out", str(scrambled_dir),
        "--seed", str(seed)
    ]
    
    try:
        result = subprocess.run(scramble_cmd, capture_output=True, text=True, check=True)
        print(f"Trial {trial_num}: Scrambling complete")
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial_num}: Scrambling failed: {e}")
        return False
    
    # Run detection pipeline on scrambled data
    detection_cmd = [
        "python", "RUN_MODERN_EXOTIC_HUNTER.py",
        "--input", str(scrambled_dir),
        "--out", str(trial_dir),
        "--seed", str(seed + 10000)
    ]
    
    try:
        result = subprocess.run(detection_cmd, capture_output=True, text=True, check=True)
        print(f"Trial {trial_num}: Detection complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial_num}: Detection failed: {e}")
        return False

def run_per_method_nulls(input_data, output_root, n_trials=500, n_workers=4):
    """Run per-method null calibration."""
    print(f"Running per-method null calibration:")
    print(f"  Input data: {input_data}")
    print(f"  Output root: {output_root}")
    print(f"  Number of trials: {n_trials}")
    print(f"  Workers: {n_workers}")
    print()
    
    # Create output directory
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    # Run trials in parallel
    start_time = time.time()
    completed_trials = 0
    failed_trials = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all trials
        futures = []
        for i in range(1, n_trials + 1):
            seed = 12345 + i * 1000  # Ensure different seeds
            future = executor.submit(run_single_trial, i, input_data, output_root, seed)
            futures.append(future)
        
        # Process completed trials
        for future in as_completed(futures):
            if future.result():
                completed_trials += 1
            else:
                failed_trials += 1
            
            if (completed_trials + failed_trials) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (completed_trials + failed_trials) / elapsed
                print(f"Progress: {completed_trials + failed_trials}/{n_trials} trials, "
                      f"{completed_trials} completed, {failed_trials} failed, "
                      f"{rate:.1f} trials/min")
    
    elapsed = time.time() - start_time
    print(f"\nPer-method null calibration complete!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Completed: {completed_trials}")
    print(f"  Failed: {failed_trials}")
    print(f"  Success rate: {completed_trials/(completed_trials+failed_trials)*100:.1f}%")
    
    # Aggregate results
    print("\nAggregating results...")
    aggregate_cmd = [
        "python", "tools/aggregate_nulls.py",
        "--root", output_root,
        "--out", str(Path(output_root) / "null_stats.csv")
    ]
    
    try:
        result = subprocess.run(aggregate_cmd, capture_output=True, text=True, check=True)
        print("Aggregation complete!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Aggregation failed: {e}")
        print(e.stderr)

def main():
    parser = argparse.ArgumentParser(description="Run per-method null calibration")
    parser.add_argument("--input", required=True, help="Input real dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for null trials")
    parser.add_argument("--n-trials", type=int, default=500, help="Number of null trials")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        sys.exit(1)
    
    run_per_method_nulls(args.input, args.out, args.n_trials, args.n_workers)

if __name__ == "__main__":
    main()
