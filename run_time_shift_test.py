#!/usr/bin/env python3
"""
TIME-SHIFT TEST DRIVER
======================

Tests if our detections survive time-shifting each pulsar by different
offsets. This breaks cross-pulsar correlations while preserving
intra-pulsar structure.

Usage:
    python run_time_shift_test.py --input 02_Data/real_ipta_dr2 --out runs/time_shift_test --n-trials 200
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

def create_time_shifted_dataset(input_dir, output_dir, trial_num):
    """Create a time-shifted version of the dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pulsar directories
    pulsar_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    for pulsar_dir in pulsar_dirs:
        pulsar_name = pulsar_dir.name
        output_pulsar_dir = output_path / pulsar_name
        output_pulsar_dir.mkdir(exist_ok=True)
        
        # Find data files in this pulsar directory
        data_files = list(pulsar_dir.glob("*.txt")) + list(pulsar_dir.glob("*.dat"))
        
        for data_file in data_files:
            output_file = output_pulsar_dir / data_file.name
            
            # Read the input file
            with open(data_file, 'r') as f:
                lines = f.readlines()
            
            # Parse data lines
            data_lines = []
            header_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line == '' or 'MJD' in line or 'Residual' in line:
                    header_lines.append(line)
                else:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            mjd = float(parts[0])
                            residual = float(parts[1])
                            error = float(parts[2]) if len(parts) > 2 else 0.1
                            data_lines.append((mjd, residual, error))
                    except (ValueError, IndexError):
                        header_lines.append(line)
            
            if not data_lines:
                continue
            
            # Apply time shift (different for each pulsar)
            np.random.seed(trial_num * 1000 + hash(pulsar_name) % 1000)
            time_shift = np.random.uniform(-1000, 1000)  # days
            
            # Write time-shifted data
            with open(output_file, 'w') as f:
                # Write headers
                for header in header_lines:
                    f.write(header + '\n')
                
                # Write time-shifted data
                for mjd, residual, error in data_lines:
                    shifted_mjd = mjd + time_shift
                    f.write(f"{shifted_mjd:.6f} {residual:.6f} {error:.6f}\n")
    
    # Copy any other files
    for item in input_path.iterdir():
        if item.is_file() and not item.name.endswith(('.txt', '.dat')):
            import shutil
            shutil.copy2(item, output_path / item.name)

def run_time_shift_test(input_data, output_root, n_trials=200):
    """Run time-shift test."""
    print(f"Running time-shift test:")
    print(f"  Input data: {input_data}")
    print(f"  Output root: {output_root}")
    print(f"  Number of trials: {n_trials}")
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
    
    # Run time-shift trials
    print(f"\nRunning {n_trials} time-shift trials...")
    start_time = time.time()
    shifted_maxima = []
    completed_trials = 0
    failed_trials = 0
    
    for i in range(n_trials):
        trial_dir = Path(output_root) / f"trial_{i:04d}"
        trial_dir.mkdir(exist_ok=True)
        
        # Create time-shifted dataset
        shifted_dir = trial_dir / "time_shifted"
        create_time_shifted_dataset(input_data, shifted_dir, i)
        
        # Run detection pipeline
        detection_cmd = [
            "python", "RUN_MODERN_EXOTIC_HUNTER.py",
            "--input", str(shifted_dir),
            "--out", str(trial_dir),
            "--seed", str(12345 + i * 1000)
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
        
        shifted_maxima.append(trial_max)
        completed_trials += 1
        
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"Progress: {i + 1}/{n_trials} trials, "
                  f"{completed_trials} completed, {failed_trials} failed, "
                  f"{rate:.1f} trials/min")
    
    elapsed = time.time() - start_time
    print(f"\nTime-shift test complete!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Completed: {completed_trials}")
    print(f"  Failed: {failed_trials}")
    print(f"  Success rate: {completed_trials/(completed_trials+failed_trials)*100:.1f}%")
    
    # Analyze results
    if shifted_maxima:
        shifted_maxima = np.array(shifted_maxima)
        
        # Compute statistics
        max_shifted = np.max(shifted_maxima)
        mean_shifted = np.mean(shifted_maxima)
        std_shifted = np.std(shifted_maxima)
        
        # Count how many trials still show high significance
        high_sig_trials = (shifted_maxima >= 5.0).sum()
        very_high_sig_trials = (shifted_maxima >= 10.0).sum()
        
        print(f"\n" + "="*60)
        print("TIME-SHIFT TEST RESULTS")
        print("="*60)
        print(f"Observed maximum significance: {observed_max:.2f}σ")
        print(f"Time-shifted distribution statistics:")
        print(f"  Maximum: {max_shifted:.2f}σ")
        print(f"  Mean: {mean_shifted:.2f}σ")
        print(f"  Std: {std_shifted:.2f}σ")
        print(f"  Trials with σ ≥ 5.0: {high_sig_trials}/{len(shifted_maxima)} ({high_sig_trials/len(shifted_maxima)*100:.1f}%)")
        print(f"  Trials with σ ≥ 10.0: {very_high_sig_trials}/{len(shifted_maxima)} ({very_high_sig_trials/len(shifted_maxima)*100:.1f}%)")
        
        # Interpretation
        if high_sig_trials / len(shifted_maxima) > 0.1:
            print(f"\n❌ FAILED: {high_sig_trials/len(shifted_maxima)*100:.1f}% of trials still show high significance")
            print(f"   This suggests the detection is due to intra-pulsar features or")
            print(f"   common instrumental artifacts, not cross-pulsar correlations.")
        else:
            print(f"\n✅ PASSED: Only {high_sig_trials/len(shifted_maxima)*100:.1f}% of trials show high significance")
            print(f"   This suggests the detection is due to genuine cross-pulsar correlations.")
        
        # Save results
        results = {
            'observed_max': observed_max,
            'shifted_maxima': shifted_maxima.tolist(),
            'max_shifted': max_shifted,
            'mean_shifted': mean_shifted,
            'std_shifted': std_shifted,
            'high_sig_trials': high_sig_trials,
            'very_high_sig_trials': very_high_sig_trials,
            'n_trials': len(shifted_maxima)
        }
        
        with open(Path(output_root) / "time_shift_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_root}/time_shift_results.json")
        
    else:
        print("No valid time-shift trials completed!")

def main():
    parser = argparse.ArgumentParser(description="Run time-shift test")
    parser.add_argument("--input", required=True, help="Input real dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for time-shift trials")
    parser.add_argument("--n-trials", type=int, default=200, help="Number of time-shift trials")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        sys.exit(1)
    
    run_time_shift_test(args.input, args.out, args.n_trials)

if __name__ == "__main__":
    main()
