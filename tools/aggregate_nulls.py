#!/usr/bin/env python3
"""
AGGREGATE NULLS - COLLECT PER-METHOD STATISTICS
==============================================

Aggregates per-method null statistics from multiple trial runs
and computes empirical p-values for each method.

Usage:
    python tools/aggregate_nulls.py --root runs/nulls_per_method --out null_stats.csv
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
import numpy as np

def collect_trial_stats(trial_dir):
    """Collect statistics from a single trial directory."""
    stats = {}
    
    # Look for report files
    report_files = list(Path(trial_dir).glob("**/modern_exotic_physics_report_*.json"))
    
    if not report_files:
        print(f"Warning: No report files found in {trial_dir}")
        return None
    
    # Use the first report file found
    report_file = report_files[0]
    
    try:
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        # Extract per-method statistics
        if 'channels' in report:
            for channel, data in report['channels'].items():
                stats[f"{channel}_sigma"] = data.get('significance', 0.0)
                stats[f"{channel}_detections"] = data.get('detections', 0)
        
        # Extract combined statistics
        if 'summary' in report:
            stats['max_combined_sigma'] = report['summary'].get('max_significance', 0.0)
            stats['total_detections'] = report['summary'].get('total_detections', 0)
        
        # Extract individual method results if available
        if 'methods' in report:
            for method, data in report['methods'].items():
                stats[f"{method}_sigma"] = data.get('significance', 0.0)
        
        return stats
        
    except Exception as e:
        print(f"Error reading {report_file}: {e}")
        return None

def aggregate_nulls(root_dir, output_file):
    """Aggregate null statistics from all trials."""
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory {root_dir} does not exist")
    
    # Find all trial directories
    trial_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('trial_')]
    
    print(f"Found {len(trial_dirs)} trial directories")
    
    all_stats = []
    
    for trial_dir in trial_dirs:
        trial_num = trial_dir.name.split('_')[1]
        stats = collect_trial_stats(trial_dir)
        
        if stats is not None:
            stats['trial'] = int(trial_num)
            all_stats.append(stats)
        else:
            print(f"Skipping trial {trial_num} - no valid stats")
    
    if not all_stats:
        print("No valid statistics found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_stats)
    
    # Fill missing values with 0
    df = df.fillna(0)
    
    # Sort by trial number
    df = df.sort_values('trial')
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Saved {len(df)} trials to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NULL STATISTICS SUMMARY")
    print("="*60)
    
    # Per-method statistics
    method_columns = [col for col in df.columns if col.endswith('_sigma') and col != 'max_combined_sigma']
    
    for col in method_columns:
        method_name = col.replace('_sigma', '')
        values = df[col].dropna()
        if len(values) > 0:
            max_val = values.max()
            mean_val = values.mean()
            std_val = values.std()
            print(f"{method_name:20s}: max={max_val:6.2f}σ, mean={mean_val:6.2f}σ, std={std_val:6.2f}σ")
    
    # Combined statistics
    if 'max_combined_sigma' in df.columns:
        combined_values = df['max_combined_sigma'].dropna()
        if len(combined_values) > 0:
            max_combined = combined_values.max()
            mean_combined = combined_values.mean()
            std_combined = combined_values.std()
            print(f"{'max_combined_sigma':20s}: max={max_combined:6.2f}σ, mean={mean_combined:6.2f}σ, std={std_combined:6.2f}σ")
    
    print("="*60)
    
    return df

def compute_empirical_p_values(df, observed_values):
    """Compute empirical p-values for observed values."""
    print("\n" + "="*60)
    print("EMPIRICAL P-VALUE ANALYSIS")
    print("="*60)
    
    for method, observed in observed_values.items():
        if method in df.columns:
            null_values = df[method].dropna()
            if len(null_values) > 0:
                # Count how many null values are >= observed
                p_value = (null_values >= observed).sum() / len(null_values)
                print(f"{method:20s}: observed={observed:6.2f}σ, p-value={p_value:.2e}")
                
                if p_value < 1e-3:
                    print(f"  -> {method} is well-calibrated (p < 1e-3)")
                else:
                    print(f"  -> {method} may be overconfident (p >= 1e-3)")
            else:
                print(f"{method:20s}: No null values found")
        else:
            print(f"{method:20s}: Column not found in data")

def main():
    parser = argparse.ArgumentParser(description="Aggregate null statistics from trial runs")
    parser.add_argument("--root", required=True, help="Root directory containing trial subdirectories")
    parser.add_argument("--out", required=True, help="Output CSV file")
    parser.add_argument("--observed", help="JSON file with observed values for p-value computation")
    
    args = parser.parse_args()
    
    print(f"Aggregating nulls from: {args.root}")
    print(f"Output file: {args.out}")
    
    df = aggregate_nulls(args.root, args.out)
    
    # Compute empirical p-values if observed values provided
    if args.observed and os.path.exists(args.observed):
        with open(args.observed, 'r') as f:
            observed_values = json.load(f)
        compute_empirical_p_values(df, observed_values)
    
    print(f"\nAggregation complete! Results saved to {args.out}")

if __name__ == "__main__":
    main()
