#!/usr/bin/env python3
"""
CHECK TEST STATUS
================

Quick status check for running validation tests.
"""

import os
import json
from pathlib import Path

def check_test_status():
    """Check status of validation tests."""
    print("VALIDATION TEST STATUS")
    print("=" * 40)
    
    # Check per-method nulls
    nulls_dir = Path("runs/nulls_per_method")
    if nulls_dir.exists():
        trial_dirs = [d for d in nulls_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
        print(f"Per-method nulls: {len(trial_dirs)} trials completed")
        
        # Check for results
        results_file = nulls_dir / "null_stats.csv"
        if results_file.exists():
            print(f"  Results file: {results_file}")
        else:
            print(f"  Results file: Not yet created")
    else:
        print("Per-method nulls: Not started")
    
    # Check full-stack nulls
    full_stack_dir = Path("runs/full_null_mc")
    if full_stack_dir.exists():
        trial_dirs = [d for d in full_stack_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
        print(f"Full-stack nulls: {len(trial_dirs)} trials completed")
        
        # Check for results
        results_file = full_stack_dir / "full_stack_results.json"
        if results_file.exists():
            print(f"  Results file: {results_file}")
        else:
            print(f"  Results file: Not yet created")
    else:
        print("Full-stack nulls: Not started")
    
    # Check time-shift test
    time_shift_dir = Path("runs/time_shift_test")
    if time_shift_dir.exists():
        trial_dirs = [d for d in time_shift_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
        print(f"Time-shift test: {len(trial_dirs)} trials completed")
        
        # Check for results
        results_file = time_shift_dir / "time_shift_results.json"
        if results_file.exists():
            print(f"  Results file: {results_file}")
        else:
            print(f"  Results file: Not yet created")
    else:
        print("Time-shift test: Not started")
    
    print("=" * 40)

if __name__ == "__main__":
    check_test_status()
