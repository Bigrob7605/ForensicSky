#!/usr/bin/env python3
"""
SCRAMBLE PRESERVE INTERVALS - TIME SCRAMBLING TOOL
=================================================

Creates scrambled versions of real datasets that preserve:
- Inter-epoch intervals within each pulsar
- Pulsar-specific properties
- Observatory metadata

But destroys:
- Cross-pulsar correlations
- Sky-coherent signals
- True astrophysical timing

Usage:
    python tools/scramble_preserve_intervals.py --input 02_Data/real_ipta_dr2 --out scrambled_data --seed 12345
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
import numpy as np

def scramble_pulsar_data(input_file, output_file, seed=None):
    """Scramble a single pulsar's timing data while preserving intervals."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse data lines (skip comments and headers)
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
                # Skip lines that can't be parsed
                header_lines.append(line)
    
    if not data_lines:
        print(f"Warning: No valid data lines found in {input_file}")
        return
    
    # Extract MJD and residuals
    mjds = np.array([d[0] for d in data_lines])
    residuals = np.array([d[1] for d in data_lines])
    errors = np.array([d[2] for d in data_lines])
    
    # Method 1: Time-scramble (preserve intervals)
    # Shuffle the residuals while keeping MJD order
    scrambled_residuals = residuals.copy()
    np.random.shuffle(scrambled_residuals)
    
    # Method 2: Add random time offset (alternative)
    # time_offset = np.random.uniform(-1000, 1000)  # days
    # scrambled_mjds = mjds + time_offset
    
    # Write scrambled data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Write headers
        for header in header_lines:
            f.write(header + '\n')
        
        # Write scrambled data
        for i, (mjd, residual, error) in enumerate(zip(mjds, scrambled_residuals, errors)):
            f.write(f"{mjd:.6f} {residual:.6f} {error:.6f}\n")
    
    print(f"Scrambled {len(data_lines)} observations in {input_file} -> {output_file}")

def scramble_dataset(input_dir, output_dir, seed=None):
    """Scramble an entire dataset directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pulsar directories
    pulsar_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(pulsar_dirs)} pulsar directories")
    
    for pulsar_dir in pulsar_dirs:
        pulsar_name = pulsar_dir.name
        output_pulsar_dir = output_path / pulsar_name
        output_pulsar_dir.mkdir(exist_ok=True)
        
        # Find data files in this pulsar directory
        data_files = list(pulsar_dir.glob("*.txt")) + list(pulsar_dir.glob("*.dat"))
        
        if not data_files:
            print(f"Warning: No data files found in {pulsar_dir}")
            continue
        
        # Scramble each data file
        for data_file in data_files:
            output_file = output_pulsar_dir / data_file.name
            scramble_pulsar_data(data_file, output_file, seed)
        
        print(f"Scrambled pulsar {pulsar_name}: {len(data_files)} files")
    
    # Copy any other files (metadata, etc.)
    for item in input_path.iterdir():
        if item.is_file() and not item.name.endswith(('.txt', '.dat')):
            shutil.copy2(item, output_path / item.name)
    
    print(f"Scrambling complete: {input_dir} -> {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Scramble timing data while preserving intervals")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--out", required=True, help="Output scrambled directory")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Scrambling dataset: {args.input} -> {args.out}")
    if args.seed is not None:
        print(f"Using seed: {args.seed}")
    
    scramble_dataset(args.input, args.out, args.seed)
    print("Scrambling complete!")

if __name__ == "__main__":
    main()
