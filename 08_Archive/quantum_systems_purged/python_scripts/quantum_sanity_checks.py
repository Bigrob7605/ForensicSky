#!/usr/bin/env python3
"""
Quantum Sanity Checks for PTA Elite Sample
==========================================

Three critical checks to validate quantum claims:
1. High-kernel pairs vs classical correlation
2. High-kernel pairs vs DM correlation  
3. Sky-distance vs quantum kernel correlation
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_quantum_results():
    """Load the quantum tomography results"""
    with open('quantum_50_premium_pulsars_20250905_193411.json', 'r') as f:
        data = json.load(f)
    return data

def load_pulsar_data():
    """Load pulsar data including positions and DM values"""
    print("🔬 Loading pulsar data for sanity checks...")
    
    try:
        # Import the Core Forensic Sky V1 engine
        sys.path.append('01_Core_Engine')
        from Core_ForensicSky_V1 import CoreForensicSkyV1
        
        # Initialize engine
        engine = CoreForensicSkyV1()
        
        # Load real IPTA DR2 data
        loading_stats = engine.load_real_ipta_data()
        
        if not hasattr(engine, 'timing_data') or not engine.timing_data:
            print("❌ No timing data loaded from engine!")
            return None, None, None
        
        # Extract data for our 39 pulsars
        pulsar_ids = load_quantum_results()['pulsar_ids']
        
        residuals = {}
        positions = {}
        dm_values = {}
        
        for pulsar_id in pulsar_ids:
            if pulsar_id in engine.timing_data:
                timing_info = engine.timing_data[pulsar_id]
                if 'residuals' in timing_info and len(timing_info['residuals']) > 0:
                    residuals[pulsar_id] = np.array(timing_info['residuals'])
                    
                    # Get position (RA, DEC) from par file data
                    if 'par_data' in timing_info and timing_info['par_data']:
                        par_data = timing_info['par_data']
                        if 'RAJ' in par_data and 'DECJ' in par_data:
                            # Convert RA/DEC to radians
                            ra = float(par_data['RAJ'].replace(':', ' ').split()[0]) * 15.0  # Convert to degrees
                            dec = float(par_data['DECJ'].replace(':', ' ').split()[0])
                            positions[pulsar_id] = (ra, dec)
                        
                        # Get DM value
                        if 'DM' in par_data:
                            dm_values[pulsar_id] = float(par_data['DM'])
        
        print(f"✅ Loaded data for {len(residuals)} pulsars")
        return residuals, positions, dm_values
        
    except Exception as e:
        print(f"❌ Error loading pulsar data: {e}")
        return None, None, None

def check_1_high_kernel_vs_classical(data, residuals):
    """Check 1: Are high-kernel pairs also high-classical?"""
    print("\n🔍 CHECK 1: High-kernel pairs vs classical correlation")
    print("-" * 60)
    
    kernels = np.array(data['kernels'])
    pulsar_ids = data['pulsar_ids']
    
    # Find top 5 high-kernel pairs
    n_pulsars = len(pulsar_ids)
    top_pairs = []
    
    for i in range(n_pulsars):
        for j in range(i+1, n_pulsars):
            top_pairs.append((i, j, kernels[i, j]))
    
    # Sort by kernel value (descending)
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_5_pairs = top_pairs[:5]
    
    print("Top 5 high-kernel pairs:")
    print("Pulsar1 ↔ Pulsar2 | Kernel | Classical ρ | Status")
    print("-" * 50)
    
    for i, (idx1, idx2, kernel_val) in enumerate(top_5_pairs):
        pulsar1 = pulsar_ids[idx1]
        pulsar2 = pulsar_ids[idx2]
        
        # Calculate classical correlation
        if pulsar1 in residuals and pulsar2 in residuals:
            # Ensure same length
            min_len = min(len(residuals[pulsar1]), len(residuals[pulsar2]))
            res1 = residuals[pulsar1][:min_len]
            res2 = residuals[pulsar2][:min_len]
            
            classical_corr = np.corrcoef(res1, res2)[0, 1]
            
            # Check if this kills the quantum claim
            status = "❌ KILLS QUANTUM" if classical_corr > 0.3 else "✅ QUANTUM SAFE"
            
            print(f"{pulsar1} ↔ {pulsar2} | {kernel_val:.3f} | {classical_corr:.3f} | {status}")
        else:
            print(f"{pulsar1} ↔ {pulsar2} | {kernel_val:.3f} | N/A | ⚠️ NO DATA")
    
    return top_5_pairs

def check_2_high_kernel_vs_dm(data, dm_values):
    """Check 2: Are high-kernel pairs also high-DM?"""
    print("\n🔍 CHECK 2: High-kernel pairs vs DM correlation")
    print("-" * 60)
    
    kernels = np.array(data['kernels'])
    pulsar_ids = data['pulsar_ids']
    
    # Find top 5 high-kernel pairs
    n_pulsars = len(pulsar_ids)
    top_pairs = []
    
    for i in range(n_pulsars):
        for j in range(i+1, n_pulsars):
            top_pairs.append((i, j, kernels[i, j]))
    
    # Sort by kernel value (descending)
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_5_pairs = top_pairs[:5]
    
    print("Top 5 high-kernel pairs DM analysis:")
    print("Pulsar1 ↔ Pulsar2 | Kernel | DM1 | DM2 | DM Diff | Status")
    print("-" * 60)
    
    for i, (idx1, idx2, kernel_val) in enumerate(top_5_pairs):
        pulsar1 = pulsar_ids[idx1]
        pulsar2 = pulsar_ids[idx2]
        
        if pulsar1 in dm_values and pulsar2 in dm_values:
            dm1 = dm_values[pulsar1]
            dm2 = dm_values[pulsar2]
            dm_diff = abs(dm1 - dm2)
            
            # Check if this flags ISM covariance
            status = "❌ ISM COVARIANCE" if dm_diff < 10 else "✅ QUANTUM SAFE"
            
            print(f"{pulsar1} ↔ {pulsar2} | {kernel_val:.3f} | {dm1:.1f} | {dm2:.1f} | {dm_diff:.1f} | {status}")
        else:
            print(f"{pulsar1} ↔ {pulsar2} | {kernel_val:.3f} | N/A | N/A | N/A | ⚠️ NO DM DATA")
    
    return top_5_pairs

def check_3_sky_distance_vs_kernel(data, positions):
    """Check 3: Sky-distance vs quantum kernel correlation"""
    print("\n🔍 CHECK 3: Sky-distance vs quantum kernel correlation")
    print("-" * 60)
    
    kernels = np.array(data['kernels'])
    pulsar_ids = data['pulsar_ids']
    
    # Calculate angular separations
    n_pulsars = len(pulsar_ids)
    angular_seps = []
    kernel_values = []
    
    for i in range(n_pulsars):
        for j in range(i+1, n_pulsars):
            if pulsar_ids[i] in positions and pulsar_ids[j] in positions:
                pos1 = positions[pulsar_ids[i]]
                pos2 = positions[pulsar_ids[j]]
                
                # Calculate angular separation
                ra1, dec1 = np.radians(pos1)
                ra2, dec2 = np.radians(pos2)
                
                # Angular separation formula
                cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
                cos_sep = np.clip(cos_sep, -1, 1)  # Avoid numerical errors
                angular_sep = np.degrees(np.arccos(cos_sep))
                
                angular_seps.append(angular_sep)
                kernel_values.append(kernels[i, j])
    
    if len(angular_seps) > 0:
        # Calculate correlation between angular separation and kernel values
        correlation, p_value = pearsonr(angular_seps, kernel_values)
        
        print(f"Angular separation vs kernel correlation: {correlation:.3f}")
        print(f"P-value: {p_value:.3f}")
        
        # Check if this shows geometric bias
        if correlation < -0.3:
            print("❌ GEOMETRIC BIAS - quantum entanglement shouldn't care about angular separation!")
        elif correlation > 0.3:
            print("❌ GEOMETRIC BIAS - quantum entanglement shouldn't care about angular separation!")
        else:
            print("✅ QUANTUM SAFE - no significant geometric bias")
        
        return angular_seps, kernel_values, correlation
    else:
        print("⚠️ No position data available for angular separation analysis")
        return None, None, None

def main():
    """Run all sanity checks"""
    print("🧠 QUANTUM SANITY CHECKS - PTA ELITE SAMPLE")
    print("=" * 60)
    
    # Load quantum results
    data = load_quantum_results()
    
    # Load pulsar data
    residuals, positions, dm_values = load_pulsar_data()
    
    if residuals is None:
        print("❌ Cannot proceed without pulsar data")
        return
    
    # Run all checks
    top_pairs = check_1_high_kernel_vs_classical(data, residuals)
    check_2_high_kernel_vs_dm(data, dm_values)
    angular_seps, kernel_values, correlation = check_3_sky_distance_vs_kernel(data, positions)
    
    print("\n🎯 SANITY CHECK SUMMARY")
    print("=" * 40)
    print("✅ All checks completed!")
    print("Review the results above to validate quantum claims.")
    
    # Save results for plotting
    if angular_seps is not None:
        results = {
            'angular_separations': angular_seps,
            'kernel_values': kernel_values,
            'correlation': correlation,
            'top_pairs': top_pairs
        }
        
        with open('quantum_sanity_check_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("📊 Results saved to quantum_sanity_check_results.json")

if __name__ == "__main__":
    main()
