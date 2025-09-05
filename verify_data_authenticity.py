#!/usr/bin/env python3
"""
Verify Data Authenticity
========================

Check if our data is real IPTA DR2 data or if we're accidentally using toy data.
"""

import numpy as np
import json
from pathlib import Path

def verify_data_authenticity():
    """Verify if our data is real or toy data"""
    print("🔍 DATA AUTHENTICATION VERIFICATION")
    print("====================================")
    print("🎯 Mission: Verify if we're using REAL IPTA DR2 data or toy data")
    print("🎯 Source: IPTA DR2 GitLab repository")
    print("====================================")
    
    # Check our processed data
    data_file = Path("data/ipta_dr2/processed/ipta_dr2_versionA_processed.npz")
    if not data_file.exists():
        print("❌ No processed data file found!")
        return
    
    data = np.load(data_file, allow_pickle=True)
    print(f"📊 Data file keys: {list(data.keys())}")
    
    # Check pulsar catalog
    pulsar_catalog = data['pulsar_catalog']
    print(f"📊 Pulsar catalog: {len(pulsar_catalog)} pulsars")
    
    if len(pulsar_catalog) > 0:
        first_pulsar = pulsar_catalog[0]
        print(f"📊 First pulsar: {first_pulsar}")
        
        # Check for toy data signatures
        print("\n🔍 TOY DATA DETECTION:")
        
        # Check if all uncertainties are the same (toy data signature)
        timing_data = data['timing_data']
        if len(timing_data) > 0:
            uncertainties = timing_data[0]['uncertainties']
            unique_uncertainties = len(np.unique(uncertainties))
            print(f"   Unique uncertainty values: {unique_uncertainties}")
            if unique_uncertainties == 1:
                print("   ⚠️  WARNING: All uncertainties are identical - TOY DATA SIGNATURE!")
            else:
                print("   ✅ Uncertainties vary - looks like real data")
            
            # Check if all FAP values are 0 (toy data signature)
            times = timing_data[0]['times']
            residuals = timing_data[0]['residuals']
            print(f"   Time range: {np.min(times):.2f} to {np.max(times):.2f}")
            print(f"   Residual range: {np.min(residuals):.6f} to {np.max(residuals):.6f}")
            print(f"   Number of observations: {len(times)}")
            
            # Check for realistic time spans (real data should span years)
            time_span = np.max(times) - np.min(times)
            print(f"   Time span: {time_span:.1f} days ({time_span/365.25:.1f} years)")
            if time_span < 100:  # Less than 100 days
                print("   ⚠️  WARNING: Very short time span - might be toy data!")
            else:
                print("   ✅ Realistic time span - looks like real data")
    
    # Check the original cosmic string inputs
    print("\n🔍 ORIGINAL DATA CHECK:")
    cosmic_file = Path("data/ipta_dr2/processed/cosmic_string_inputs_versionA.npz")
    if cosmic_file.exists():
        cosmic_data = np.load(cosmic_file, allow_pickle=True)
        print(f"📊 Cosmic string inputs keys: {list(cosmic_data.keys())}")
        
        # Check if this looks like real IPTA data
        if 'pulsar_names' in cosmic_data:
            names = cosmic_data['pulsar_names']
            print(f"📊 Pulsar names: {names[:5]}...")  # First 5 names
            
            # Check if names look like real pulsar names (J+coordinates)
            real_name_count = sum(1 for name in names if str(name).startswith('J'))
            print(f"📊 Real pulsar name format: {real_name_count}/{len(names)} ({100*real_name_count/len(names):.1f}%)")
            
            if real_name_count > 0.8 * len(names):
                print("   ✅ Pulsar names look real - IPTA format!")
            else:
                print("   ⚠️  WARNING: Pulsar names don't look like real IPTA format!")
        
        if 'pulsar_positions' in cosmic_data:
            positions = cosmic_data['pulsar_positions']
            print(f"📊 Position data shape: {positions.shape}")
            print(f"📊 RA range: {np.min(positions[:, 0]):.3f} to {np.max(positions[:, 0]):.3f}")
            print(f"📊 DEC range: {np.min(positions[:, 1]):.3f} to {np.max(positions[:, 1]):.3f}")
            
            # Check if positions look realistic (RA: 0-2π, DEC: -π/2 to π/2)
            if (np.min(positions[:, 0]) >= 0 and np.max(positions[:, 0]) <= 2*np.pi and
                np.min(positions[:, 1]) >= -np.pi/2 and np.max(positions[:, 1]) <= np.pi/2):
                print("   ✅ Position ranges look realistic!")
            else:
                print("   ⚠️  WARNING: Position ranges don't look realistic!")
    
    # Check if we have the real IPTA DR2 data directory
    print("\n🔍 REAL IPTA DR2 DATA CHECK:")
    real_data_dir = Path("data/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA")
    if real_data_dir.exists():
        par_files = list(real_data_dir.glob("*/J*.par"))
        print(f"📊 Found {len(par_files)} real IPTA DR2 parameter files")
        
        if len(par_files) > 0:
            print("   ✅ REAL IPTA DR2 data is available!")
            print("   📁 This is the authentic data from GitLab")
            print("   🎯 We should be using THIS data, not the processed version!")
        else:
            print("   ❌ No real IPTA DR2 files found")
    else:
        print("   ❌ Real IPTA DR2 data directory not found")
    
    print("\n🎯 CONCLUSION:")
    print("==============")
    if real_data_dir.exists() and len(list(real_data_dir.glob("*/J*.par"))) > 0:
        print("✅ REAL IPTA DR2 data is available from GitLab")
        print("⚠️  We should process and use the REAL data, not simulated data")
        print("🚀 Let's extract the real patterns from the authentic IPTA DR2 data!")
    else:
        print("❌ No real IPTA DR2 data found")
        print("🔍 We need to download the real data from GitLab")

if __name__ == "__main__":
    verify_data_authenticity()
