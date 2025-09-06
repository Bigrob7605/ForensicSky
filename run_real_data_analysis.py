#!/usr/bin/env python3
"""
RUN REAL DATA ANALYSIS
Uses the Core ForensicSky V1 engine to process actual IPTA DR2 data
"""

import sys
import os
sys.path.append('01_Core_Engine')

from Core_ForensicSky_V1 import CoreForensicSkyV1

def main():
    print("🚀 RUNNING REAL IPTA DR2 DATA ANALYSIS")
    print("=" * 50)
    
    # Initialize the core engine with correct data path
    data_path = "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA"
    engine = CoreForensicSkyV1(data_path)
    
    print(f"📁 Data path set to: {engine.data_path}")
    
    # Load real IPTA DR2 data
    print("\n🔍 LOADING REAL IPTA DR2 DATA...")
    try:
        engine.load_real_ipta_data()
        print("✅ Real data loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run complete analysis
    print("\n🔬 RUNNING COMPLETE ANALYSIS...")
    try:
        results = engine.run_complete_analysis()
        print("✅ Analysis completed successfully!")
        print(f"📊 Results: {results}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return
    
    print("\n🎉 REAL DATA ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()
