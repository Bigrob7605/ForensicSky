#!/usr/bin/env python3
"""
Run the full V1 Core ForensicSky analysis on real IPTA DR2 data
"""

import sys
import os
sys.path.append('01_Core_Engine')

from Core_ForensicSky_V1 import CoreForensicSkyV1

def main():
    print("🚀 STARTING V1 CORE FORENSIC SKY FULL ANALYSIS")
    print("=" * 60)
    
    # Initialize the V1 engine
    print("Initializing V1 Core ForensicSky engine...")
    engine = CoreForensicSkyV1()
    
    print("✅ V1 Engine initialized successfully!")
    print("GPU acceleration enabled!")
    
    # Run the complete analysis
    print("\n🔬 Running complete analysis...")
    try:
        results = engine.run_complete_analysis()
        print("✅ Complete analysis finished!")
        
        # Run additional specialized analyses
        print("\n🌌 Running cosmic string gold analysis...")
        cosmic_results = engine.run_cosmic_string_gold_analysis()
        print("✅ Cosmic string gold analysis finished!")
        
        print("\n🧠 Running ultra-deep analysis...")
        ultra_results = engine.run_ultra_deep_analysis()
        print("✅ Ultra-deep analysis finished!")
        
        print("\n🏆 Running world-shattering analysis...")
        world_results = engine.run_world_shattering_analysis()
        print("✅ World-shattering analysis finished!")
        
        print("\n🎯 Running lab-grade analysis...")
        lab_results = engine.run_lab_grade_analysis()
        print("✅ Lab-grade analysis finished!")
        
        print("\n" + "=" * 60)
        print("🎉 ALL V1 ANALYSES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
