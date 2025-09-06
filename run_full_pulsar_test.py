#!/usr/bin/env python3
"""
Run the V1 Core ForensicSky engine on ALL pulsars in the IPTA DR2 dataset
"""

import sys
import os
sys.path.append('01_Core_Engine')

from Core_ForensicSky_V1 import CoreForensicSkyV1

def main():
    print("🚀 STARTING FULL PULSAR ANALYSIS WITH V1 CORE ENGINE")
    print("=" * 70)
    
    # Initialize the V1 engine with the correct data path
    print("Initializing V1 Core ForensicSky engine...")
    engine = CoreForensicSkyV1(data_path="02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/pulsars")
    
    print("✅ V1 Engine initialized successfully!")
    print("GPU acceleration enabled!")
    
    # Load the real data first
    print("\n🔬 Loading real IPTA DR2 data...")
    try:
        loading_stats = engine.load_real_ipta_data()
        print(f"✅ Data loading completed!")
        print(f"   - Total par files found: {loading_stats.get('total_par_files', 0)}")
        print(f"   - Successfully loaded: {loading_stats.get('successful_loads', 0)}")
        print(f"   - Failed loads: {loading_stats.get('failed_loads', 0)}")
        
        if engine.pulsar_catalog is None or len(engine.pulsar_catalog) == 0:
            print("❌ No pulsar data loaded!")
            return
            
        print(f"✅ Loaded {len(engine.pulsar_catalog)} pulsars with timing data")
        
        # Run the complete analysis
        print("\n🌌 Running complete cosmic string analysis...")
        results = engine.run_complete_analysis()
        print("✅ Complete analysis finished!")
        
        # Run additional specialized analyses
        print("\n🧠 Running ultra-deep analysis...")
        ultra_results = engine.run_ultra_deep_analysis()
        print("✅ Ultra-deep analysis finished!")
        
        print("\n🏆 Running world-shattering analysis...")
        world_results = engine.run_world_shattering_analysis()
        print("✅ World-shattering analysis finished!")
        
        print("\n🎯 Running lab-grade analysis...")
        lab_results = engine.run_lab_grade_analysis()
        print("✅ Lab-grade analysis finished!")
        
        print("\n🥇 Running cosmic string gold analysis...")
        cosmic_results = engine.run_cosmic_string_gold_analysis()
        print("✅ Cosmic string gold analysis finished!")
        
        print("\n" + "=" * 70)
        print("🎉 ALL V1 ANALYSES COMPLETED SUCCESSFULLY!")
        print(f"📊 Processed {len(engine.pulsar_catalog)} pulsars")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
