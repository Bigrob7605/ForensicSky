#!/usr/bin/env python3
"""
FULL V1 CORE SYSTEM TEST - Run the complete Core ForensicSky V1 system on real IPTA DR2 data
"""

import sys
import os
sys.path.append('01_Core_Engine')

from Core_ForensicSky_V1 import CoreForensicSkyV1

def main():
    print("🚀 FULL V1 CORE SYSTEM TEST - ALL 30+ ADVANCED TECHNOLOGIES")
    print("=" * 70)
    
    # Initialize the V1 core system with correct data path
    data_path = "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA"
    print(f"📁 Data path: {data_path}")
    
    # Create the engine
    engine = CoreForensicSkyV1(data_path)
    
    # Manually set up the data loading since the core system has path issues
    print("\n🔬 LOADING REAL IPTA DR2 DATA...")
    
    # Get all pulsar directories
    pulsar_dirs = [d for d in os.listdir(data_path) if d.startswith('J') and os.path.isdir(os.path.join(data_path, d))]
    print(f"✅ Found {len(pulsar_dirs)} pulsars")
    
    # Load data manually
    engine.pulsar_catalog = []
    engine.timing_data = []
    
    successful_loads = 0
    total_observations = 0
    
    for i, pulsar_dir in enumerate(pulsar_dirs, 1):
        print(f"[{i}/{len(pulsar_dirs)}] Loading {pulsar_dir}...")
        
        try:
            # Load par file
            par_file = None
            for par_name in [f"{pulsar_dir}.par", f"{pulsar_dir}.IPTADR2.par", f"{pulsar_dir}.IPTADR2.TDB.par"]:
                test_par = os.path.join(data_path, pulsar_dir, par_name)
                if os.path.exists(test_par):
                    par_file = test_par
                    break
            
            if not par_file:
                continue
            
            # Load all timing files from tims subdirectory
            tims_dir = os.path.join(data_path, pulsar_dir, "tims")
            if not os.path.exists(tims_dir):
                continue
            
            # Get all .tim files in tims directory
            import glob
            tim_files = glob.glob(os.path.join(tims_dir, "*.tim"))
            if not tim_files:
                continue
            
            # Load par file
            par_data = engine.load_par_file(par_file)
            if not par_data:
                continue
            
            # Load all timing files and combine data
            all_timing_data = []
            
            for tim_file in tim_files:
                try:
                    tim_data = engine.load_tim_file(tim_file)
                    if tim_data:
                        all_timing_data.extend(tim_data)
                except Exception as e:
                    continue
            
            if not all_timing_data:
                continue
            
            # Store the data
            pulsar_info = {
                'name': pulsar_dir,
                'par_data': par_data,
                'tim_data': all_timing_data,
                'n_observations': len(all_timing_data)
            }
            
            engine.pulsar_catalog.append(pulsar_info)
            engine.timing_data.extend(all_timing_data)
            successful_loads += 1
            total_observations += len(all_timing_data)
            
            print(f"  ✅ Loaded {pulsar_dir} - {len(all_timing_data)} observations")
            
        except Exception as e:
            print(f"  ❌ Error loading {pulsar_dir}: {e}")
            continue
    
    print(f"\n✅ SUCCESSFULLY LOADED {successful_loads} PULSARS")
    print(f"📊 Total observations: {total_observations}")
    
    if successful_loads == 0:
        print("❌ No data loaded! Cannot proceed with analysis.")
        return
    
    # Now run the complete V1 system analysis
    print("\n🔬 RUNNING COMPLETE V1 CORE SYSTEM ANALYSIS...")
    print("🚀 Using ALL 30+ advanced technologies...")
    
    try:
        # Run the complete analysis
        results = engine.run_complete_analysis()
        
        print("\n🎉 FULL V1 CORE SYSTEM ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"📊 Results: {results}")
        
        # Save results
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"04_Results/full_v1_system_test_{timestamp}.json"
        
        os.makedirs("04_Results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during V1 system analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
