#!/usr/bin/env python3
"""
REAL TIMING DATA LOADER - Loads actual IPTA DR2 timing data from tims subdirectories
"""

import os
import glob
import numpy as np
import json
from datetime import datetime
import sys
sys.path.append('01_Core_Engine')

from Core_ForensicSky_V1 import CoreForensicSkyV1

class RealTimingDataLoader:
    """Loads and processes real IPTA DR2 timing data from tims subdirectories"""
    
    def __init__(self):
        self.data_path = "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA"
        self.clock_path = "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/clock"
        self.pulsar_data = []
        self.timing_data = []
        
    def load_real_timing_data(self):
        """Load real IPTA DR2 timing data from tims subdirectories"""
        print("🔍 LOADING REAL IPTA DR2 TIMING DATA...")
        print(f"📁 Data path: {self.data_path}")
        
        # Get all pulsar directories
        pulsar_dirs = [d for d in os.listdir(self.data_path) if d.startswith('J') and os.path.isdir(os.path.join(self.data_path, d))]
        print(f"✅ Found {len(pulsar_dirs)} pulsars")
        
        successful_loads = 0
        total_observations = 0
        
        for i, pulsar_dir in enumerate(pulsar_dirs, 1):
            print(f"[{i}/{len(pulsar_dirs)}] Loading {pulsar_dir}...")
            
            try:
                # Load par file
                par_file = None
                for par_name in [f"{pulsar_dir}.par", f"{pulsar_dir}.IPTADR2.par", f"{pulsar_dir}.IPTADR2.TDB.par"]:
                    test_par = os.path.join(self.data_path, pulsar_dir, par_name)
                    if os.path.exists(test_par):
                        par_file = test_par
                        break
                
                if not par_file:
                    print(f"  ❌ No par file found for {pulsar_dir}")
                    continue
                
                # Load all timing files from tims subdirectory
                tims_dir = os.path.join(self.data_path, pulsar_dir, "tims")
                if not os.path.exists(tims_dir):
                    print(f"  ❌ No tims directory found for {pulsar_dir}")
                    continue
                
                # Get all .tim files in tims directory
                tim_files = glob.glob(os.path.join(tims_dir, "*.tim"))
                if not tim_files:
                    print(f"  ❌ No timing files found in tims directory for {pulsar_dir}")
                    continue
                
                print(f"  📁 Found {len(tim_files)} timing files")
                
                # Load the data using the core engine's methods
                engine = CoreForensicSkyV1()
                
                # Load par file
                par_data = engine.load_par_file(par_file)
                if not par_data:
                    print(f"  ❌ Failed to load par file for {pulsar_dir}")
                    continue
                
                # Load all timing files and combine data
                all_timing_data = []
                total_obs = 0
                
                for tim_file in tim_files:
                    try:
                        tim_data = engine.load_tim_file(tim_file)
                        if tim_data:
                            all_timing_data.extend(tim_data)
                            total_obs += len(tim_data)
                            print(f"    ✅ Loaded {os.path.basename(tim_file)}: {len(tim_data)} observations")
                    except Exception as e:
                        print(f"    ⚠️ Error loading {os.path.basename(tim_file)}: {e}")
                        continue
                
                if not all_timing_data:
                    print(f"  ❌ No timing data loaded for {pulsar_dir}")
                    continue
                
                # Store the data
                pulsar_info = {
                    'name': pulsar_dir,
                    'par_data': par_data,
                    'tim_data': all_timing_data,
                    'n_observations': len(all_timing_data),
                    'n_timing_files': len(tim_files)
                }
                
                self.pulsar_data.append(pulsar_info)
                successful_loads += 1
                total_observations += len(all_timing_data)
                
                print(f"  ✅ Loaded {pulsar_dir} - {len(all_timing_data)} total observations from {len(tim_files)} files")
                
            except Exception as e:
                print(f"  ❌ Error loading {pulsar_dir}: {e}")
                continue
        
        print(f"\n✅ SUCCESSFULLY LOADED {successful_loads} PULSARS")
        print(f"📊 Total observations: {total_observations}")
        return successful_loads > 0
    
    def run_analysis(self):
        """Run cosmic string analysis on the loaded data"""
        print("\n🔬 RUNNING COSMIC STRING ANALYSIS...")
        
        if not self.pulsar_data:
            print("❌ No data loaded!")
            return None
        
        # Initialize analysis engine
        engine = CoreForensicSkyV1()
        
        # Set up the data for analysis
        engine.pulsar_data = self.pulsar_data
        engine.timing_data = self.timing_data
        
        # Run correlation analysis
        print("📊 Running correlation analysis...")
        try:
            correlation_results = engine.correlation_analysis()
            print(f"✅ Correlation analysis completed")
        except Exception as e:
            print(f"❌ Correlation analysis failed: {e}")
            correlation_results = None
        
        # Run spectral analysis
        print("📈 Running spectral analysis...")
        try:
            spectral_results = engine.spectral_analysis()
            print(f"✅ Spectral analysis completed")
        except Exception as e:
            print(f"❌ Spectral analysis failed: {e}")
            spectral_results = None
        
        # Run ML analysis
        print("🤖 Running ML analysis...")
        try:
            ml_results = engine.ml_analysis()
            print(f"✅ ML analysis completed")
        except Exception as e:
            print(f"❌ ML analysis failed: {e}")
            ml_results = None
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_pulsars': len(self.pulsar_data),
            'total_observations': sum(p['n_observations'] for p in self.pulsar_data),
            'correlation_analysis': correlation_results,
            'spectral_analysis': spectral_results,
            'ml_analysis': ml_results,
            'pulsar_summary': [
                {
                    'name': p['name'],
                    'n_observations': p['n_observations'],
                    'n_timing_files': p['n_timing_files']
                } for p in self.pulsar_data
            ]
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"04_Results/real_timing_data_analysis_{timestamp}.json"
        
        os.makedirs("04_Results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        
        return results

def main():
    print("🚀 REAL IPTA DR2 TIMING DATA ANALYSIS")
    print("=" * 50)
    
    # Create loader
    loader = RealTimingDataLoader()
    
    # Load real timing data
    if not loader.load_real_timing_data():
        print("❌ Failed to load real timing data!")
        return
    
    # Run analysis
    results = loader.run_analysis()
    
    if results:
        print("\n🎉 REAL TIMING DATA ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {results['total_pulsars']} pulsars")
        print(f"📊 Total observations: {results['total_observations']}")
        print("Check 04_Results/ for detailed results")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main()
