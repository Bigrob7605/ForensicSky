#!/usr/bin/env python3
"""
REAL DATA LOADER - Loads actual IPTA DR2 data and runs analysis
"""

import os
import glob
import numpy as np
import json
from datetime import datetime
import sys
sys.path.append('01_Core_Engine')

from Core_ForensicSky_V1 import CoreForensicSkyV1

class RealDataLoader:
    """Loads and processes real IPTA DR2 data"""
    
    def __init__(self):
        self.data_path = "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA"
        self.clock_path = "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/clock"
        self.pulsar_data = []
        self.timing_data = []
        
    def load_real_data(self):
        """Load real IPTA DR2 data from VersionA directory"""
        print("🔍 LOADING REAL IPTA DR2 DATA...")
        print(f"📁 Data path: {self.data_path}")
        
        # Get all pulsar directories
        pulsar_dirs = [d for d in os.listdir(self.data_path) if d.startswith('J') and os.path.isdir(os.path.join(self.data_path, d))]
        print(f"✅ Found {len(pulsar_dirs)} pulsars")
        
        successful_loads = 0
        
        for i, pulsar_dir in enumerate(pulsar_dirs, 1):
            print(f"[{i}/{len(pulsar_dirs)}] Loading {pulsar_dir}...")
            
            try:
                # Load par file (try different naming conventions)
                par_file = None
                for par_name in [f"{pulsar_dir}.par", f"{pulsar_dir}.IPTADR2.par", f"{pulsar_dir}.IPTADR2.TDB.par"]:
                    test_par = os.path.join(self.data_path, pulsar_dir, par_name)
                    if os.path.exists(test_par):
                        par_file = test_par
                        break
                
                if not par_file:
                    print(f"  ❌ No par file found for {pulsar_dir}")
                    continue
                
                # Load tim file (try different naming conventions)
                tim_file = None
                for tim_name in [f"{pulsar_dir}.tim", f"{pulsar_dir}.IPTADR2.tim"]:
                    test_tim = os.path.join(self.data_path, pulsar_dir, tim_name)
                    if os.path.exists(test_tim):
                        tim_file = test_tim
                        break
                
                if not tim_file:
                    print(f"  ❌ No tim file found for {pulsar_dir}")
                    continue
                
                # Load the data using the core engine's methods
                engine = CoreForensicSkyV1()
                
                # Load par file
                par_data = engine.load_par_file(par_file)
                if not par_data:
                    print(f"  ❌ Failed to load par file for {pulsar_dir}")
                    continue
                
                # Load tim file
                tim_data = engine.load_tim_file(tim_file)
                if not tim_data:
                    print(f"  ❌ Failed to load tim file for {pulsar_dir}")
                    continue
                
                # Store the data
                pulsar_info = {
                    'name': pulsar_dir,
                    'par_data': par_data,
                    'tim_data': tim_data,
                    'n_observations': len(tim_data) if tim_data else 0
                }
                
                self.pulsar_data.append(pulsar_info)
                successful_loads += 1
                
                print(f"  ✅ Loaded {pulsar_dir} - {pulsar_info['n_observations']} observations")
                
            except Exception as e:
                print(f"  ❌ Error loading {pulsar_dir}: {e}")
                continue
        
        print(f"\n✅ SUCCESSFULLY LOADED {successful_loads} PULSARS")
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
            'correlation_analysis': correlation_results,
            'spectral_analysis': spectral_results,
            'ml_analysis': ml_results,
            'pulsar_summary': [
                {
                    'name': p['name'],
                    'n_observations': p['n_observations']
                } for p in self.pulsar_data
            ]
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"04_Results/real_data_analysis_{timestamp}.json"
        
        os.makedirs("04_Results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        
        return results

def main():
    print("🚀 REAL IPTA DR2 DATA ANALYSIS")
    print("=" * 50)
    
    # Create loader
    loader = RealDataLoader()
    
    # Load real data
    if not loader.load_real_data():
        print("❌ Failed to load real data!")
        return
    
    # Run analysis
    results = loader.run_analysis()
    
    if results:
        print("\n🎉 REAL DATA ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {results['total_pulsars']} pulsars")
        print("Check 04_Results/ for detailed results")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main()
