#!/usr/bin/env python3
"""
QUANTUM SMALL BATCH TEST
========================

Test the unified quantum cosmic string platform with a small batch of real pulsars
from IPTA DR2 data to validate the quantum analysis works correctly.
"""

import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add paths
sys.path.append(str(Path(__file__).parent / "01_Core_Engine"))
sys.path.append(str(Path(__file__).parent))

# Import our systems
from UNIFIED_QUANTUM_COSMIC_STRING_PLATFORM import UnifiedQuantumCosmicStringPlatform
from Core_ForensicSky_V1 import CoreForensicSkyV1

def load_small_batch_real_data():
    """Load a small batch of real IPTA DR2 pulsars for testing"""
    print("📡 Loading small batch of real IPTA DR2 data...")
    
    # Define a small test set of specific pulsars
    test_pulsar_names = [
        'J1909-3744',  # Known high-precision pulsar
        'J1713+0747',  # Another high-precision pulsar
        'J0437-4715'   # Southern sky pulsar
    ]
    
    print(f"🎯 Testing with predefined set: {test_pulsar_names}")
    
    # Initialize the core engine
    core_engine = CoreForensicSkyV1()
    
    # Load specific pulsars only
    try:
        # Load real data but limit to our test set
        print("🔄 Loading real IPTA DR2 data for test pulsars only...")
        
        # Load clock files first
        core_engine.load_clock_files()
        
        # Initialize data structures
        pulsar_catalog = []
        timing_data = []
        
        # Data path
        real_data_path = Path("02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master")
        
        if not real_data_path.exists():
            print("❌ Real IPTA DR2 data not found!")
            return []
        
        print("✅ Real IPTA DR2 data path confirmed")
        
        # Load only our test pulsars
        for pulsar_name in test_pulsar_names:
            print(f"🔍 Loading {pulsar_name}...")
            
            # Find par file for this pulsar
            par_files = list(real_data_path.glob(f"**/{pulsar_name}*.par"))
            if not par_files:
                print(f"  ⚠️ No par file found for {pulsar_name}")
                continue
            
            par_file = par_files[0]
            print(f"  📁 Found par file: {par_file}")
            
            # Load par file
            params = core_engine.load_par_file(par_file)
            if not params:
                print(f"  ⚠️ Failed to load par file for {pulsar_name}")
                continue
            
            # Find timing file
            tim_files = list(real_data_path.glob(f"**/{pulsar_name}*.tim"))
            if not tim_files:
                print(f"  ⚠️ No timing file found for {pulsar_name}")
                continue
            
            tim_file = tim_files[0]
            print(f"  📁 Found timing file: {tim_file}")
            
            # Load timing data
            times, residuals, uncertainties = core_engine.load_tim_file(tim_file)
            if len(times) == 0:
                print(f"  ⚠️ No timing data for {pulsar_name}")
                continue
            
            # Convert to GPU arrays
            times = core_engine._to_gpu_array(times)
            residuals = core_engine._to_gpu_array(residuals)
            uncertainties = core_engine._to_gpu_array(uncertainties)
            
            # Extract pulsar info
            ra = params.get('RAJ', 0.0)
            dec = params.get('DECJ', 0.0)
            
            # Convert coordinates
            if isinstance(ra, str):
                ra_parts = ra.split(':')
                if len(ra_parts) == 3:
                    ra = float(ra_parts[0]) + float(ra_parts[1])/60 + float(ra_parts[2])/3600
                else:
                    ra = float(ra)
            
            if isinstance(dec, str):
                dec_parts = dec.split(':')
                if len(dec_parts) == 3:
                    dec = float(dec_parts[0]) + float(dec_parts[1])/60 + float(dec_parts[2])/3600
                else:
                    dec = float(dec)
            
            ra_rad = np.radians(ra * 15.0)
            dec_rad = np.radians(dec)
            
            # Create pulsar info
            pulsar_info = {
                'name': pulsar_name,
                'ra': ra_rad,
                'dec': dec_rad,
                'timing_data_count': len(times),
                'timing_residual_rms': np.std(residuals),
                'frequency': float(params.get('F0', 1.0)),
                'dm': float(params.get('DM', 0.0)),
                'period': 1.0/float(params.get('F0', 1.0)) if float(params.get('F0', 1.0)) > 0 else 1.0,
                'period_derivative': float(params.get('F1', 0.0))
            }
            pulsar_catalog.append(pulsar_info)
            
            # Store timing data
            timing_entry = {
                'pulsar_name': pulsar_name,
                'times': core_engine._to_cpu_array(times),
                'residuals': core_engine._to_cpu_array(residuals),
                'uncertainties': core_engine._to_cpu_array(uncertainties)
            }
            timing_data.append(timing_entry)
            
            print(f"  ✅ Loaded {pulsar_name}: {len(times)} data points")
        
        print(f"📊 Successfully loaded {len(pulsar_catalog)} pulsars for quantum analysis")
        
        # Convert to format expected by quantum platform
        pulsar_data = []
        for i, pulsar in enumerate(pulsar_catalog):
            pulsar_data.append({
                'pulsar_id': pulsar['name'],
                'timing_data': {
                    'times': timing_data[i]['times'],
                    'residuals': timing_data[i]['residuals']
                }
            })
        
        return pulsar_data
        
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_quantum_analysis(pulsar_data):
    """Run quantum analysis on the pulsar data"""
    print("\n🚀 Starting Quantum Analysis...")
    
    # Initialize quantum platform
    platform = UnifiedQuantumCosmicStringPlatform()
    
    # Run unified analysis
    results = platform.run_unified_analysis(pulsar_data)
    
    return results

def analyze_quantum_results(results):
    """Analyze and display quantum results"""
    print("\n📊 QUANTUM ANALYSIS RESULTS")
    print("=" * 50)
    
    # Quantum states analysis
    print("\n🧠 QUANTUM STATES:")
    for pulsar_id, state in results['quantum_states'].items():
        print(f"  {pulsar_id}:")
        print(f"    Coherence: {state['coherence']:.4f}")
        print(f"    Dream Lucidity: {state['dream_lucidity']:.4f}")
        print(f"    Cosmic String Signature: {state['cosmic_string_signature']:.4f}")
        print(f"    Entanglement Count: {state['entanglement_count']}")
    
    # Correlations analysis
    print("\n🔗 QUANTUM CORRELATIONS:")
    high_correlations = 0
    for pair, corr in results['correlations'].items():
        if corr['correlation_strength'] > 0.5:
            high_correlations += 1
            print(f"  {pair}: {corr['correlation_strength']:.4f} (HIGH)")
        else:
            print(f"  {pair}: {corr['correlation_strength']:.4f}")
    
    # Cusp candidates
    print("\n🌌 CUSP CANDIDATES:")
    if results['cusp_candidates']:
        for pulsar_id, cusp in results['cusp_candidates'].items():
            print(f"  {pulsar_id}: {cusp['cusp_signature']:.4f} confidence")
    else:
        print("  No cusp candidates found")
    
    # Pattern analysis
    print("\n🧠 DREAM PATTERN ANALYSIS:")
    for pulsar_id, pattern in results['patterns'].items():
        print(f"  {pulsar_id}:")
        print(f"    Pattern Complexity: {pattern['pattern_complexity']:.4f}")
        print(f"    Cosmic String Signature: {pattern['cosmic_string_signature']:.4f}")
        print(f"    Dream Insights: {len(pattern['dream_insights'])} insights")
        if pattern['dream_insights']:
            for insight in pattern['dream_insights']:
                print(f"      - {insight}")
    
    # Summary
    print("\n📈 SUMMARY:")
    summary = results['summary']
    print(f"  Total Pulsars: {summary['total_pulsars']}")
    print(f"  High Correlation Pairs: {summary['high_correlation_pairs']}")
    print(f"  Cusp Candidates: {summary['cusp_candidates']}")
    print(f"  High Coherence Pulsars: {summary['high_coherence_pulsars']}")
    print(f"  Analysis Time: {summary['analysis_time']:.2f} seconds")
    
    return high_correlations > 0 or summary['cusp_candidates'] > 0

def main():
    """Main execution function"""
    print("🧪 QUANTUM SMALL BATCH TEST")
    print("=" * 50)
    print("Testing quantum cosmic string detection with real IPTA DR2 data")
    print()
    
    # Load small batch of real data
    pulsar_data = load_small_batch_real_data()
    
    if not pulsar_data:
        print("❌ No pulsar data available for testing")
        return
    
    # Run quantum analysis
    results = run_quantum_analysis(pulsar_data)
    
    # Analyze results
    has_interesting_results = analyze_quantum_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quantum_small_batch_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {filename}")
    
    if has_interesting_results:
        print("🎯 INTERESTING RESULTS DETECTED!")
        print("   Consider running full quantum scan")
    else:
        print("📊 Quantum analysis complete - no significant detections")
    
    print("\n✅ Small batch quantum test complete!")

if __name__ == "__main__":
    main()
