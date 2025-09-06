#!/usr/bin/env python3
"""
RUN FULL QUANTUM SCAN
=====================

Execute the full quantum scan using the integrated Core Forensic Sky V1 engine.
This will load ALL pulsars and apply quantum analysis methods.
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add the core engine path
sys.path.append(str(Path(__file__).parent / "01_Core_Engine"))

from Core_ForensicSky_V1 import CoreForensicSkyV1

def main():
    """Run the full quantum scan"""
    print("🚀 FULL QUANTUM SCAN - INTEGRATED MAIN ENGINE")
    print("=" * 60)
    print("Loading ALL pulsars and applying quantum analysis!")
    print("The gold is inside scanning them all!")
    print()
    
    start_time = time.time()
    
    # Initialize the integrated engine
    print("🔧 Initializing integrated Core Forensic Sky V1 with quantum technology...")
    engine = CoreForensicSkyV1()
    
    if not engine.quantum_available:
        print("❌ Quantum technology not available!")
        return
    
    print("✅ Quantum technology integrated successfully!")
    print()
    
    # Run the full quantum scan
    print("🚀 Starting FULL QUANTUM SCAN...")
    print("This will load ALL pulsars and apply quantum analysis methods.")
    print("We will not miss a single pulsar!")
    print()
    
    results = engine.run_full_quantum_scan()
    
    if results:
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n🎯 FULL QUANTUM SCAN COMPLETE!")
        print("=" * 60)
        
        scan_stats = results['scan_stats']
        print(f"📊 SCAN STATISTICS:")
        print(f"   Total pulsars loaded: {scan_stats['total_pulsars_loaded']}")
        print(f"   Quantum analyses completed: {scan_stats['quantum_analyses_completed']}")
        print(f"   High correlation pairs: {scan_stats['high_correlation_pairs']}")
        print(f"   Cusp candidates: {scan_stats['cusp_candidates']}")
        print(f"   High coherence pulsars: {scan_stats['high_coherence_pulsars']}")
        print(f"   Analysis time: {scan_stats['analysis_time']:.2f} seconds")
        print(f"   Total scan duration: {duration:.2f} seconds")
        
        # Show quantum results summary
        if 'quantum_results' in results:
            quantum_summary = results['quantum_results']['summary']
            print(f"\n🧠 QUANTUM ANALYSIS SUMMARY:")
            print(f"   Total pulsars: {quantum_summary['total_pulsars']}")
            print(f"   High correlation pairs: {quantum_summary['high_correlation_pairs']}")
            print(f"   Cusp candidates: {quantum_summary['cusp_candidates']}")
            print(f"   High coherence pulsars: {quantum_summary['high_coherence_pulsars']}")
        
        # Show classical results summary
        if 'classical_results' in results:
            classical_stats = results['classical_results']['statistical_analysis']
            print(f"\n📈 CLASSICAL ANALYSIS SUMMARY:")
            print(f"   Total pulsars: {classical_stats['total_pulsars']}")
            print(f"   Sky coverage: {classical_stats['sky_coverage']:.4f}")
            if 'frequency_distribution' in classical_stats:
                freq_dist = classical_stats['frequency_distribution']
                print(f"   Frequency range: {freq_dist['min']:.2f} - {freq_dist['max']:.2f} Hz")
        
        print(f"\n💾 Results saved to: full_quantum_scan_results_*.json")
        print("\n🌌 Ready to analyze the results for cosmic string signatures!")
        
    else:
        print("❌ Full quantum scan failed!")
        print("Check the logs for details.")

if __name__ == "__main__":
    main()
