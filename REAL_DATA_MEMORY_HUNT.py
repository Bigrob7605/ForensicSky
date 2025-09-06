#!/usr/bin/env python3
"""
Test the gravitational wave memory hunter with real IPTA data
"""

from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter
from IPTA_TIMING_PARSER import load_ipta_timing_data

def hunt_real_memory_effects():
    """Hunt for gravitational wave memory effects in real IPTA data"""
    print("🌌 HUNTING COSMIC STRING MEMORY EFFECTS IN REAL IPTA DATA")
    print("="*70)
    
    hunter = GravitationalWaveMemoryHunter()
    
    # Load real IPTA data
    print("📡 Loading real IPTA DR2 data...")
    real_data = load_ipta_timing_data()
    
    print(f"✅ Loaded {len(real_data)} pulsars")
    
    # Run memory effect analysis
    print("\n🔍 Running gravitational wave memory effect analysis...")
    results = hunter.detect_memory_effects(real_data)
    
    print(f"\n📊 REAL DATA RESULTS:")
    print(f"   Step candidates: {len(results['step_candidates'])}")
    print(f"   Coincident events: {len(results['coincident_events'])}")
    print(f"   Memory effects: {len(results['memory_effects'])}")
    print(f"   Significance: {results['significance']:.2f}")
    
    if results['memory_effects']:
        print(f"\n🎯 COSMIC STRING MEMORY EFFECTS DETECTED!")
        for i, effect in enumerate(results['memory_effects']):
            print(f"   Effect {i+1}: {effect['n_pulsars']} pulsars")
            print(f"   Strain amplitude: {effect['strain_amplitude']:.2e}")
            print(f"   String tension: {effect['string_tension_estimate']:.2e}")
            print(f"   Time index: {effect['time_index']}")
    else:
        print(f"\n❌ No cosmic string memory effects detected in real data")
        
        # Show what we found
        if results['coincident_events']:
            print(f"\n🔍 COINCIDENT EVENTS FOUND (but filtered out):")
            for i, event in enumerate(results['coincident_events'][:3]):
                avg_amp = event['total_amplitude'] / event['n_pulsars']
                print(f"   Event {i+1}: {event['n_pulsars']} pulsars, avg_amp={avg_amp:.2e}")
                print(f"   In cosmic string range: {1e-8 <= abs(avg_amp) <= 1e-6}")

if __name__ == "__main__":
    hunt_real_memory_effects()
