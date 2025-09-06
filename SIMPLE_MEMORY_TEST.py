#!/usr/bin/env python3
"""
Simple test of the gravitational wave memory hunter with correct amplitudes
"""

import numpy as np
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter

def simple_test():
    """Simple test with correct amplitudes"""
    print("🧪 SIMPLE MEMORY EFFECT TEST")
    print("="*40)
    
    hunter = GravitationalWaveMemoryHunter()
    
    # Create simple test data with memory effects
    data = {}
    memory_amplitude = 5e-9  # In the correct range (1e-12 to 1e-9)
    memory_time = 50
    
    for i in range(5):
        pulsar_name = f"J{2000+i:04d}+0000"
        
        # Generate simple noise
        noise = np.random.normal(0, 1e-7, 100)
        
        # Add memory effect (step change)
        noise[memory_time:] += memory_amplitude
        
        data[pulsar_name] = noise
    
    print(f"Memory amplitude: {memory_amplitude:.2e}")
    print(f"Expected range: 1e-12 to 1e-9")
    print(f"In range: {1e-12 <= memory_amplitude <= 1e-9}")
    
    # Run analysis
    results = hunter.detect_memory_effects(data)
    
    print(f"\n📊 RESULTS:")
    print(f"   Step candidates: {len(results['step_candidates'])}")
    print(f"   Coincident events: {len(results['coincident_events'])}")
    print(f"   Memory effects: {len(results['memory_effects'])}")
    print(f"   Significance: {results['significance']:.2f}")
    
    if results['memory_effects']:
        print(f"\n✅ SUCCESS! Found cosmic string memory effects!")
        for i, effect in enumerate(results['memory_effects']):
            print(f"   Effect {i+1}: {effect['n_pulsars']} pulsars")
            print(f"   Strain amplitude: {effect['strain_amplitude']:.2e}")
            print(f"   String tension: {effect['string_tension_estimate']:.2e}")
    else:
        print(f"\n❌ No memory effects found")
        
        # Debug why
        if results['coincident_events']:
            print(f"   Coincident events found but filtered out:")
            for i, event in enumerate(results['coincident_events']):
                avg_amp = event['total_amplitude'] / event['n_pulsars']
                print(f"     Event {i+1}: avg_amplitude = {avg_amp:.2e}")
                print(f"     In range: {1e-12 <= abs(avg_amp) <= 1e-9}")

if __name__ == "__main__":
    simple_test()
