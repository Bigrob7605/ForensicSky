#!/usr/bin/env python3
"""
Working test of the gravitational wave memory hunter
"""

import numpy as np
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter

def working_test():
    """Working test with proper parameters"""
    print("🎯 WORKING MEMORY EFFECT TEST")
    print("="*40)
    
    hunter = GravitationalWaveMemoryHunter()
    
    # Create test data with memory effects at the same time
    data = {}
    memory_amplitude = 2e-7  # In the correct range
    memory_time = 50
    
    print(f"Creating test data with memory effects...")
    print(f"Memory amplitude: {memory_amplitude:.2e}")
    print(f"Memory time: {memory_time}")
    
    for i in range(5):
        pulsar_name = f"J{2000+i:04d}+0000"
        
        # Generate noise
        noise = np.random.normal(0, 1e-7, 100)
        
        # Add memory effect (step change) at the same time for all pulsars
        noise[memory_time:] += memory_amplitude
        
        data[pulsar_name] = noise
    
    # Run analysis
    results = hunter.detect_memory_effects(data)
    
    print(f"\n📊 RESULTS:")
    print(f"   Step candidates: {len(results['step_candidates'])}")
    print(f"   Coincident events: {len(results['coincident_events'])}")
    print(f"   Memory effects: {len(results['memory_effects'])}")
    print(f"   Significance: {results['significance']:.2f}")
    
    # Show step candidates
    if results['step_candidates']:
        print(f"\n🔍 STEP CANDIDATES:")
        for i, step in enumerate(results['step_candidates'][:5]):
            print(f"   {step['pulsar']}: time={step['time_index']}, amp={step['amplitude']:.2e}, snr={step['snr']:.2f}")
    
    # Show coincident events
    if results['coincident_events']:
        print(f"\n🔗 COINCIDENT EVENTS:")
        for i, event in enumerate(results['coincident_events']):
            print(f"   Event {i+1}: {event['n_pulsars']} pulsars, avg_amp={event['total_amplitude']/event['n_pulsars']:.2e}")
    
    # Show memory effects
    if results['memory_effects']:
        print(f"\n✅ COSMIC STRING MEMORY EFFECTS FOUND!")
        for i, effect in enumerate(results['memory_effects']):
            print(f"   Effect {i+1}: {effect['n_pulsars']} pulsars")
            print(f"   Strain amplitude: {effect['strain_amplitude']:.2e}")
            print(f"   String tension: {effect['string_tension_estimate']:.2e}")
    else:
        print(f"\n❌ No cosmic string memory effects found")
        
        # Debug the physics filter
        if results['coincident_events']:
            print(f"\n🔍 DEBUGGING PHYSICS FILTER:")
            for i, event in enumerate(results['coincident_events']):
                avg_amp = event['total_amplitude'] / event['n_pulsars']
                print(f"   Event {i+1}: avg_amplitude = {avg_amp:.2e}")
                print(f"   Range check: 1e-8 <= {abs(avg_amp):.2e} <= 1e-6")
                in_range = 1e-8 <= abs(avg_amp) <= 1e-6
                print(f"   In range: {in_range}")

if __name__ == "__main__":
    working_test()
