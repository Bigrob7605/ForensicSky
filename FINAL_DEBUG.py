#!/usr/bin/env python3
"""
Final debug of the gravitational wave memory hunter
"""

import numpy as np
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter, generate_test_data_with_memory_effects

def final_debug():
    """Final debug of the entire pipeline"""
    print("🔍 FINAL DEBUG - GRAVITATIONAL WAVE MEMORY HUNTER")
    print("="*60)
    
    hunter = GravitationalWaveMemoryHunter()
    
    # Test 1: No memory effects
    print("\n1️⃣ NO MEMORY EFFECTS:")
    noise_data = generate_test_data_with_memory_effects(n_pulsars=5, n_points=100, inject_memory=False)
    noise_results = hunter.detect_memory_effects(noise_data)
    print(f"   Step candidates: {len(noise_results['step_candidates'])}")
    print(f"   Coincident events: {len(noise_results['coincident_events'])}")
    print(f"   Memory effects: {len(noise_results['memory_effects'])}")
    
    # Test 2: With memory effects
    print("\n2️⃣ WITH MEMORY EFFECTS:")
    memory_data = generate_test_data_with_memory_effects(n_pulsars=5, n_points=100, inject_memory=True)
    memory_results = hunter.detect_memory_effects(memory_data)
    print(f"   Step candidates: {len(memory_results['step_candidates'])}")
    print(f"   Coincident events: {len(memory_results['coincident_events'])}")
    print(f"   Memory effects: {len(memory_results['memory_effects'])}")
    
    # Debug the physics filter
    print(f"\n🔍 DEBUGGING PHYSICS FILTER:")
    for i, event in enumerate(memory_results['coincident_events']):
        avg_amplitude = event['total_amplitude'] / event['n_pulsars']
        print(f"   Event {i+1}: avg_amplitude = {avg_amplitude:.2e}")
        print(f"   Range check: 1e-12 <= {abs(avg_amplitude):.2e} <= 1e-9")
        in_range = 1e-12 <= abs(avg_amplitude) <= 1e-9
        print(f"   In range: {in_range}")
        
        if not in_range:
            if abs(avg_amplitude) < 1e-12:
                print(f"   ❌ Too small (need >= 1e-12)")
            else:
                print(f"   ❌ Too large (need <= 1e-9)")
    
    # Test with adjusted amplitude
    print(f"\n3️⃣ TESTING WITH ADJUSTED AMPLITUDE:")
    # Use amplitude in the right range
    adjusted_data = {}
    for i in range(5):
        pulsar_name = f"J{2000+i:04d}+0000"
        
        # Generate noise
        freqs = np.fft.fftfreq(100, d=1/30.0)[1:50]
        power = freqs**(-13/3)
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        spectrum = np.sqrt(power) * np.exp(1j * phases)
        full_spectrum = np.zeros(100, dtype=complex)
        full_spectrum[1:50] = spectrum
        full_spectrum[50:] = np.conj(spectrum[::-1])
        noise = np.fft.ifft(full_spectrum).real
        noise = (noise - np.mean(noise)) / np.std(noise) * 1e-7
        
        # Add memory effect with correct amplitude
        memory_amplitude = 5e-9  # In the right range
        noise[50:] += memory_amplitude
        
        adjusted_data[pulsar_name] = noise
    
    adjusted_results = hunter.detect_memory_effects(adjusted_data)
    print(f"   Step candidates: {len(adjusted_results['step_candidates'])}")
    print(f"   Coincident events: {len(adjusted_results['coincident_events'])}")
    print(f"   Memory effects: {len(adjusted_results['memory_effects'])}")
    
    if adjusted_results['memory_effects']:
        print(f"   ✅ SUCCESS! Found {len(adjusted_results['memory_effects'])} memory effects")
        for i, effect in enumerate(adjusted_results['memory_effects']):
            print(f"      Effect {i+1}: {effect['n_pulsars']} pulsars, "
                  f"strain = {effect['strain_amplitude']:.2e}")

if __name__ == "__main__":
    final_debug()
