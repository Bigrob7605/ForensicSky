#!/usr/bin/env python3
"""
Debug the cosmic string physics filter
"""

import numpy as np
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter, generate_test_data_with_memory_effects

def debug_physics_filter():
    """Debug the cosmic string physics filtering"""
    print("🔍 DEBUGGING COSMIC STRING PHYSICS FILTER")
    print("="*50)
    
    # Generate test data
    hunter = GravitationalWaveMemoryHunter()
    data = generate_test_data_with_memory_effects(n_pulsars=5, n_points=100, inject_memory=True)
    
    # Run analysis
    results = hunter.detect_memory_effects(data)
    
    print(f"Step candidates: {len(results['step_candidates'])}")
    print(f"Coincident events: {len(results['coincident_events'])}")
    
    # Debug the physics filter
    for i, event in enumerate(results['coincident_events'][:3]):
        print(f"\nEvent {i+1}:")
        print(f"  N pulsars: {event['n_pulsars']}")
        print(f"  Total amplitude: {event['total_amplitude']:.2e}")
        print(f"  Average amplitude: {event['total_amplitude'] / event['n_pulsars']:.2e}")
        
        # Check strain conversion
        avg_amplitude = event['total_amplitude'] / event['n_pulsars']
        strain_amplitude = avg_amplitude / 1e-7  # Conversion factor
        print(f"  Strain amplitude: {strain_amplitude:.2e}")
        
        # Check if within expected range
        in_range = 1e-15 <= abs(strain_amplitude) <= 1e-12
        print(f"  In cosmic string range: {in_range}")
        print(f"  Range: 1e-15 to 1e-12")
        
        if not in_range:
            print(f"  ❌ Outside expected range for cosmic strings")
        else:
            print(f"  ✅ Within expected range for cosmic strings")

def test_with_realistic_amplitudes():
    """Test with more realistic cosmic string amplitudes"""
    print("\n🔧 TESTING WITH REALISTIC AMPLITUDES")
    print("="*50)
    
    # Generate data with realistic cosmic string memory effect amplitude
    # Cosmic string memory effects are typically 10^-15 to 10^-12 in strain
    # Convert to timing residuals: strain * 1e-7 (rough conversion)
    realistic_amplitude = 1e-12 * 1e-7  # 1e-19 in timing units
    
    print(f"Realistic amplitude: {realistic_amplitude:.2e}")
    print(f"This is much smaller than our test amplitude (5e-7)")
    print(f"Need to adjust the physics filter for realistic amplitudes")

if __name__ == "__main__":
    debug_physics_filter()
    test_with_realistic_amplitudes()
