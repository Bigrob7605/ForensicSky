#!/usr/bin/env python3
"""
Debug the gravitational wave memory hunter
"""

import numpy as np
import matplotlib.pyplot as plt
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter, generate_test_data_with_memory_effects

def debug_step_detection():
    """Debug the step detection algorithm"""
    print("🔍 DEBUGGING STEP DETECTION")
    print("="*40)
    
    # Generate test data with clear step
    data = generate_test_data_with_memory_effects(n_pulsars=1, n_points=100, inject_memory=True)
    pulsar_data = list(data.values())[0]
    
    print(f"Data shape: {pulsar_data.shape}")
    print(f"Data range: {pulsar_data.min():.2e} to {pulsar_data.max():.2e}")
    print(f"Data std: {pulsar_data.std():.2e}")
    
    # Look for the step manually
    step_time = 50  # We know it's at index 50
    before = pulsar_data[:step_time]
    after = pulsar_data[step_time:]
    
    print(f"\nStep analysis:")
    print(f"Before mean: {np.mean(before):.2e}")
    print(f"After mean: {np.mean(after):.2e}")
    print(f"Step amplitude: {np.mean(after) - np.mean(before):.2e}")
    
    # Test the hunter's step detection
    hunter = GravitationalWaveMemoryHunter()
    steps = hunter._find_step_changes(pulsar_data, "test_pulsar")
    
    print(f"\nHunter found {len(steps)} steps")
    for i, step in enumerate(steps):
        print(f"  Step {i+1}: time={step['time_index']}, amp={step['amplitude']:.2e}, snr={step['snr']:.2f}")
    
    # Plot the data to visualize
    plt.figure(figsize=(12, 6))
    plt.plot(pulsar_data, 'b-', alpha=0.7, label='Data')
    plt.axvline(step_time, color='r', linestyle='--', label='True step time')
    plt.xlabel('Time index')
    plt.ylabel('Residual')
    plt.title('Test Data with Memory Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('debug_memory_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return steps

def test_improved_step_detection():
    """Test an improved step detection algorithm"""
    print("\n🔧 TESTING IMPROVED STEP DETECTION")
    print("="*40)
    
    # Generate test data
    data = generate_test_data_with_memory_effects(n_pulsars=1, n_points=100, inject_memory=True)
    pulsar_data = list(data.values())[0]
    
    # Improved step detection
    def find_steps_improved(data, min_step_size=2.0):
        """Improved step detection using sliding window"""
        steps = []
        window_size = max(5, len(data) // 20)
        
        for i in range(window_size, len(data) - window_size):
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            # Calculate step amplitude
            step_amp = np.mean(after) - np.mean(before)
            
            # Calculate noise level
            noise_level = np.std(data)
            
            # Check if step is significant
            if abs(step_amp) > min_step_size * noise_level:
                # Calculate significance
                t_stat, p_value = stats.ttest_ind(before, after)
                significance = -np.log10(max(p_value, 1e-10))
                
                steps.append({
                    'time_index': i,
                    'amplitude': step_amp,
                    'significance': significance,
                    'snr': abs(step_amp) / noise_level
                })
        
        return steps
    
    from scipy import stats
    steps = find_steps_improved(pulsar_data)
    
    print(f"Improved method found {len(steps)} steps")
    for i, step in enumerate(steps):
        print(f"  Step {i+1}: time={step['time_index']}, amp={step['amplitude']:.2e}, snr={step['snr']:.2f}")

if __name__ == "__main__":
    debug_step_detection()
    test_improved_step_detection()
