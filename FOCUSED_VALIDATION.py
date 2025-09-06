#!/usr/bin/env python3
"""
FOCUSED VALIDATION TEST
Get the complete validation results without the data loading noise
"""

import numpy as np
from ADVANCED_COSMIC_STRING_HUNTER import CosmicStringHunter

def generate_pure_noise_data(n_pulsars: int = 45, n_points: int = 1000) -> dict:
    """Generate pure white noise data"""
    residuals = {}
    for i in range(n_pulsars):
        pulsar_name = f"J{2000+i:04d}+0000"
        noise = np.random.normal(0, 1e-7, n_points)
        residuals[pulsar_name] = noise
    return residuals

def generate_red_noise_data(n_pulsars: int = 45, n_points: int = 1000) -> dict:
    """Generate realistic red noise (like real PTA data)"""
    residuals = {}
    for i in range(n_pulsars):
        pulsar_name = f"J{2000+i:04d}+0000"
        freqs = np.fft.fftfreq(n_points, d=1/30.0)[1:n_points//2]
        power = freqs**(-13/3)
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        spectrum = np.sqrt(power) * np.exp(1j * phases)
        full_spectrum = np.zeros(n_points, dtype=complex)
        full_spectrum[1:n_points//2] = spectrum
        full_spectrum[n_points//2+1:] = np.conj(spectrum[::-1])
        noise = np.fft.ifft(full_spectrum).real
        noise = (noise - np.mean(noise)) / np.std(noise) * 1e-7
        residuals[pulsar_name] = noise
    return residuals

def main():
    print("🚨 FOCUSED VALIDATION TEST")
    print("="*50)
    
    hunter = CosmicStringHunter()
    
    # Test 1: Pure white noise
    print("\n1️⃣ WHITE NOISE TEST:")
    white_data = generate_pure_noise_data()
    white_results = hunter.run_full_analysis(white_data)
    white_score = white_results['combined_significance']['total_score']
    print(f"   Score: {white_score}")
    
    # Test 2: Red noise
    print("\n2️⃣ RED NOISE TEST:")
    red_data = generate_red_noise_data()
    red_results = hunter.run_full_analysis(red_data)
    red_score = red_results['combined_significance']['total_score']
    print(f"   Score: {red_score}")
    
    # Test 3: String signals
    print("\n3️⃣ STRING SIGNALS TEST:")
    string_data = hunter.generate_validation_data(n_pulsars=45, inject_string=True)
    string_results = hunter.run_full_analysis(string_data)
    string_score = string_results['combined_significance']['total_score']
    print(f"   Score: {string_score}")
    
    # CRITICAL ANALYSIS
    print(f"\n🔍 CRITICAL ANALYSIS:")
    print(f"   White noise: {white_score}")
    print(f"   Red noise: {red_score}")
    print(f"   String signals: {string_score}")
    print(f"   Real data: 826")
    
    # VALIDATION LOGIC
    print(f"\n⚡ VALIDATION RESULTS:")
    
    if white_score < 10 and red_score < 100 and string_score > 500:
        print("   ✅ METHOD IS WORKING CORRECTLY!")
        print("   ✅ White noise gives low scores")
        print("   ✅ Red noise gives moderate scores") 
        print("   ✅ String signals give high scores")
        
        if 826 > red_score * 2:
            print("   🚨 REAL DATA SCORE IS SUSPICIOUSLY HIGH!")
            print("   🚨 This suggests either:")
            print("      - Major cosmic string discovery 🏆")
            print("      - Systematic effect in data parsing 🔧")
        else:
            print("   ✅ Real data score is reasonable")
            print("   ✅ This could be real cosmic string signatures!")
            
    elif white_score > 50 or red_score > 500:
        print("   ❌ METHOD HAS SYSTEMATIC ISSUES!")
        print("   ❌ Even noise is giving high scores")
        
    else:
        print("   ⚠️  MIXED RESULTS - Need deeper investigation")

if __name__ == "__main__":
    main()
