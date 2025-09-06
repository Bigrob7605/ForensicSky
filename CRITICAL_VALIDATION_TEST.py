#!/usr/bin/env python3
"""
CRITICAL VALIDATION TEST
Test the cosmic string hunter with pure synthetic noise to validate the method
This is the make-or-break test to see if we have real signals or systematic effects
"""

import numpy as np
from ADVANCED_COSMIC_STRING_HUNTER import CosmicStringHunter
import matplotlib.pyplot as plt

def generate_pure_noise_data(n_pulsars: int = 45, n_points: int = 1000) -> dict:
    """Generate pure white noise data with same structure as real IPTA data"""
    residuals = {}
    
    for i in range(n_pulsars):
        pulsar_name = f"J{2000+i:04d}+0000"
        
        # Pure white noise - no cosmic string signals
        noise = np.random.normal(0, 1e-7, n_points)
        residuals[pulsar_name] = noise
        
    return residuals

def generate_red_noise_data(n_pulsars: int = 45, n_points: int = 1000) -> dict:
    """Generate realistic red noise (like real PTA data) but no cosmic strings"""
    residuals = {}
    
    for i in range(n_pulsars):
        pulsar_name = f"J{2000+i:04d}+0000"
        
        # Generate realistic red noise (like real PTA data)
        freqs = np.fft.fftfreq(n_points, d=1/30.0)[1:n_points//2]  # 30-day cadence
        power = freqs**(-13/3)  # Red noise power law
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        
        spectrum = np.sqrt(power) * np.exp(1j * phases)
        full_spectrum = np.zeros(n_points, dtype=complex)
        full_spectrum[1:n_points//2] = spectrum
        full_spectrum[n_points//2+1:] = np.conj(spectrum[::-1])
        
        noise = np.fft.ifft(full_spectrum).real
        noise = (noise - np.mean(noise)) / np.std(noise) * 1e-7
        
        residuals[pulsar_name] = noise
        
    return residuals

def run_critical_validation():
    """Run the critical validation test"""
    print("🚨 CRITICAL VALIDATION TEST - MAKE OR BREAK")
    print("="*60)
    
    hunter = CosmicStringHunter()
    
    # Test 1: Pure white noise (should give very low scores)
    print("\n1️⃣ Testing with PURE WHITE NOISE...")
    white_noise_data = generate_pure_noise_data()
    white_results = hunter.run_full_analysis(white_noise_data)
    white_score = white_results['combined_significance']['total_score']
    print(f"   White noise score: {white_score}")
    
    # Test 2: Red noise (like real PTA data, should give moderate scores)
    print("\n2️⃣ Testing with RED NOISE (like real PTA data)...")
    red_noise_data = generate_red_noise_data()
    red_results = hunter.run_full_analysis(red_noise_data)
    red_score = red_results['combined_significance']['total_score']
    print(f"   Red noise score: {red_score}")
    
    # Test 3: Red noise + cosmic string signals (should give high scores)
    print("\n3️⃣ Testing with RED NOISE + COSMIC STRING SIGNALS...")
    string_data = hunter.generate_validation_data(n_pulsars=45, inject_string=True)
    string_results = hunter.run_full_analysis(string_data)
    string_score = string_results['combined_significance']['total_score']
    print(f"   String signal score: {string_score}")
    
    # CRITICAL ANALYSIS
    print(f"\n🔍 CRITICAL ANALYSIS:")
    print(f"   White noise score: {white_score}")
    print(f"   Red noise score: {red_score}")
    print(f"   String signal score: {string_score}")
    print(f"   Real data score: 826 (from previous run)")
    
    # VALIDATION LOGIC
    print(f"\n⚡ VALIDATION RESULTS:")
    
    if white_score < 10 and red_score < 100 and string_score > 500:
        print("   ✅ METHOD IS WORKING CORRECTLY!")
        print("   ✅ White noise gives low scores")
        print("   ✅ Red noise gives moderate scores") 
        print("   ✅ String signals give high scores")
        
        if 826 > red_score * 2:  # Real data significantly higher than red noise
            print("   🚨 REAL DATA SCORE IS SUSPICIOUSLY HIGH!")
            print("   🚨 This suggests either:")
            print("      - Major cosmic string discovery 🏆")
            print("      - Systematic effect in data parsing 🔧")
            print("   🚨 NEEDS IMMEDIATE INVESTIGATION!")
        else:
            print("   ✅ Real data score is reasonable compared to red noise")
            print("   ✅ This could be real cosmic string signatures!")
            
    elif white_score > 50 or red_score > 500:
        print("   ❌ METHOD HAS SYSTEMATIC ISSUES!")
        print("   ❌ Even noise is giving high scores")
        print("   ❌ Need to fix the detection logic")
        
    else:
        print("   ⚠️  MIXED RESULTS - Need deeper investigation")
        
    return {
        'white_score': white_score,
        'red_score': red_score, 
        'string_score': string_score,
        'real_data_score': 826
    }

def investigate_real_data_periods():
    """Investigate the periods detected in real data"""
    print(f"\n🔬 INVESTIGATING REAL DATA PERIODS...")
    
    # Load real data
    from IPTA_TIMING_PARSER import load_ipta_timing_data
    real_data = load_ipta_timing_data()
    
    hunter = CosmicStringHunter()
    results = hunter.run_full_analysis(real_data)
    
    # Check kink radiation periods
    if 'kink_radiation' in results:
        periodic_signals = results['kink_radiation'].get('periodic_signals', [])
        print(f"   Found {len(periodic_signals)} periodic signals")
        
        for i, signal in enumerate(periodic_signals[:5]):  # Show first 5
            period = signal.get('period_years', 0)
            snr = signal.get('snr', 0)
            print(f"   Signal {i+1}: Period = {period:.2f} years, SNR = {snr:.2f}")
            
            # Check if period is realistic for cosmic string loops
            if 0.1 < period < 100:
                print(f"      ✅ Realistic period for cosmic string loops")
            else:
                print(f"      ❌ Unrealistic period for cosmic string loops")

if __name__ == "__main__":
    print("🚀 CRITICAL VALIDATION TEST - COSMIC STRING HUNTER")
    print("This test will determine if we have real signals or systematic effects")
    print("="*70)
    
    # Run critical validation
    validation_results = run_critical_validation()
    
    # Investigate real data periods
    investigate_real_data_periods()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. If method is working: Investigate real data signatures")
    print(f"   2. If method has issues: Fix detection logic")
    print(f"   3. Either way: This is valuable research infrastructure!")
