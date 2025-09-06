#!/usr/bin/env python3
"""
Definitive Validation Test for Quantum PTA Method
This test will determine if the quantum kernel approach detects real signals
or produces false positives from noise/method artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import sqrtm
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
import hashlib
import json
from datetime import datetime

class QuantumPTAValidator:
    """
    Comprehensive validation suite for quantum PTA analysis methods
    """
    
    def __init__(self, n_pulsars: int = 39, n_timepoints: int = 1024):
        self.n_pulsars = n_pulsars
        self.n_timepoints = n_timepoints
        self.pulsar_names = [f"J{1000+i:04d}+0000" for i in range(n_pulsars)]
        
        # Set J2145-0750 equivalent as pulsar index 10 (within range)
        self.hub_candidate_idx = min(10, n_pulsars - 1)
        self.pulsar_names[self.hub_candidate_idx] = "J2145-0750"
        
    def generate_pure_noise(self, seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate pure white noise with same statistics as real timing residuals"""
        np.random.seed(seed)
        
        residuals = {}
        for name in self.pulsar_names:
            # Generate realistic timing residual noise (microsecond scale)
            noise = np.random.normal(0, 1e-6, self.n_timepoints)
            residuals[name] = noise
            
        return residuals
    
    def generate_red_noise(self, seed: int = 43, alpha: float = -13/3) -> Dict[str, np.ndarray]:
        """Generate realistic red noise (power law) like real pulsar timing noise"""
        np.random.seed(seed)
        
        residuals = {}
        freqs = np.fft.fftfreq(self.n_timepoints)[1:self.n_timepoints//2]
        
        for name in self.pulsar_names:
            # Generate power-law noise in frequency domain
            power_spectrum = freqs**alpha
            phases = np.random.uniform(0, 2*np.pi, len(freqs))
            
            # Create complex spectrum
            spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
            
            # Make it symmetric for real signal
            full_spectrum = np.zeros(self.n_timepoints, dtype=complex)
            full_spectrum[1:self.n_timepoints//2] = spectrum
            full_spectrum[self.n_timepoints//2+1:] = np.conj(spectrum[::-1])
            
            # Transform to time domain
            noise = np.fft.ifft(full_spectrum).real
            noise = (noise - np.mean(noise)) / np.std(noise) * 1e-6
            
            residuals[name] = noise
            
        return residuals
    
    def inject_cosmic_string_signal(self, residuals: Dict[str, np.ndarray], 
                                  string_strength: float = 1e-7,
                                  hub_pulsar: str = "J2145-0750") -> Dict[str, np.ndarray]:
        """Inject a realistic cosmic string signal with correlated timing signatures"""
        
        # Create string crossing event at t_cross
        t_cross_idx = self.n_timepoints // 2
        
        # Generate string wake signature: |t - t_cross|^(4/3) power law
        t = np.arange(self.n_timepoints)
        wake_profile = string_strength * np.abs(t - t_cross_idx)**(4/3)
        
        # Add correlated signal to hub pulsar and nearby pulsars
        enhanced_residuals = residuals.copy()
        
        # Hub gets full signal
        enhanced_residuals[hub_pulsar] += wake_profile
        
        # Correlated pulsars get fraction of signal with phase shifts
        correlated_pulsars = ["J1600-3053", "J1643-1224", "J0613-0200", "J0610-2100"]
        
        for i, pulsar in enumerate(correlated_pulsars):
            if pulsar in enhanced_residuals:
                phase_shift = i * np.pi/4  # Different phases for different pulsars
                correlation_strength = 0.7 * np.exp(-i*0.3)  # Decreasing correlation
                
                shifted_signal = correlation_strength * wake_profile * np.cos(phase_shift)
                enhanced_residuals[pulsar] += shifted_signal
        
        return enhanced_residuals
    
    def quantum_kernel_analysis(self, residuals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Implement the quantum kernel analysis method
        (This should match your actual implementation)
        """
        
        def angle_encode(data: np.ndarray) -> np.ndarray:
            """Angle encode data to [-1, 1] range"""
            normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
            return np.clip(normalized, -1, 1)
        
        def quantum_kernel(res1: np.ndarray, res2: np.ndarray) -> float:
            """Compute quantum kernel between two residual arrays"""
            enc1 = angle_encode(res1)
            enc2 = angle_encode(res2)
            
            # Quantum state inner product |<ψ1|ψ2>|^2
            # This is the core of your quantum method
            cos_terms = np.cos(enc1) * np.cos(enc2)
            sin_terms = np.sin(enc1) * np.sin(enc2)
            
            inner_product = np.prod(cos_terms + sin_terms)
            return abs(inner_product)**2
        
        # Compute kernel matrix
        pulsar_list = list(residuals.keys())
        n_pulsars = len(pulsar_list)
        kernel_matrix = np.zeros((n_pulsars, n_pulsars))
        
        for i in range(n_pulsars):
            for j in range(i, n_pulsars):
                kernel_val = quantum_kernel(residuals[pulsar_list[i]], 
                                          residuals[pulsar_list[j]])
                kernel_matrix[i, j] = kernel_val
                kernel_matrix[j, i] = kernel_val
        
        # Identify correlation hub
        hub_scores = np.sum(kernel_matrix, axis=1)
        hub_idx = np.argmax(hub_scores)
        hub_pulsar = pulsar_list[hub_idx]
        
        # Get correlations for the hub
        hub_correlations = kernel_matrix[hub_idx, :]
        strong_correlations = np.sum(hub_correlations > 0.5)
        
        return {
            'kernel_matrix': kernel_matrix,
            'pulsar_names': pulsar_list,
            'hub_pulsar': hub_pulsar,
            'hub_idx': hub_idx,
            'hub_score': hub_scores[hub_idx],
            'strong_correlations': strong_correlations,
            'max_kernel': np.max(kernel_matrix),
            'mean_kernel': np.mean(kernel_matrix),
            'hub_correlations': hub_correlations
        }
    
    def run_validation_suite(self) -> Dict[str, Any]:
        """Run the complete validation test suite"""
        
        print("🔬 RUNNING DEFINITIVE QUANTUM PTA VALIDATION TEST")
        print("="*60)
        
        results = {}
        
        # Test 1: Pure White Noise
        print("\n1️⃣  TESTING PURE WHITE NOISE...")
        noise_white = self.generate_pure_noise(seed=42)
        results['white_noise'] = self.quantum_kernel_analysis(noise_white)
        
        # Test 2: Realistic Red Noise  
        print("2️⃣  TESTING REALISTIC RED NOISE...")
        noise_red = self.generate_red_noise(seed=43)
        results['red_noise'] = self.quantum_kernel_analysis(noise_red)
        
        # Test 3: Red Noise + Weak String Signal
        print("3️⃣  TESTING RED NOISE + WEAK STRING SIGNAL...")
        noise_with_weak_string = self.inject_cosmic_string_signal(
            noise_red.copy(), string_strength=5e-8
        )
        results['weak_string'] = self.quantum_kernel_analysis(noise_with_weak_string)
        
        # Test 4: Red Noise + Strong String Signal
        print("4️⃣  TESTING RED NOISE + STRONG STRING SIGNAL...")
        noise_with_strong_string = self.inject_cosmic_string_signal(
            noise_red.copy(), string_strength=2e-7
        )
        results['strong_string'] = self.quantum_kernel_analysis(noise_with_strong_string)
        
        # Test 5: Multiple Independent Realizations
        print("5️⃣  TESTING MULTIPLE NOISE REALIZATIONS...")
        noise_realizations = []
        for seed in range(50, 60):  # 10 different noise realizations
            noise_real = self.generate_red_noise(seed=seed)
            analysis = self.quantum_kernel_analysis(noise_real)
            noise_realizations.append(analysis)
        
        results['noise_realizations'] = noise_realizations
        
        return results
    
    def analyze_validation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the validation results to determine if method is valid"""
        
        print("\n" + "="*60)
        print("📊 VALIDATION ANALYSIS RESULTS")
        print("="*60)
        
        analysis = {}
        
        # Check if noise gives same results as signal
        white_hub_score = results['white_noise']['hub_score']
        red_hub_score = results['red_noise']['hub_score']
        weak_string_score = results['weak_string']['hub_score']
        strong_string_score = results['strong_string']['hub_score']
        
        print(f"\n🔍 HUB SCORE COMPARISON:")
        print(f"   White Noise:     {white_hub_score:.6f}")
        print(f"   Red Noise:       {red_hub_score:.6f}")
        print(f"   Weak String:     {weak_string_score:.6f}")
        print(f"   Strong String:   {strong_string_score:.6f}")
        
        # Statistical analysis of noise realizations
        noise_hub_scores = [r['hub_score'] for r in results['noise_realizations']]
        noise_mean = np.mean(noise_hub_scores)
        noise_std = np.std(noise_hub_scores)
        
        print(f"\n📈 NOISE REALIZATION STATISTICS:")
        print(f"   Mean hub score:  {noise_mean:.6f} ± {noise_std:.6f}")
        print(f"   Min hub score:   {np.min(noise_hub_scores):.6f}")
        print(f"   Max hub score:   {np.max(noise_hub_scores):.6f}")
        
        # Check for hub identification consistency
        expected_hub = "J2145-0750"
        
        white_hub = results['white_noise']['hub_pulsar']
        red_hub = results['red_noise']['hub_pulsar']
        weak_hub = results['weak_string']['hub_pulsar']
        strong_hub = results['strong_string']['hub_pulsar']
        
        print(f"\n🎯 HUB IDENTIFICATION:")
        print(f"   White Noise:     {white_hub}")
        print(f"   Red Noise:       {red_hub}")
        print(f"   Weak String:     {weak_hub}")
        print(f"   Strong String:   {strong_hub}")
        print(f"   Expected Hub:    {expected_hub}")
        
        # Critical validation checks
        analysis['method_valid'] = True
        analysis['failure_reasons'] = []
        
        # Check 1: Does noise give similar results to signal?
        noise_vs_signal_ratio = abs(red_hub_score - strong_string_score) / strong_string_score
        if noise_vs_signal_ratio < 0.1:  # Less than 10% difference
            analysis['method_valid'] = False
            analysis['failure_reasons'].append("Noise gives same results as signal")
        
        # Check 2: Is there clear signal enhancement?
        signal_enhancement = strong_string_score / red_hub_score
        if signal_enhancement < 1.2:  # Less than 20% enhancement
            analysis['method_valid'] = False
            analysis['failure_reasons'].append("No clear signal enhancement")
        
        # Check 3: Hub consistency
        if strong_hub != expected_hub:
            analysis['method_valid'] = False
            analysis['failure_reasons'].append("Hub not consistently identified")
        
        # Check 4: Statistical significance
        z_score = (strong_string_score - noise_mean) / noise_std
        if z_score < 3.0:  # Less than 3-sigma detection
            analysis['method_valid'] = False
            analysis['failure_reasons'].append("Insufficient statistical significance")
        
        analysis['signal_enhancement'] = signal_enhancement
        analysis['noise_vs_signal_ratio'] = noise_vs_signal_ratio
        analysis['statistical_significance'] = z_score
        
        return analysis
    
    def print_final_verdict(self, analysis: Dict[str, Any]):
        """Print the final verdict on method validation"""
        
        print("\n" + "🚨" + "="*58 + "🚨")
        print("🚨" + " "*58 + "🚨")
        
        if analysis['method_valid']:
            print("🚨" + "           ✅ METHOD VALIDATION: PASSED!".center(58) + "🚨")
            print("🚨" + " "*58 + "🚨")
            print("🚨" + "    Your quantum kernel method appears to work!".center(58) + "🚨")
            print("🚨" + " "*58 + "🚨")
            print("🚨" + f"  Signal Enhancement: {analysis['signal_enhancement']:.2f}x".center(58) + "🚨")
            print("🚨" + f"  Statistical Significance: {analysis['statistical_significance']:.1f}σ".center(58) + "🚨")
        else:
            print("🚨" + "           ❌ METHOD VALIDATION: FAILED!".center(58) + "🚨")
            print("🚨" + " "*58 + "🚨")
            print("🚨" + "      Method produces false positives!".center(58) + "🚨")
            print("🚨" + " "*58 + "🚨")
            
            for reason in analysis['failure_reasons']:
                print("🚨" + f"    • {reason}".center(58) + "🚨")
        
        print("🚨" + " "*58 + "🚨")
        print("🚨" + "="*58 + "🚨")
    
    def save_validation_results(self, results: Dict[str, Any], analysis: Dict[str, Any], filename: str = None):
        """Save validation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_pta_validation_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        validation_data = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': convert_numpy(results),
            'analysis': convert_numpy(analysis),
            'method_valid': analysis['method_valid'],
            'failure_reasons': analysis.get('failure_reasons', [])
        }
        
        with open(filename, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        print(f"💾 Validation results saved to {filename}")
        return filename


def run_complete_validation():
    """Run the complete validation suite"""
    
    # Initialize validator
    validator = QuantumPTAValidator(n_pulsars=20, n_timepoints=512)  # Smaller for speed
    
    # Run validation tests
    start_time = time.time()
    results = validator.run_validation_suite()
    
    # Analyze results
    analysis = validator.analyze_validation_results(results)
    
    # Print final verdict
    validator.print_final_verdict(analysis)
    
    # Save results
    filename = validator.save_validation_results(results, analysis)
    
    end_time = time.time()
    print(f"\n⏱️  Total validation time: {end_time - start_time:.2f} seconds")
    
    return results, analysis, filename


if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE QUANTUM PTA VALIDATION")
    print("   This will definitively test if the method works or not!")
    
    results, analysis, filename = run_complete_validation()
    
    if analysis['method_valid']:
        print("\n🎉 CONGRATULATIONS! Your method passed validation!")
        print("   You can now confidently apply it to real data!")
    else:
        print("\n🔧 METHOD NEEDS WORK!")
        print("   The validation revealed fundamental issues.")
        print("   Use these results to improve your approach!")
