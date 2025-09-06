#!/usr/bin/env python3
"""
RIGOROUS VALIDATION TEST - No Bullshit Analysis
==============================================

Let's find out if this is real or not. We're going to:
1. Test on synthetic data with known properties
2. Compare with classical PTA methods
3. Check for systematic errors
4. Validate the statistical significance properly
5. Test the methodology against established techniques

NO HYPE. JUST FACTS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from astropy.timeseries import LombScargle
from scipy import stats
import json
from datetime import datetime
import sys
import os
import glob

# Add the core engine to path
sys.path.append('01_Core_Engine')
from Core_ForensicSky_V1 import CoreForensicSkyV1

class RigorousValidationTest:
    def __init__(self):
        self.engine = CoreForensicSkyV1()
        self.target_pulsars = [
            'J2145-0750', 'J1600-3053', 'J1643-1224', 'J0613-0200'
        ]
        self.frequency_band = np.logspace(-9, -7, 50)
        self.results = {}
        
    def generate_synthetic_data(self, n_pulsars=4, n_points=2048, noise_level=1.0, 
                              include_string_signal=False, string_amplitude=0.1):
        """Generate synthetic timing residuals for testing"""
        print("🔬 Generating synthetic data for validation...")
        
        # Time grid (6 years)
        t = np.linspace(53432, 55618, n_points)  # MJD
        
        synthetic_data = {}
        
        for i, pulsar in enumerate(self.target_pulsars[:n_pulsars]):
            # Base red noise (random walk)
            red_noise = np.cumsum(np.random.normal(0, noise_level, n_points))
            
            # White noise
            white_noise = np.random.normal(0, noise_level * 0.1, n_points)
            
            # Combine
            residuals = red_noise + white_noise
            
            # Add cosmic string signal if requested
            if include_string_signal:
                # String signal: coherent across pulsars with phase shifts
                string_signal = string_amplitude * np.sin(2 * np.pi * t / 365.25 + i * np.pi/2)
                residuals += string_signal
            
            synthetic_data[pulsar] = {
                'times': t,
                'residuals': residuals,
                'uncertainties': np.ones(n_points) * noise_level * 0.1
            }
            
        return synthetic_data
        
    def load_real_data(self):
        """Load the real pulsar data we used before"""
        print("📊 Loading real pulsar data...")
        
        pulsar_data = {}
        for pulsar in self.target_pulsars:
            data = self.load_single_pulsar(pulsar)
            if data is not None:
                pulsar_data[pulsar] = data
                
        return pulsar_data
        
    def load_single_pulsar(self, pulsar_name):
        """Load a single pulsar's timing data"""
        # Search for par files
        par_files = []
        search_paths = [
            "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionB/*/",
            "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA/*/",
            "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionC/*/"
        ]
        
        for search_path in search_paths:
            par_files.extend(glob.glob(f"{search_path}*{pulsar_name}*.par"))
            
        if not par_files:
            return None
            
        par_file = par_files[0]
        
        # Load par file
        try:
            par_data = self.engine.load_par_file(par_file)
        except Exception as e:
            return None
            
        # Search for tim files
        tim_files = []
        tim_search_paths = [
            "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionB/*/tims/",
            "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA/*/tims/",
            "02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionC/*/tims/"
        ]
        
        for search_path in tim_search_paths:
            tim_files.extend(glob.glob(f"{search_path}*{pulsar_name}*.tim"))
            
        if not tim_files:
            return None
            
        tim_file = tim_files[0]
        
        # Load tim file
        try:
            times, residuals, uncertainties = self.engine.load_tim_file(tim_file)
            return {
                'times': times,
                'residuals': residuals,
                'uncertainties': uncertainties,
                'par_data': par_data
            }
        except Exception as e:
            return None
            
    def build_common_time_grid(self, pulsar_data):
        """Build a common time grid for all pulsars"""
        all_times = []
        all_residuals = []
        
        for pulsar, data in pulsar_data.items():
            if data is not None:
                all_times.append(data['times'])
                all_residuals.append(data['residuals'])
                
        if not all_times:
            raise ValueError("No valid pulsar data found!")
            
        # Find common time range
        t_min = max(times.min() for times in all_times)
        t_max = min(times.max() for times in all_times)
        
        # Create common grid with 2048 points
        t_common = np.linspace(t_min, t_max, 2048)
        
        # Interpolate all residuals to common grid
        residuals_common = []
        for times, res in zip(all_times, all_residuals):
            res_interp = np.interp(t_common, times, res)
            residuals_common.append(res_interp)
            
        return t_common, residuals_common
        
    def compute_phase_coherence(self, t_common, residuals_common):
        """Compute phase coherence using the same method as before"""
        # J2145-0750 is the reference (index 0)
        reference_residuals = residuals_common[0]
        phase_matrix = np.zeros((len(self.target_pulsars)-1, len(self.frequency_band)))
        
        for i, partner in enumerate(self.target_pulsars[1:], 1):
            partner_residuals = residuals_common[i]
            
            # Compute cross-spectral phase using Lomb-Scargle
            try:
                # Reference pulsar power spectrum
                ls_ref = LombScargle(t_common, reference_residuals)
                power_ref = ls_ref.power(self.frequency_band)
                
                # Partner pulsar power spectrum
                ls_partner = LombScargle(t_common, partner_residuals)
                power_partner = ls_partner.power(self.frequency_band)
                
                # Cross-spectral phase
                phase = np.angle(power_ref * np.conj(power_partner))
                phase_matrix[i-1] = phase
                
            except Exception as e:
                phase_matrix[i-1] = np.nan
                
        return phase_matrix
        
    def analyze_phase_coherence(self, phase_matrix):
        """Analyze phase coherence across pulsars"""
        # Remove NaN values
        valid_phases = phase_matrix[~np.isnan(phase_matrix).any(axis=1)]
        
        if len(valid_phases) == 0:
            return {
                'coherence_score': 0.0,
                'phase_std': np.full(len(self.frequency_band), np.nan),
                'phase_mean': np.full(len(self.frequency_band), np.nan),
                'coherent_frequencies': 0
            }
            
        # Phase statistics at each frequency
        phase_std = np.std(valid_phases, axis=0)
        phase_mean = np.mean(valid_phases, axis=0)
        
        # Coherence score: fraction of frequencies with phase_std < 0.09 rad (5°)
        coherent_mask = phase_std < 0.09
        coherence_score = np.mean(coherent_mask)
        coherent_frequencies = np.sum(coherent_mask)
        
        return {
            'coherence_score': coherence_score,
            'phase_std': phase_std,
            'phase_mean': phase_mean,
            'coherent_frequencies': coherent_frequencies,
            'phase_matrix': phase_matrix
        }
        
    def test_synthetic_data(self):
        """Test on synthetic data with known properties"""
        print("\n🧪 TEST 1: Synthetic Data Validation")
        print("=" * 50)
        
        # Test 1: Pure noise (should give low coherence)
        print("   Test 1a: Pure noise (should give low coherence)")
        noise_data = self.generate_synthetic_data(include_string_signal=False)
        t_common, residuals_common = self.build_common_time_grid(noise_data)
        phase_matrix = self.compute_phase_coherence(t_common, residuals_common)
        noise_result = self.analyze_phase_coherence(phase_matrix)
        
        print(f"     Coherence score: {noise_result['coherence_score']:.3f}")
        print(f"     Mean phase std: {np.mean(noise_result['phase_std']):.3f} rad")
        
        # Test 2: With string signal (should give high coherence)
        print("   Test 1b: With string signal (should give high coherence)")
        string_data = self.generate_synthetic_data(include_string_signal=True, string_amplitude=0.5)
        t_common, residuals_common = self.build_common_time_grid(string_data)
        phase_matrix = self.compute_phase_coherence(t_common, residuals_common)
        string_result = self.analyze_phase_coherence(phase_matrix)
        
        print(f"     Coherence score: {string_result['coherence_score']:.3f}")
        print(f"     Mean phase std: {np.mean(string_result['phase_std']):.3f} rad")
        
        return {
            'noise_result': noise_result,
            'string_result': string_result
        }
        
    def test_real_data(self):
        """Test on real data"""
        print("\n📊 TEST 2: Real Data Analysis")
        print("=" * 50)
        
        real_data = self.load_real_data()
        if len(real_data) < 2:
            print("   ❌ Insufficient real data for analysis")
            return None
            
        t_common, residuals_common = self.build_common_time_grid(real_data)
        phase_matrix = self.compute_phase_coherence(t_common, residuals_common)
        real_result = self.analyze_phase_coherence(phase_matrix)
        
        print(f"   Coherence score: {real_result['coherence_score']:.3f}")
        print(f"   Mean phase std: {np.mean(real_result['phase_std']):.3f} rad")
        print(f"   Coherent frequencies: {real_result['coherent_frequencies']}/{len(self.frequency_band)}")
        
        return real_result
        
    def test_statistical_significance(self, result):
        """Test the statistical significance properly"""
        print("\n📈 TEST 3: Statistical Significance Validation")
        print("=" * 50)
        
        if result is None:
            print("   ❌ No result to validate")
            return None
            
        coherence_score = result['coherence_score']
        n_frequencies = len(self.frequency_band)
        n_coherent = result['coherent_frequencies']
        
        # Binomial test: what's the probability of getting n_coherent out of n_frequencies by chance?
        # Assuming 5% chance per frequency (0.05)
        p_value = stats.binomtest(n_coherent, n_frequencies, 0.05, alternative='greater').pvalue
        
        # Z-score
        expected = n_frequencies * 0.05
        variance = n_frequencies * 0.05 * 0.95
        z_score = (n_coherent - expected) / np.sqrt(variance)
        
        print(f"   Coherent frequencies: {n_coherent}/{n_frequencies}")
        print(f"   Expected by chance: {expected:.1f}")
        print(f"   P-value (binomial): {p_value:.2e}")
        print(f"   Z-score: {z_score:.1f}")
        print(f"   Significance: {z_score:.1f}σ")
        
        return {
            'p_value': p_value,
            'z_score': z_score,
            'significance': z_score
        }
        
    def test_classical_comparison(self, real_data):
        """Compare with classical PTA methods"""
        print("\n🔬 TEST 4: Classical PTA Comparison")
        print("=" * 50)
        
        if real_data is None:
            print("   ❌ No real data for comparison")
            return None
            
        # Extract residuals
        residuals = []
        for pulsar in self.target_pulsars:
            if pulsar in real_data:
                residuals.append(real_data[pulsar]['residuals'])
                
        if len(residuals) < 2:
            print("   ❌ Insufficient data for classical comparison")
            return None
            
        # Classical cross-correlation
        correlations = []
        for i in range(len(residuals)):
            for j in range(i+1, len(residuals)):
                corr = np.corrcoef(residuals[i], residuals[j])[0,1]
                correlations.append(corr)
                
        mean_correlation = np.mean(correlations)
        max_correlation = np.max(correlations)
        
        print(f"   Mean cross-correlation: {mean_correlation:.3f}")
        print(f"   Max cross-correlation: {max_correlation:.3f}")
        print(f"   Number of pairs: {len(correlations)}")
        
        return {
            'mean_correlation': mean_correlation,
            'max_correlation': max_correlation,
            'correlations': correlations
        }
        
    def run_rigorous_validation(self):
        """Run the complete rigorous validation test"""
        print("🚀 RIGOROUS VALIDATION TEST - NO BULLSHIT ANALYSIS")
        print("=" * 70)
        print("Let's find out if this is real or not...")
        print()
        
        # Test 1: Synthetic data
        synthetic_results = self.test_synthetic_data()
        
        # Test 2: Real data
        real_data = self.load_real_data()
        real_result = self.test_real_data()
        
        # Test 3: Statistical significance
        if real_result is not None:
            stat_result = self.test_statistical_significance(real_result)
        else:
            stat_result = None
            
        # Test 4: Classical comparison
        classical_result = self.test_classical_comparison(real_data)
        
        # Store all results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'synthetic_results': synthetic_results,
            'real_result': real_result,
            'statistical_result': stat_result,
            'classical_result': classical_result
        }
        
        return self.results
        
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rigorous_validation_test_{timestamp}.json"
            
        # Convert numpy arrays to lists for JSON serialization
        results_copy = self.results.copy()
        
        # Handle synthetic results
        if 'synthetic_results' in results_copy:
            synth_copy = results_copy['synthetic_results'].copy()
            for key in ['noise_result', 'string_result']:
                if key in synth_copy:
                    synth_copy[key] = self._convert_numpy_arrays(synth_copy[key])
            results_copy['synthetic_results'] = synth_copy
            
        # Handle real result
        if 'real_result' in results_copy and results_copy['real_result'] is not None:
            results_copy['real_result'] = self._convert_numpy_arrays(results_copy['real_result'])
            
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
            
        print(f"💾 Results saved to {filename}")
        return filename
        
    def _convert_numpy_arrays(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

def main():
    """Run the rigorous validation test"""
    print("🔬 RIGOROUS VALIDATION TEST - NO BULLSHIT ANALYSIS")
    print("=" * 70)
    print("Time to find out if this is real or not...")
    print()
    
    # Run the validation
    validator = RigorousValidationTest()
    results = validator.run_rigorous_validation()
    
    # Save results
    filename = validator.save_results()
    
    print("\n" + "=" * 70)
    print("🎯 RIGOROUS VALIDATION COMPLETE!")
    print(f"📄 Results saved to: {filename}")
    
    # Summary
    if results['real_result'] is not None:
        coherence_score = results['real_result']['coherence_score']
        print(f"📊 Real data coherence score: {coherence_score:.3f}")
        
        if results['statistical_result'] is not None:
            significance = results['statistical_result']['significance']
            print(f"📈 Statistical significance: {significance:.1f}σ")
            
        if results['classical_result'] is not None:
            mean_corr = results['classical_result']['mean_correlation']
            print(f"🔬 Mean classical correlation: {mean_corr:.3f}")
            
    print("\n🔍 Now let's see what the data actually tells us...")
    
    return results

if __name__ == "__main__":
    main()
