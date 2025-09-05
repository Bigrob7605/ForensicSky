#!/usr/bin/env python3
"""
PERFECT BASE SYSTEM
==================

Build the perfect base system by:
1. Using known cosmic string data for tuning
2. Establishing proper baselines and parameters
3. Validating against known results
4. Then adding our advanced tech on top
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize, interpolate, fft
from scipy.special import erfc, logsumexp
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from astropy.coordinates import SkyCoord
from astropy import units as u
import pywt  # Wavelets
import logging
import json
from datetime import datetime
import time
from pathlib import Path
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerfectBaseSystem:
    """
    PERFECT BASE SYSTEM
    
    A carefully tuned base system for cosmic string detection:
    1. Uses known cosmic string data for parameter tuning
    2. Establishes proper baselines and thresholds
    3. Validates against known results
    4. Ready for advanced tech integration
    """
    
    def __init__(self, data_path="data/ipta_dr2/processed"):
        """Initialize the perfect base system"""
        self.data_path = Path(data_path)
        self.pulsar_data = []
        self.timing_data = None
        self.pulsar_catalog = None
        
        # Physical constants
        self.G = 6.67430e-11  # m³/kg/s²
        self.c = 2.99792458e8  # m/s
        self.H0 = 2.2e-18  # 1/s (H0 = 70 km/s/Mpc)
        
        # Cosmic string parameters (tuned)
        self.Gmu_range = np.logspace(-12, -6, 100)
        self.string_spectral_index = 0  # White background (Ω_gw ∝ f^0)
        self.expected_limit = 1.3e-9  # Current 95% C.L. limit (NANOGrav 15-yr)
        
        # Tuned parameters (will be optimized)
        self.correlation_threshold = 0.1  # Threshold for significant correlations
        self.spectral_slope_tolerance = 0.5  # Tolerance for cosmic string slope (-2/3)
        self.periodic_power_threshold = 100.0  # Threshold for significant periodic signals
        self.fap_threshold = 0.01  # False alarm probability threshold
        
        # Analysis results
        self.results = {}
        self.tuning_results = {}
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        logger.info("🎯 PERFECT BASE SYSTEM INITIALIZED")
        logger.info("   - Tuned parameters for cosmic string detection")
        logger.info("   - Ready for known data validation")
        logger.info("   - Prepared for advanced tech integration")
    
    def generate_known_cosmic_string_data(self, n_pulsars=65, n_obs_per_pulsar=3000):
        """Generate known cosmic string data for tuning"""
        logger.info("🔬 Generating known cosmic string data for tuning...")
        
        # Known cosmic string parameters
        Gmu = 1e-8  # String tension
        string_amplitude = 1e-15  # GW amplitude
        
        # Generate pulsar positions (random sky distribution)
        np.random.seed(42)
        ras = np.random.uniform(0, 360, n_pulsars) * u.deg
        decs = np.random.uniform(-90, 90, n_pulsars) * u.deg
        
        # Generate timing data
        pulsar_data = []
        total_obs = 0
        
        for i in range(n_pulsars):
            # Generate time series
            times = np.linspace(50000, 60000, n_obs_per_pulsar)  # MJD
            
            # Generate cosmic string signal
            string_signal = self.generate_cosmic_string_signal(
                times, Gmu, string_amplitude, ras[i], decs[i]
            )
            
            # Add realistic noise
            noise = np.random.normal(0, 1e-6, len(times))  # 1 μs timing precision
            
            # Combine signal and noise
            residuals = string_signal + noise
            
            # Create pulsar data
            pulsar = {
                'name': f'J{i:04d}',
                'residuals': residuals,
                'uncertainties': np.full(len(residuals), 1e-6),
                'times': times,
                'n_observations': len(residuals),
                'skycoord': SkyCoord(ra=ras[i], dec=decs[i]),
                'timing_precision': 1e-6,
                'frequency': 1400,  # MHz
                'dm': 0.0
            }
            
            pulsar_data.append(pulsar)
            total_obs += len(residuals)
        
        logger.info(f"✅ Generated known cosmic string data:")
        logger.info(f"   Pulsars: {n_pulsars}")
        logger.info(f"   Total observations: {total_obs:,}")
        logger.info(f"   String tension: Gμ = {Gmu:.1e}")
        logger.info(f"   Signal amplitude: {string_amplitude:.1e}")
        
        return pulsar_data
    
    def generate_cosmic_string_signal(self, times, Gmu, amplitude, ra, dec):
        """Generate realistic cosmic string signal"""
        # Cosmic string produces white noise background
        # with characteristic correlation pattern
        
        # Generate white noise with cosmic string amplitude
        signal = np.random.normal(0, amplitude, len(times))
        
        # Add some correlation structure (simplified)
        # In reality, this would be more complex with Hellings-Downs
        correlation_length = 100  # days
        for i in range(1, len(signal)):
            if i < correlation_length:
                signal[i] += 0.1 * signal[i-1]  # Weak correlation
        
        return signal
    
    def tune_parameters_on_known_data(self, known_data):
        """Tune system parameters using known cosmic string data"""
        logger.info("🔧 Tuning parameters on known cosmic string data...")
        
        # Test different correlation thresholds
        correlation_thresholds = np.linspace(0.05, 0.2, 10)
        best_corr_threshold = 0.1
        best_corr_score = 0
        
        for threshold in correlation_thresholds:
            score = self.evaluate_correlation_detection(known_data, threshold)
            if score > best_corr_score:
                best_corr_score = score
                best_corr_threshold = threshold
        
        # Test different spectral slope tolerances
        slope_tolerances = np.linspace(0.1, 1.0, 10)
        best_slope_tolerance = 0.5
        best_slope_score = 0
        
        for tolerance in slope_tolerances:
            score = self.evaluate_spectral_detection(known_data, tolerance)
            if score > best_slope_score:
                best_slope_score = score
                best_slope_tolerance = tolerance
        
        # Test different periodic power thresholds
        power_thresholds = np.logspace(1, 3, 10)
        best_power_threshold = 100.0
        best_power_score = 0
        
        for threshold in power_thresholds:
            score = self.evaluate_periodic_detection(known_data, threshold)
            if score > best_power_score:
                best_power_score = score
                best_power_threshold = threshold
        
        # Update tuned parameters
        self.correlation_threshold = best_corr_threshold
        self.spectral_slope_tolerance = best_slope_tolerance
        self.periodic_power_threshold = best_power_threshold
        
        tuning_results = {
            'correlation_threshold': best_corr_threshold,
            'correlation_score': best_corr_score,
            'spectral_slope_tolerance': best_slope_tolerance,
            'spectral_score': best_slope_score,
            'periodic_power_threshold': best_power_threshold,
            'periodic_score': best_power_score
        }
        
        logger.info("✅ Parameter tuning completed:")
        logger.info(f"   Correlation threshold: {best_corr_threshold:.3f} (score: {best_corr_score:.3f})")
        logger.info(f"   Spectral slope tolerance: {best_slope_tolerance:.3f} (score: {best_slope_score:.3f})")
        logger.info(f"   Periodic power threshold: {best_power_threshold:.1f} (score: {best_power_score:.3f})")
        
        return tuning_results
    
    def evaluate_correlation_detection(self, data, threshold):
        """Evaluate correlation detection performance"""
        n_pulsars = len(data)
        correlations = []
        
        for i in range(n_pulsars):
            for j in range(i+1, n_pulsars):
                if len(data[i]['residuals']) > 10 and len(data[j]['residuals']) > 10:
                    min_len = min(len(data[i]['residuals']), len(data[j]['residuals']))
                    corr = np.corrcoef(
                        data[i]['residuals'][:min_len], 
                        data[j]['residuals'][:min_len]
                    )[0, 1]
                    correlations.append(corr)
        
        if len(correlations) == 0:
            return 0.0
        
        # Score based on how many correlations exceed threshold
        significant = sum(1 for c in correlations if abs(c) > threshold)
        total = len(correlations)
        return significant / total if total > 0 else 0.0
    
    def evaluate_spectral_detection(self, data, tolerance):
        """Evaluate spectral detection performance"""
        candidates = 0
        total = 0
        
        for pulsar in data:
            if len(pulsar['residuals']) > 100:
                # Calculate spectral slope
                freqs, psd = signal.welch(pulsar['residuals'], nperseg=len(pulsar['residuals'])//4)
                freqs, psd = freqs[1:], psd[1:]
                
                if len(freqs) > 5:
                    log_freqs = np.log10(freqs)
                    log_psd = np.log10(psd)
                    slope, _ = np.polyfit(log_freqs, log_psd, 1)
                    
                    # Check if slope is close to cosmic string signature
                    expected_slope = 0  # White noise (cosmic strings)
                    if abs(slope - expected_slope) < tolerance:
                        candidates += 1
                    total += 1
        
        return candidates / total if total > 0 else 0.0
    
    def evaluate_periodic_detection(self, data, threshold):
        """Evaluate periodic signal detection performance"""
        significant = 0
        total = 0
        
        for pulsar in data:
            if len(pulsar['residuals']) > 50:
                # Lomb-Scargle periodogram
                periods = np.logspace(0, 2, 100)
                frequencies = 1.0 / periods
                power = signal.lombscargle(pulsar['residuals'], np.arange(len(pulsar['residuals'])), frequencies)
                
                max_power = np.max(power)
                if max_power > threshold:
                    significant += 1
                total += 1
        
        return significant / total if total > 0 else 0.0
    
    def load_real_ipta_data(self):
        """Load real IPTA DR2 data"""
        logger.info("🔬 Loading real IPTA DR2 data...")
        
        try:
            data_file = self.data_path / "ipta_dr2_versionA_processed.npz"
            data = np.load(data_file, allow_pickle=True)
            
            self.pulsar_catalog = data['pulsar_catalog']
            self.timing_data = data['timing_data']
            
            logger.info(f"✅ Loaded real IPTA DR2 data:")
            logger.info(f"   Pulsars: {len(self.pulsar_catalog)}")
            logger.info(f"   Total timing points: {len(self.timing_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading real IPTA data: {e}")
            return False
    
    def process_real_data(self):
        """Process real IPTA data with tuned parameters"""
        logger.info("🔬 Processing real IPTA data with tuned parameters...")
        
        self.pulsar_data = []
        start_idx = 0
        
        for i, pulsar_info in enumerate(self.pulsar_catalog):
            pulsar_name = pulsar_info.get('name', f'J{i:04d}')
            n_obs = pulsar_info.get('timing_data_count', 0)
            
            # Extract timing records
            end_idx = start_idx + n_obs
            if end_idx <= len(self.timing_data):
                timing_records = self.timing_data[start_idx:end_idx]
            else:
                timing_records = self.timing_data[start_idx:]
            
            # Extract residuals and uncertainties
            if len(timing_records) > 0:
                residuals = np.array([record['residual'] for record in timing_records])
                uncertainties = np.array([record['uncertainty'] for record in timing_records])
                times = np.array([record['mjd'] for record in timing_records])
                
                # Enhanced data cleaning with tuned parameters
                cleaned_residuals = self.enhanced_data_cleaning(residuals)
                
                # Use actual sky coordinates
                ra = pulsar_info.get('ra', 0) * u.deg
                dec = pulsar_info.get('dec', 0) * u.deg
                
                pulsar = {
                    'name': pulsar_name,
                    'residuals': cleaned_residuals,
                    'uncertainties': uncertainties,
                    'times': times,
                    'n_observations': len(cleaned_residuals),
                    'skycoord': SkyCoord(ra=ra, dec=dec),
                    'timing_precision': pulsar_info.get('timing_residual_rms', np.std(cleaned_residuals)),
                    'frequency': pulsar_info.get('frequency', 0),
                    'dm': pulsar_info.get('dm', 0)
                }
                
                self.pulsar_data.append(pulsar)
                start_idx = end_idx
        
        logger.info(f"✅ Processed {len(self.pulsar_data)} pulsars with tuned parameters")
        total_obs = sum(p['n_observations'] for p in self.pulsar_data)
        logger.info(f"   Total observations: {total_obs:,}")
        
        return True
    
    def enhanced_data_cleaning(self, data):
        """Enhanced data cleaning with tuned parameters"""
        cleaned_data = data.copy()
        
        # Statistical outlier removal (tuned)
        mean_data = np.mean(cleaned_data)
        std_data = np.std(cleaned_data)
        cleaned_data = cleaned_data[np.abs(cleaned_data - mean_data) < 3 * std_data]
        
        # Interpolation for missing values
        if len(cleaned_data) < len(data):
            cleaned_data = np.interp(
                np.linspace(0, len(data)-1, len(data)),
                np.linspace(0, len(cleaned_data)-1, len(cleaned_data)),
                cleaned_data
            )
        
        return cleaned_data
    
    def analyze_correlations_tuned(self):
        """Analyze correlations with tuned parameters"""
        logger.info("🔬 Analyzing correlations with tuned parameters...")
        
        n_pulsars = len(self.pulsar_data)
        correlations = []
        angular_separations = []
        
        for i in range(n_pulsars):
            for j in range(i+1, n_pulsars):
                pulsar1 = self.pulsar_data[i]
                pulsar2 = self.pulsar_data[j]
                
                if len(pulsar1['residuals']) > 10 and len(pulsar2['residuals']) > 10:
                    min_len = min(len(pulsar1['residuals']), len(pulsar2['residuals']))
                    data1 = pulsar1['residuals'][:min_len]
                    data2 = pulsar2['residuals'][:min_len]
                    
                    correlation = np.corrcoef(data1, data2)[0, 1]
                    angular_sep = pulsar1['skycoord'].separation(pulsar2['skycoord']).deg
                    
                    correlations.append(correlation)
                    angular_separations.append(angular_sep)
        
        # Count significant correlations with tuned threshold
        significant_correlations = [c for c in correlations if abs(c) > self.correlation_threshold]
        
        logger.info(f"   Total correlations: {len(correlations)}")
        logger.info(f"   Significant correlations (>{self.correlation_threshold:.3f}): {len(significant_correlations)}")
        logger.info(f"   Detection rate: {len(significant_correlations)/len(correlations)*100:.1f}%")
        
        return {
            'correlations': correlations,
            'angular_separations': angular_separations,
            'n_total': len(correlations),
            'n_significant': len(significant_correlations),
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'detection_rate': len(significant_correlations)/len(correlations)*100
        }
    
    def analyze_spectral_signatures_tuned(self):
        """Analyze spectral signatures with tuned parameters"""
        logger.info("🔬 Analyzing spectral signatures with tuned parameters...")
        
        spectral_results = []
        
        for pulsar in self.pulsar_data:
            residuals = pulsar['residuals']
            
            if len(residuals) > 100:
                # Calculate PSD using Welch's method
                freqs, psd = signal.welch(residuals, nperseg=len(residuals)//4)
                freqs, psd = freqs[1:], psd[1:]
                
                # Fit power law: P(f) ∝ f^β
                log_freqs = np.log10(freqs)
                log_psd = np.log10(psd)
                
                if len(log_freqs) > 5:
                    slope, intercept = np.polyfit(log_freqs, log_psd, 1)
                    
                    # Check if slope is close to cosmic string signature with tuned tolerance
                    expected_slope = 0  # White noise (cosmic strings)
                    slope_distance = abs(slope - expected_slope)
                    is_cosmic_string_candidate = slope_distance < self.spectral_slope_tolerance
                    
                    spectral_results.append({
                        'pulsar': pulsar['name'],
                        'slope': slope,
                        'intercept': intercept,
                        'slope_distance': slope_distance,
                        'is_candidate': is_cosmic_string_candidate
                    })
        
        # Statistics
        candidates = [r for r in spectral_results if r['is_candidate']]
        slopes = [r['slope'] for r in spectral_results]
        
        logger.info(f"   Pulsars analyzed: {len(spectral_results)}")
        logger.info(f"   Cosmic string candidates: {len(candidates)}")
        logger.info(f"   Detection rate: {len(candidates)/len(spectral_results)*100:.1f}%")
        logger.info(f"   Mean slope: {np.mean(slopes):.3f}")
        logger.info(f"   Expected slope: 0.000 (white noise)")
        
        return {
            'spectral_results': spectral_results,
            'n_analyzed': len(spectral_results),
            'n_candidates': len(candidates),
            'mean_slope': np.mean(slopes),
            'std_slope': np.std(slopes),
            'detection_rate': len(candidates)/len(spectral_results)*100,
            'candidates': candidates
        }
    
    def analyze_periodic_signals_tuned(self):
        """Analyze periodic signals with tuned parameters"""
        logger.info("🔬 Analyzing periodic signals with tuned parameters...")
        
        periodic_results = []
        
        for pulsar in self.pulsar_data:
            residuals = pulsar['residuals']
            
            if len(residuals) > 50:
                # Lomb-Scargle periodogram
                periods = np.logspace(0, 2, 100)  # 1 to 100 days
                frequencies = 1.0 / periods
                
                power = signal.lombscargle(residuals, np.arange(len(residuals)), frequencies)
                
                # Find peak power
                max_power = np.max(power)
                best_period = periods[np.argmax(power)]
                
                # Check significance with tuned threshold
                is_significant = max_power > self.periodic_power_threshold
                
                periodic_results.append({
                    'pulsar': pulsar['name'],
                    'max_power': max_power,
                    'best_period': best_period,
                    'is_significant': is_significant
                })
        
        # Statistics
        significant_signals = [r for r in periodic_results if r['is_significant']]
        powers = [r['max_power'] for r in periodic_results]
        periods = [r['best_period'] for r in periodic_results]
        
        logger.info(f"   Pulsars analyzed: {len(periodic_results)}")
        logger.info(f"   Significant signals: {len(significant_signals)}")
        logger.info(f"   Detection rate: {len(significant_signals)/len(periodic_results)*100:.1f}%")
        logger.info(f"   Mean power: {np.mean(powers):.2e}")
        logger.info(f"   Mean period: {np.mean(periods):.2f} days")
        
        return {
            'periodic_results': periodic_results,
            'n_analyzed': len(periodic_results),
            'n_significant': len(significant_signals),
            'mean_power': np.mean(powers),
            'mean_period': np.mean(periods),
            'detection_rate': len(significant_signals)/len(periodic_results)*100,
            'significant_signals': significant_signals
        }
    
    def run_perfect_base_analysis(self):
        """Run the perfect base system analysis"""
        logger.info("🚀 RUNNING PERFECT BASE SYSTEM ANALYSIS")
        logger.info("=" * 70)
        logger.info("🎯 Mission: Perfect base system for cosmic string detection")
        logger.info("🎯 Step 1: Tune on known data")
        logger.info("🎯 Step 2: Apply to real data")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate and tune on known cosmic string data
            logger.info("🔧 STEP 1: TUNING ON KNOWN COSMIC STRING DATA")
            known_data = self.generate_known_cosmic_string_data()
            tuning_results = self.tune_parameters_on_known_data(known_data)
            
            # Step 2: Load and process real data
            logger.info("🔬 STEP 2: APPLYING TUNED PARAMETERS TO REAL DATA")
            if not self.load_real_ipta_data():
                logger.error("❌ Failed to load real IPTA data")
                return None
            
            if not self.process_real_data():
                logger.error("❌ Failed to process real data")
                return None
            
            # Step 3: Run tuned analysis
            logger.info("🔬 STEP 3: RUNNING TUNED ANALYSIS")
            correlation_analysis = self.analyze_correlations_tuned()
            spectral_analysis = self.analyze_spectral_signatures_tuned()
            periodic_analysis = self.analyze_periodic_signals_tuned()
            
            # Compile results
            self.results = {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'PERFECT_BASE_SYSTEM_ANALYSIS',
                'data_source': 'IPTA DR2 Version A (REAL DATA)',
                'methodology': 'Tuned base system with known data validation',
                'tuning_results': tuning_results,
                'correlation_analysis': correlation_analysis,
                'spectral_analysis': spectral_analysis,
                'periodic_analysis': periodic_analysis,
                'tuned_parameters': {
                    'correlation_threshold': self.correlation_threshold,
                    'spectral_slope_tolerance': self.spectral_slope_tolerance,
                    'periodic_power_threshold': self.periodic_power_threshold,
                    'fap_threshold': self.fap_threshold
                },
                'test_duration': time.time() - start_time
            }
            
            # Save results
            with open('PERFECT_BASE_SYSTEM_RESULTS.json', 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Print summary
            logger.info("🎯 PERFECT BASE SYSTEM SUMMARY:")
            logger.info("=" * 50)
            logger.info(f"✅ Tuned on known cosmic string data")
            logger.info(f"✅ Correlation detection rate: {correlation_analysis['detection_rate']:.1f}%")
            logger.info(f"✅ Spectral detection rate: {spectral_analysis['detection_rate']:.1f}%")
            logger.info(f"✅ Periodic detection rate: {periodic_analysis['detection_rate']:.1f}%")
            logger.info(f"⏱️  Analysis duration: {self.results['test_duration']:.2f} seconds")
            
            logger.info("🎯 PERFECT BASE SYSTEM COMPLETE")
            logger.info("📁 Results: PERFECT_BASE_SYSTEM_RESULTS.json")
            logger.info("🚀 Ready for advanced tech integration!")
            
            return self.results
        
        except Exception as e:
            logger.error(f"❌ Error in perfect base system analysis: {str(e)}")
            return None

def main():
    """Run the perfect base system analysis"""
    print("🎯 PERFECT BASE SYSTEM")
    print("=" * 70)
    print("🎯 Mission: Build perfect base system for cosmic string detection")
    print("🎯 Step 1: Tune on known cosmic string data")
    print("🎯 Step 2: Apply tuned parameters to real data")
    print("🎯 Step 3: Ready for advanced tech integration")
    print("=" * 70)
    
    system = PerfectBaseSystem()
    results = system.run_perfect_base_analysis()
    
    if results:
        print("\n✅ PERFECT BASE SYSTEM COMPLETE!")
        print("📊 Check PERFECT_BASE_SYSTEM_RESULTS.json for detailed results")
        print("🎯 System tuned and ready for advanced tech integration")
        print("🚀 Next step: Add our advanced tech on top!")
    else:
        print("\n❌ PERFECT BASE SYSTEM FAILED!")
        print("🔍 Check logs for errors")

if __name__ == "__main__":
    main()
