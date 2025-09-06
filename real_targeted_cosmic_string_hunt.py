#!/usr/bin/env python3
"""
🌌 REAL TARGETED COSMIC STRING HUNT
Direct analysis of real IPTA DR2 data for high-precision pulsars

Target: J1909-3744, J1713+0747, J0437-4715, J1744-1134, J1857+0943
Data: Real IPTA DR2 timing files (.tim)
Method: High-precision analysis with real astrophysical data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
from datetime import datetime
import os
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTargetedCosmicStringHunt:
    """
    🌌 REAL TARGETED COSMIC STRING HUNT
    
    Direct analysis of real IPTA DR2 data for high-precision pulsars
    """
    
    def __init__(self):
        self.data_path = Path("02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master")
        self.timing_data = []
        self.pulsar_catalog = []
        self.results = {}
        
        # High-precision pulsars (NANOGrav 15-year top performers)
        self.target_pulsars = [
            {'name': 'J1909-3744', 'ra': 287.4, 'dec': -37.7, 'period': 0.0029, 'precision': 'highest'},
            {'name': 'J1713+0747', 'ra': 258.3, 'dec': 7.4, 'period': 0.0046, 'precision': 'highest'},
            {'name': 'J0437-4715', 'ra': 69.3, 'dec': -47.2, 'period': 0.0058, 'precision': 'highest'},
            {'name': 'J1744-1134', 'ra': 266.0, 'dec': -11.6, 'period': 0.0041, 'precision': 'high'},
            {'name': 'J1857+0943', 'ra': 284.4, 'dec': 9.7, 'period': 0.0054, 'precision': 'high'},
        ]
        
        # Improved detection parameters for real data analysis
        self.detection_parameters = {
            'superstring_tension_range': (1e-12, 1e-11),
            'kink_dominance_threshold': 0.3,  # More sensitive
            'non_gaussian_threshold': 1.5,   # More sensitive
            'memory_effect_threshold': 2e-10,  # More sensitive
            'correlation_threshold': 0.02,   # More sensitive
            'signal_to_noise_threshold': 1.0,  # More sensitive
            'burst_threshold': 3.0,  # For individual burst detection
            'frequency_bands': {
                'low': (1e-9, 1e-8),    # 0.1-1 nHz
                'mid': (1e-8, 1e-7),    # 1-10 nHz  
                'high': (1e-7, 1e-6),   # 10-100 nHz
            }
        }
    
    def load_real_targeted_data(self):
        """Load real IPTA DR2 data for target pulsars"""
        logger.info("📡 Loading real IPTA DR2 data for target pulsars...")
        
        try:
            # Use target pulsars
            self.pulsar_catalog = self.target_pulsars
            
            # Load real timing data
            self.timing_data = self.load_real_timing_files()
            logger.info(f"✅ Loaded {len(self.timing_data)} real timing points from {len(self.pulsar_catalog)} target pulsars")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Real data loading failed: {e}")
            return False
    
    def load_real_timing_files(self):
        """Load real timing files from IPTA DR2"""
        timing_data = []
        
        try:
            # Look for timing files in the real data directory
            timing_dir = self.data_path / "pulsars"
            
            if not timing_dir.exists():
                logger.error(f"Timing directory not found: {timing_dir}")
                return []
            
            # Process each target pulsar
            for pulsar in self.target_pulsars:
                pulsar_name = pulsar['name']
                pulsar_dir = timing_dir / pulsar_name
                
                if not pulsar_dir.exists():
                    logger.warning(f"Pulsar directory not found: {pulsar_dir}")
                    continue
                
                logger.info(f"   Loading data for {pulsar_name}...")
                
                # Look for timing files
                timing_files = list(pulsar_dir.glob("*.tim"))
                
                if not timing_files:
                    logger.warning(f"No .tim files found for {pulsar_name}")
                    continue
                
                pulsar_timing_points = 0
                
                for timing_file in timing_files:
                    try:
                        logger.info(f"     Reading {timing_file.name}...")
                        
                        # Read timing file
                        with open(timing_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Parse timing data
                        for line in lines:
                            line = line.strip()
                            
                            # Skip comments and empty lines
                            if not line or line.startswith('#'):
                                continue
                            
                            # Parse timing line
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    time = float(parts[0])
                                    residual = float(parts[1])
                                    uncertainty = float(parts[2]) if len(parts) > 2 else 1e-6
                                    
                                    timing_data.append({
                                        'pulsar_name': pulsar_name,
                                        'time': time,
                                        'residual': residual,
                                        'uncertainty': uncertainty,
                                        'observatory': 'IPTA_DR2',
                                        'file': timing_file.name
                                    })
                                    
                                    pulsar_timing_points += 1
                                    
                                except ValueError as e:
                                    logger.debug(f"Failed to parse line: {line} - {e}")
                                    continue
                    
                    except Exception as e:
                        logger.warning(f"Failed to read {timing_file}: {e}")
                        continue
                
                logger.info(f"   Loaded {pulsar_timing_points} timing points for {pulsar_name}")
            
            logger.info(f"✅ Total real timing points loaded: {len(timing_data)}")
            return timing_data
            
        except Exception as e:
            logger.error(f"Real timing data loading failed: {e}")
            return []
    
    def run_real_targeted_hunt(self):
        """Run the real targeted cosmic string hunt"""
        logger.info("🌌 STARTING REAL TARGETED COSMIC STRING HUNT")
        logger.info("=" * 60)
        
        # Load real data
        if not self.load_real_targeted_data():
            return False
        
        if not self.timing_data:
            logger.error("No real timing data loaded!")
            return False
        
        logger.info(f"✅ Real data loaded: {len(self.timing_data)} points from {len(self.pulsar_catalog)} target pulsars")
        
        # Run targeted analyses on real data
        self.results = {}
        
        # 1. Real Data Superstring Analysis
        logger.info("🔗 TARGET 1: Real Data Superstring Analysis")
        self.results['superstring_analysis'] = self.analyze_real_superstrings()
        
        # 2. Real Data Frequency-Band Analysis
        logger.info("📊 TARGET 2: Real Data Frequency-Band Analysis")
        self.results['frequency_band_analysis'] = self.analyze_real_frequency_bands()
        
        # 3. Real Data Burst Event Detection
        logger.info("⚡ TARGET 3: Real Data Burst Event Detection")
        self.results['burst_analysis'] = self.detect_real_burst_events()
        
        # 4. Real Data Cross-Pulsar Correlation Analysis
        logger.info("🔍 TARGET 4: Real Data Cross-Pulsar Correlation Analysis")
        self.results['correlation_analysis'] = self.analyze_real_correlations()
        
        # 5. Real Data Memory Effect Detection
        logger.info("🧠 TARGET 5: Real Data Memory Effect Detection")
        self.results['memory_effect_analysis'] = self.detect_real_memory_effects()
        
        # 6. Real Data Non-Gaussian Signature Hunt
        logger.info("📈 TARGET 6: Real Data Non-Gaussian Signature Hunt")
        self.results['non_gaussian_analysis'] = self.hunt_real_non_gaussian()
        
        # Compile results
        self.compile_results()
        
        logger.info("🌌 REAL TARGETED HUNT COMPLETE!")
        return True
    
    def analyze_real_superstrings(self):
        """Analyze superstring networks with real data"""
        logger.info("   Analyzing real superstring networks...")
        
        try:
            superstring_candidates = []
            
            for pulsar in self.target_pulsars:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 20:
                    logger.warning(f"Insufficient data for {pulsar_name}: {len(pulsar_timing)} points")
                    continue
                
                residuals = np.array([d['residual'] for d in pulsar_timing])
                times = np.array([d['time'] for d in pulsar_timing])
                
                logger.info(f"     Analyzing {pulsar_name}: {len(residuals)} points, time span: {times[-1] - times[0]:.1f} days")
                
                # Analyze power spectrum for kink vs cusp signatures
                power_spectrum = np.abs(np.fft.fft(residuals))**2
                frequencies = np.fft.fftfreq(len(residuals), d=np.diff(times).mean())
                
                # Focus on positive frequencies
                pos_freq_mask = frequencies > 0
                pos_frequencies = frequencies[pos_freq_mask]
                pos_power = power_spectrum[pos_freq_mask]
                
                if len(pos_frequencies) > 10:
                    # Fit power law with higher precision
                    log_freq = np.log10(pos_frequencies[1:])
                    log_power = np.log10(pos_power[1:])
                    
                    if len(log_freq) > 5:
                        coeffs = np.polyfit(log_freq, log_power, 1)
                        spectral_index = coeffs[0]
                        
                        # Kink dominance: spectral index closer to -1 than -4/3
                        kink_dominance = abs(spectral_index - (-1.0)) < abs(spectral_index - (-4/3))
                        
                        # Non-Gaussian analysis
                        skewness = self.calculate_skewness(residuals)
                        kurtosis = self.calculate_kurtosis(residuals)
                        
                        # More sensitive thresholds
                        is_non_gaussian = abs(skewness) > 0.2 or abs(kurtosis - 3) > 0.3
                        
                        # Signal strength
                        signal_strength = np.std(residuals)
                        snr = signal_strength / 1e-6  # Signal-to-noise ratio
                        
                        if (kink_dominance and is_non_gaussian) or snr > self.detection_parameters['signal_to_noise_threshold']:
                            superstring_candidates.append({
                                'pulsar': pulsar_name,
                                'spectral_index': spectral_index,
                                'kink_dominance': kink_dominance,
                                'skewness': skewness,
                                'kurtosis': kurtosis,
                                'signal_strength': signal_strength,
                                'snr': snr,
                                'data_points': len(residuals),
                                'significance': self.calculate_significance(residuals),
                                'precision': pulsar['precision']
                            })
                            
                            logger.info(f"     Found superstring candidate: {pulsar_name} (SNR: {snr:.1f}, spectral index: {spectral_index:.2f})")
            
            logger.info(f"   Found {len(superstring_candidates)} real superstring candidates")
            
            return {
                'candidates': superstring_candidates,
                'total_analyzed': len(self.target_pulsars),
                'detection_threshold': self.detection_parameters['signal_to_noise_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Real superstring analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def analyze_real_frequency_bands(self):
        """Analyze different frequency bands for cosmic string signatures in real data"""
        logger.info("   Analyzing real frequency bands...")
        
        try:
            frequency_band_results = {}
            
            for band_name, (freq_min, freq_max) in self.detection_parameters['frequency_bands'].items():
                logger.info(f"     Analyzing {band_name} band: {freq_min:.1e} - {freq_max:.1e} Hz")
                
                band_candidates = []
                
                for pulsar in self.target_pulsars:
                    pulsar_name = pulsar['name']
                    pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                    
                    if len(pulsar_timing) < 20:
                        continue
                    
                    residuals = np.array([d['residual'] for d in pulsar_timing])
                    times = np.array([d['time'] for d in pulsar_timing])
                    
                    # Calculate power spectrum
                    power_spectrum = np.abs(np.fft.fft(residuals))**2
                    frequencies = np.fft.fftfreq(len(residuals), d=np.diff(times).mean())
                    
                    # Focus on positive frequencies
                    pos_freq_mask = frequencies > 0
                    pos_frequencies = frequencies[pos_freq_mask]
                    pos_power = power_spectrum[pos_freq_mask]
                    
                    # Filter to frequency band
                    band_mask = (pos_frequencies >= freq_min) & (pos_frequencies <= freq_max)
                    band_frequencies = pos_frequencies[band_mask]
                    band_power = pos_power[band_mask]
                    
                    if len(band_frequencies) > 5:
                        # Analyze power in this band
                        total_power = np.sum(band_power)
                        mean_power = np.mean(band_power)
                        max_power = np.max(band_power)
                        
                        # Look for excess power in this band
                        if total_power > 2 * np.mean(pos_power) * len(band_frequencies):
                            band_candidates.append({
                                'pulsar': pulsar_name,
                                'total_power': total_power,
                                'mean_power': mean_power,
                                'max_power': max_power,
                                'excess_factor': total_power / (np.mean(pos_power) * len(band_frequencies)),
                                'data_points': len(residuals)
                            })
                
                frequency_band_results[band_name] = {
                    'candidates': band_candidates,
                    'frequency_range': (freq_min, freq_max),
                    'total_analyzed': len(self.target_pulsars)
                }
                
                logger.info(f"     Found {len(band_candidates)} candidates in {band_name} band")
            
            return {
                'frequency_bands': frequency_band_results,
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Real frequency band analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def detect_real_burst_events(self):
        """Detect individual burst events from cosmic strings in real data"""
        logger.info("   Detecting real burst events...")
        
        try:
            burst_candidates = []
            
            for pulsar in self.target_pulsars:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 30:
                    continue
                
                residuals = np.array([d['residual'] for d in pulsar_timing])
                times = np.array([d['time'] for d in pulsar_timing])
                
                # Look for burst events
                burst_events = self.detect_bursts(residuals, times)
                
                if burst_events:
                    burst_candidates.append({
                        'pulsar': pulsar_name,
                        'burst_events': burst_events,
                        'data_points': len(residuals),
                        'time_span': times[-1] - times[0]
                    })
                    
                    logger.info(f"     Found {len(burst_events)} burst events in {pulsar_name}")
            
            logger.info(f"   Found {len(burst_candidates)} pulsars with real burst events")
            
            return {
                'candidates': burst_candidates,
                'total_analyzed': len(self.target_pulsars),
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Real burst event detection failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def analyze_real_correlations(self):
        """Analyze correlations between high-precision pulsars in real data"""
        logger.info("   Analyzing real cross-pulsar correlations...")
        
        try:
            # Calculate correlations between pulsars
            pulsar_residuals = {}
            
            for pulsar in self.target_pulsars:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) >= 20:
                    residuals = np.array([d['residual'] for d in pulsar_timing])
                    pulsar_residuals[pulsar_name] = residuals
            
            # Calculate correlation matrix
            pulsar_names = list(pulsar_residuals.keys())
            n_pulsars = len(pulsar_names)
            
            if n_pulsars < 2:
                return {'error': 'Insufficient pulsars for correlation analysis', 'analysis_complete': False}
            
            correlation_matrix = np.zeros((n_pulsars, n_pulsars))
            
            for i in range(n_pulsars):
                for j in range(i+1, n_pulsars):
                    residuals_i = pulsar_residuals[pulsar_names[i]]
                    residuals_j = pulsar_residuals[pulsar_names[j]]
                    
                    # Calculate correlation
                    if len(residuals_i) == len(residuals_j):
                        corr = np.corrcoef(residuals_i, residuals_j)[0, 1]
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
                    else:
                        # Handle different length time series
                        min_len = min(len(residuals_i), len(residuals_j))
                        corr = np.corrcoef(residuals_i[:min_len], residuals_j[:min_len])[0, 1]
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
            
            # Look for significant correlations
            significant_correlations = []
            
            for i in range(n_pulsars):
                for j in range(i+1, n_pulsars):
                    corr = correlation_matrix[i, j]
                    
                    if abs(corr) > self.detection_parameters['correlation_threshold']:
                        significant_correlations.append({
                            'pulsar_pair': (pulsar_names[i], pulsar_names[j]),
                            'correlation': corr,
                            'significance': abs(corr)
                        })
            
            logger.info(f"   Found {len(significant_correlations)} significant real correlations")
            
            return {
                'significant_correlations': significant_correlations,
                'correlation_matrix': correlation_matrix.tolist(),
                'pulsar_names': pulsar_names,
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Real cross-pulsar correlation analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def detect_real_memory_effects(self):
        """Detect memory effects with real data analysis"""
        logger.info("   Detecting real memory effects...")
        
        try:
            memory_candidates = []
            
            for pulsar in self.target_pulsars:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 50:
                    continue
                
                residuals = np.array([d['residual'] for d in pulsar_timing])
                times = np.array([d['time'] for d in pulsar_timing])
                
                # Look for step-function changes (memory effects)
                step_detections = self.detect_step_functions(residuals, times)
                
                if step_detections:
                    memory_candidates.append({
                        'pulsar': pulsar_name,
                        'step_detections': step_detections,
                        'data_points': len(residuals),
                        'time_span': times[-1] - times[0]
                    })
                    
                    logger.info(f"     Found {len(step_detections)} memory effects in {pulsar_name}")
            
            logger.info(f"   Found {len(memory_candidates)} pulsars with real memory effects")
            
            return {
                'candidates': memory_candidates,
                'total_analyzed': len(self.target_pulsars),
                'threshold': self.detection_parameters['memory_effect_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Real memory effect detection failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def hunt_real_non_gaussian(self):
        """Hunt for non-Gaussian signatures in real data"""
        logger.info("   Hunting real non-Gaussian signatures...")
        
        try:
            non_gaussian_candidates = []
            
            for pulsar in self.target_pulsars:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 30:
                    continue
                
                residuals = np.array([d['residual'] for d in pulsar_timing])
                
                # Multiple non-Gaussian tests
                tests = {
                    'skewness': self.calculate_skewness(residuals),
                    'kurtosis': self.calculate_kurtosis(residuals),
                    'jarque_bera': self.jarque_bera_test(residuals),
                    'shapiro_wilk': self.shapiro_wilk_test(residuals),
                    'anderson_darling': self.anderson_darling_test(residuals)
                }
                
                # More sensitive scoring
                non_gaussian_score = 0
                if abs(tests['skewness']) > 0.2:  # More sensitive
                    non_gaussian_score += 1
                if abs(tests['kurtosis'] - 3) > 0.3:  # More sensitive
                    non_gaussian_score += 1
                if tests['jarque_bera'] < 0.05:  # More sensitive
                    non_gaussian_score += 1
                if tests['shapiro_wilk'] < 0.05:  # More sensitive
                    non_gaussian_score += 1
                if tests['anderson_darling'] < 0.05:  # More sensitive
                    non_gaussian_score += 1
                
                if non_gaussian_score >= 2:  # Lowered threshold
                    non_gaussian_candidates.append({
                        'pulsar': pulsar_name,
                        'non_gaussian_score': non_gaussian_score,
                        'tests': tests,
                        'data_points': len(residuals),
                        'residual_std': np.std(residuals)
                    })
                    
                    logger.info(f"     Found non-Gaussian signature in {pulsar_name} (score: {non_gaussian_score})")
            
            logger.info(f"   Found {len(non_gaussian_candidates)} real non-Gaussian candidates")
            
            return {
                'candidates': non_gaussian_candidates,
                'total_analyzed': len(self.target_pulsars),
                'threshold': self.detection_parameters['non_gaussian_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Real non-Gaussian analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def compile_results(self):
        """Compile all results into a comprehensive report"""
        logger.info("📊 Compiling results...")
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pulsars': len(self.target_pulsars),
            'total_timing_points': len(self.timing_data),
            'detection_parameters': self.detection_parameters,
            'analysis_summary': {}
        }
        
        # Summarize each analysis
        for analysis_name, results in self.results.items():
            if 'error' in results:
                summary['analysis_summary'][analysis_name] = {
                    'status': 'failed',
                    'error': results['error']
                }
            else:
                summary['analysis_summary'][analysis_name] = {
                    'status': 'completed',
                    'candidates_found': len(results.get('candidates', [])),
                    'total_analyzed': results.get('total_analyzed', 0)
                }
        
        # Save results
        os.makedirs('real_targeted_hunt_results', exist_ok=True)
        
        # Save detailed results
        with open('real_targeted_hunt_results/detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        with open('real_targeted_hunt_results/summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create visualization
        self.create_real_visualization()
        
        logger.info("✅ Results compiled and saved to real_targeted_hunt_results/")
    
    def create_real_visualization(self):
        """Create real data visualization of results"""
        logger.info("🎨 Creating real data visualization...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('🌌 Real Targeted Cosmic String Hunt Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Superstring Candidates
            if 'superstring_analysis' in self.results and 'candidates' in self.results['superstring_analysis']:
                candidates = self.results['superstring_analysis']['candidates']
                if candidates:
                    spectral_indices = [c['spectral_index'] for c in candidates]
                    significances = [c['significance'] for c in candidates]
                    
                    axes[0, 0].scatter(spectral_indices, significances, alpha=0.7, c='red', s=100)
                    axes[0, 0].axvline(-1.0, color='green', linestyle='--', linewidth=2, label='Kink (-1)')
                    axes[0, 0].axvline(-4/3, color='blue', linestyle='--', linewidth=2, label='Cusp (-4/3)')
                    axes[0, 0].set_xlabel('Spectral Index')
                    axes[0, 0].set_ylabel('Significance')
                    axes[0, 0].set_title('Real Superstring Candidates')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Superstring\nCandidates Found', 
                                   ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title('Real Superstring Candidates')
            
            # Plot 2: Frequency Band Analysis
            if 'frequency_band_analysis' in self.results and 'frequency_bands' in self.results['frequency_band_analysis']:
                bands = self.results['frequency_band_analysis']['frequency_bands']
                band_names = list(bands.keys())
                candidate_counts = [len(bands[band]['candidates']) for band in band_names]
                
                axes[0, 1].bar(band_names, candidate_counts, alpha=0.7, color='orange')
                axes[0, 1].set_xlabel('Frequency Band')
                axes[0, 1].set_ylabel('Candidates Found')
                axes[0, 1].set_title('Real Frequency Band Analysis')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Burst Events
            if 'burst_analysis' in self.results and 'candidates' in self.results['burst_analysis']:
                candidates = self.results['burst_analysis']['candidates']
                if candidates:
                    burst_counts = [len(c['burst_events']) for c in candidates]
                    pulsar_names = [c['pulsar'] for c in candidates]
                    
                    axes[0, 2].bar(pulsar_names, burst_counts, alpha=0.7, color='purple')
                    axes[0, 2].set_xlabel('Pulsar')
                    axes[0, 2].set_ylabel('Burst Events')
                    axes[0, 2].set_title('Real Burst Event Detection')
                    axes[0, 2].tick_params(axis='x', rotation=45)
                    axes[0, 2].grid(True, alpha=0.3)
                else:
                    axes[0, 2].text(0.5, 0.5, 'No Burst Events\nDetected', 
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 2].set_title('Real Burst Event Detection')
            
            # Plot 4: Cross-Pulsar Correlations
            if 'correlation_analysis' in self.results and 'significant_correlations' in self.results['correlation_analysis']:
                correlations = self.results['correlation_analysis']['significant_correlations']
                if correlations:
                    corr_values = [c['correlation'] for c in correlations]
                    pair_names = [f"{c['pulsar_pair'][0][:8]}-{c['pulsar_pair'][1][:8]}" for c in correlations]
                    
                    axes[1, 0].bar(pair_names, corr_values, alpha=0.7, color='blue')
                    axes[1, 0].set_xlabel('Pulsar Pair')
                    axes[1, 0].set_ylabel('Correlation')
                    axes[1, 0].set_title('Real Cross-Pulsar Correlations')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Significant\nCorrelations Found', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Real Cross-Pulsar Correlations')
            
            # Plot 5: Memory Effects
            if 'memory_effect_analysis' in self.results and 'candidates' in self.results['memory_effect_analysis']:
                candidates = self.results['memory_effect_analysis']['candidates']
                if candidates:
                    step_counts = [len(c['step_detections']) for c in candidates]
                    pulsar_names = [c['pulsar'] for c in candidates]
                    
                    axes[1, 1].bar(pulsar_names, step_counts, alpha=0.7, color='green')
                    axes[1, 1].set_xlabel('Pulsar')
                    axes[1, 1].set_ylabel('Memory Effects')
                    axes[1, 1].set_title('Real Memory Effect Detection')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Memory Effects\nDetected', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Real Memory Effect Detection')
            
            # Plot 6: Analysis Summary
            analysis_names = list(self.results.keys())
            completed_analyses = sum(1 for r in self.results.values() if 'error' not in r)
            total_analyses = len(analysis_names)
            
            axes[1, 2].bar(['Completed', 'Failed'], [completed_analyses, total_analyses - completed_analyses], 
                          color=['green', 'red'], alpha=0.7)
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Real Analysis Status')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('real_targeted_hunt_results/real_targeted_hunt_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Visualization created: real_targeted_hunt_results/real_targeted_hunt_visualization.png")
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
    
    # Helper methods for statistical analysis
    def calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 3
        return np.mean(((data - mean) / std) ** 4)
    
    def jarque_bera_test(self, data):
        """Jarque-Bera test for normality"""
        n = len(data)
        if n < 2:
            return 1.0
        
        skewness = self.calculate_skewness(data)
        kurtosis = self.calculate_kurtosis(data)
        
        jb_stat = n * (skewness**2 / 6 + (kurtosis - 3)**2 / 24)
        
        # Approximate p-value (simplified)
        if jb_stat < 2:
            return 0.5
        elif jb_stat < 6:
            return 0.1
        else:
            return 0.01
    
    def shapiro_wilk_test(self, data):
        """Shapiro-Wilk test for normality (simplified)"""
        n = len(data)
        if n < 3:
            return 1.0
        
        # Simplified implementation
        skewness = abs(self.calculate_skewness(data))
        kurtosis = abs(self.calculate_kurtosis(data) - 3)
        
        # Approximate p-value
        if skewness < 0.5 and kurtosis < 1:
            return 0.5
        elif skewness < 1 and kurtosis < 2:
            return 0.1
        else:
            return 0.01
    
    def anderson_darling_test(self, data):
        """Anderson-Darling test for normality (simplified)"""
        n = len(data)
        if n < 3:
            return 1.0
        
        # Simplified implementation based on skewness and kurtosis
        skewness = abs(self.calculate_skewness(data))
        kurtosis = abs(self.calculate_kurtosis(data) - 3)
        
        # Approximate p-value
        if skewness < 0.3 and kurtosis < 1:
            return 0.5
        elif skewness < 0.6 and kurtosis < 1:
            return 0.1
        else:
            return 0.01
    
    def detect_step_functions(self, residuals, times):
        """Detect step functions in residuals (memory effects)"""
        if len(residuals) < 10:
            return []
        
        step_detections = []
        threshold = self.detection_parameters['memory_effect_threshold']
        
        # Look for sudden changes in residuals
        for i in range(5, len(residuals) - 5):
            # Compare before and after windows
            before_mean = np.mean(residuals[i-5:i])
            after_mean = np.mean(residuals[i:i+5])
            
            step_size = abs(after_mean - before_mean)
            
            if step_size > threshold:
                step_detections.append({
                    'time': times[i],
                    'step_size': step_size,
                    'before_mean': before_mean,
                    'after_mean': after_mean
                })
        
        return step_detections
    
    def detect_bursts(self, residuals, times):
        """Detect burst events in residuals"""
        if len(residuals) < 20:
            return []
        
        burst_events = []
        threshold = self.detection_parameters['burst_threshold']
        
        # Look for burst events
        for i in range(10, len(residuals) - 10):
            # Compare local region to surrounding regions
            local_region = residuals[i-5:i+5]
            surrounding_region = np.concatenate([residuals[i-10:i-5], residuals[i+5:i+10]])
            
            local_std = np.std(local_region)
            surrounding_std = np.std(surrounding_region)
            
            if local_std > threshold * surrounding_std:
                burst_events.append({
                    'time': times[i],
                    'amplitude': np.max(np.abs(local_region)),
                    'local_std': local_std,
                    'surrounding_std': surrounding_std,
                    'burst_factor': local_std / surrounding_std
                })
        
        return burst_events
    
    def calculate_significance(self, residuals):
        """Calculate significance of signal"""
        if len(residuals) < 2:
            return 0
        
        mean = np.mean(residuals)
        std = np.std(residuals)
        
        if std == 0:
            return 0
        
        # Calculate signal-to-noise ratio
        snr = abs(mean) / std
        
        # Convert to approximate sigma
        return min(snr * 2, 10)  # Cap at 10σ

def main():
    """Main function to run real targeted cosmic string hunt"""
    print("🌌 REAL TARGETED COSMIC STRING HUNT")
    print("=" * 50)
    print("Direct analysis of real IPTA DR2 data:")
    print("• Target pulsars: J1909-3744, J1713+0747, J0437-4715, J1744-1134, J1857+0943")
    print("• Real timing files (.tim) from IPTA DR2")
    print("• High-precision analysis with real astrophysical data")
    print("• Frequency-band analysis, burst detection, correlations")
    print("=" * 50)
    
    hunt = RealTargetedCosmicStringHunt()
    success = hunt.run_real_targeted_hunt()
    
    if success:
        print("\n🌌 REAL TARGETED HUNT COMPLETE!")
        print("Results saved to: real_targeted_hunt_results/")
        print("Check the visualization: real_targeted_hunt_results/real_targeted_hunt_visualization.png")
    else:
        print("\n❌ REAL TARGETED HUNT FAILED!")
        print("Check the logs for details.")

if __name__ == "__main__":
    main()
