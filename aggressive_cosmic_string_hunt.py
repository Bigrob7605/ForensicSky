#!/usr/bin/env python3
"""
🚀 AGGRESSIVE COSMIC STRING HUNT
High-sensitivity test designed to detect cosmic string signatures

This test creates synthetic data with embedded cosmic string signatures
and uses aggressive detection thresholds to find them.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
from datetime import datetime
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AggressiveCosmicStringHunt:
    """
    🚀 AGGRESSIVE COSMIC STRING HUNT
    
    High-sensitivity test designed to detect cosmic string signatures
    """
    
    def __init__(self):
        self.timing_data = []
        self.pulsar_catalog = []
        self.results = {}
        
        # Aggressive detection parameters
        self.detection_parameters = {
            'superstring_tension_range': (1e-12, 1e-11),
            'kink_dominance_threshold': 0.5,  # More aggressive
            'non_gaussian_threshold': 2.0,   # More sensitive
            'memory_effect_threshold': 5e-10,  # More sensitive
            'correlation_threshold': 0.05,   # More sensitive
            'signal_to_noise_threshold': 1.5,  # More sensitive
        }
    
    def create_synthetic_data_with_signatures(self):
        """Create synthetic data with embedded cosmic string signatures"""
        logger.info("🔬 Creating synthetic data with cosmic string signatures...")
        
        # High-priority pulsars from NANOGrav 15-year
        self.pulsar_catalog = [
            {'name': 'J1909-3744', 'ra': 287.4, 'dec': -37.7, 'period': 0.0029},
            {'name': 'J1713+0747', 'ra': 258.3, 'dec': 7.4, 'period': 0.0046},
            {'name': 'J0437-4715', 'ra': 69.3, 'dec': -47.2, 'period': 0.0058},
            {'name': 'J1744-1134', 'ra': 266.0, 'dec': -11.6, 'period': 0.0041},
            {'name': 'J1857+0943', 'ra': 284.4, 'dec': 9.7, 'period': 0.0054},
            {'name': 'J1939+2134', 'ra': 294.8, 'dec': 21.6, 'period': 0.0016},
            {'name': 'J2145-0750', 'ra': 326.3, 'dec': -7.8, 'period': 0.0161},
            {'name': 'J2317+1439', 'ra': 349.3, 'dec': 14.7, 'period': 0.0034},
        ]
        
        self.timing_data = []
        
        for i, pulsar in enumerate(self.pulsar_catalog):
            pulsar_name = pulsar['name']
            
            # Create realistic timing data
            n_points = np.random.randint(100, 300)
            times = np.linspace(50000, 60000, n_points)  # MJD range
            
            # Base residuals with realistic noise
            residuals = np.random.normal(0, 1e-6, n_points)
            
            # EMBED COSMIC STRING SIGNATURES
            if i < 3:  # First 3 pulsars get cosmic string signatures
                logger.info(f"   Embedding cosmic string signature in {pulsar_name}")
                
                # 1. Kink-dominated spectrum (f^-1)
                frequencies = np.fft.fftfreq(n_points, d=np.diff(times).mean())
                power_spectrum = np.abs(frequencies[1:]) ** (-1.0)  # Kink spectrum
                power_spectrum = np.concatenate([[power_spectrum[0]], power_spectrum])
                
                # Apply to residuals
                residual_fft = np.fft.fft(residuals)
                residual_fft[1:] *= np.sqrt(power_spectrum[1:]) * 2.0  # Amplify signal
                residuals = np.real(np.fft.ifft(residual_fft))
                
                # 2. Memory effect (step function)
                if np.random.random() < 0.7:  # 70% chance
                    step_time = np.random.choice(times)
                    step_size = np.random.normal(0, 2e-9)  # Stronger signal
                    residuals[times >= step_time] += step_size
                
                # 3. Non-Gaussian features
                # Add skewness
                residuals += np.random.normal(0, 0.5e-6, n_points) * np.sin(2 * np.pi * times / 1000)
                
                # Add kurtosis (heavy tails)
                heavy_tails = np.random.choice([-1, 1], n_points) * np.random.exponential(1e-6, n_points)
                residuals += heavy_tails * 0.3
                
            elif i < 6:  # Next 3 pulsars get partial signatures
                logger.info(f"   Embedding partial cosmic string signature in {pulsar_name}")
                
                # Partial kink spectrum
                frequencies = np.fft.fftfreq(n_points, d=np.diff(times).mean())
                power_spectrum = np.abs(frequencies[1:]) ** (-0.8)  # Between kink and cusp
                power_spectrum = np.concatenate([[power_spectrum[0]], power_spectrum])
                
                residual_fft = np.fft.fft(residuals)
                residual_fft[1:] *= np.sqrt(power_spectrum[1:]) * 1.2
                residuals = np.real(np.fft.ifft(residual_fft))
                
                # Mild non-Gaussian features
                residuals += np.random.normal(0, 0.3e-6, n_points) * np.sin(2 * np.pi * times / 2000)
            
            # Add to timing data
            for j in range(n_points):
                self.timing_data.append({
                    'pulsar_name': pulsar_name,
                    'time': times[j],
                    'residual': residuals[j],
                    'uncertainty': 1e-6,
                    'observatory': 'Synthetic'
                })
        
        logger.info(f"✅ Created synthetic data: {len(self.timing_data)} points from {len(self.pulsar_catalog)} pulsars")
    
    def run_aggressive_hunt(self):
        """Run the aggressive cosmic string hunt"""
        logger.info("🚀 STARTING AGGRESSIVE COSMIC STRING HUNT")
        logger.info("=" * 60)
        
        # Create synthetic data with signatures
        self.create_synthetic_data_with_signatures()
        
        # Run aggressive analyses
        self.results = {}
        
        # 1. Superstring Network Analysis
        logger.info("🔗 TARGET 1: Superstring Network Analysis")
        self.results['superstring_analysis'] = self.analyze_superstring_networks()
        
        # 2. Non-Gaussian Signature Hunt
        logger.info("📊 TARGET 2: Non-Gaussian Signature Hunt")
        self.results['non_gaussian_analysis'] = self.hunt_non_gaussian_signatures()
        
        # 3. Memory Effect Detection
        logger.info("🧠 TARGET 3: Memory Effect Detection")
        self.results['memory_effect_analysis'] = self.detect_memory_effects()
        
        # 4. Kink Dominance Analysis
        logger.info("⚡ TARGET 4: Kink Dominance Analysis")
        self.results['kink_dominance_analysis'] = self.analyze_kink_dominance()
        
        # 5. Cross-Correlation Pattern Mining
        logger.info("🔍 TARGET 5: Cross-Correlation Pattern Mining")
        self.results['correlation_pattern_analysis'] = self.mine_correlation_patterns()
        
        # 6. Sky Localization Analysis
        logger.info("🌌 TARGET 6: Sky Localization Analysis")
        self.results['sky_localization_analysis'] = self.analyze_sky_localization()
        
        # Compile results
        self.compile_results()
        
        logger.info("🚀 AGGRESSIVE HUNT COMPLETE!")
        return True
    
    def analyze_superstring_networks(self):
        """Analyze superstring networks with aggressive detection"""
        logger.info("   Analyzing superstring networks...")
        
        try:
            superstring_candidates = []
            
            for pulsar in self.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 10:
                    continue
                
                residuals = np.array([d['residual'] for d in pulsar_timing])
                times = np.array([d['time'] for d in pulsar_timing])
                
                # Analyze power spectrum for kink vs cusp signatures
                power_spectrum = np.abs(np.fft.fft(residuals))**2
                frequencies = np.fft.fftfreq(len(residuals), d=np.diff(times).mean())
                
                # Focus on positive frequencies
                pos_freq_mask = frequencies > 0
                pos_frequencies = frequencies[pos_freq_mask]
                pos_power = power_spectrum[pos_freq_mask]
                
                if len(pos_frequencies) > 5:
                    # Fit power law
                    log_freq = np.log10(pos_frequencies[1:])
                    log_power = np.log10(pos_power[1:])
                    
                    if len(log_freq) > 3:
                        coeffs = np.polyfit(log_freq, log_power, 1)
                        spectral_index = coeffs[0]
                        
                        # Kink dominance: spectral index closer to -1 than -4/3
                        kink_dominance = abs(spectral_index - (-1.0)) < abs(spectral_index - (-4/3))
                        
                        # Non-Gaussian analysis
                        skewness = self.calculate_skewness(residuals)
                        kurtosis = self.calculate_kurtosis(residuals)
                        
                        # More aggressive thresholds
                        is_non_gaussian = abs(skewness) > 0.3 or abs(kurtosis - 3) > 0.5
                        
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
                                'significance': self.calculate_significance(residuals)
                            })
            
            logger.info(f"   Found {len(superstring_candidates)} superstring candidates")
            
            return {
                'candidates': superstring_candidates,
                'total_analyzed': len(self.pulsar_catalog),
                'detection_threshold': self.detection_parameters['signal_to_noise_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Superstring analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def hunt_non_gaussian_signatures(self):
        """Hunt for non-Gaussian signatures with aggressive detection"""
        logger.info("   Hunting non-Gaussian signatures...")
        
        try:
            non_gaussian_candidates = []
            
            for pulsar in self.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 20:
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
                
                # More aggressive scoring
                non_gaussian_score = 0
                if abs(tests['skewness']) > 0.3:  # Lowered threshold
                    non_gaussian_score += 1
                if abs(tests['kurtosis'] - 3) > 0.5:  # Lowered threshold
                    non_gaussian_score += 1
                if tests['jarque_bera'] < 0.1:  # More sensitive
                    non_gaussian_score += 1
                if tests['shapiro_wilk'] < 0.1:  # More sensitive
                    non_gaussian_score += 1
                if tests['anderson_darling'] < 0.1:  # More sensitive
                    non_gaussian_score += 1
                
                if non_gaussian_score >= 2:  # Lowered threshold
                    non_gaussian_candidates.append({
                        'pulsar': pulsar_name,
                        'non_gaussian_score': non_gaussian_score,
                        'tests': tests,
                        'data_points': len(residuals),
                        'residual_std': np.std(residuals)
                    })
            
            logger.info(f"   Found {len(non_gaussian_candidates)} non-Gaussian candidates")
            
            return {
                'candidates': non_gaussian_candidates,
                'total_analyzed': len(self.pulsar_catalog),
                'threshold': self.detection_parameters['non_gaussian_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Non-Gaussian analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def detect_memory_effects(self):
        """Detect memory effects with aggressive detection"""
        logger.info("   Detecting memory effects...")
        
        try:
            memory_candidates = []
            
            for pulsar in self.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 30:
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
            
            logger.info(f"   Found {len(memory_candidates)} memory effect candidates")
            
            return {
                'candidates': memory_candidates,
                'total_analyzed': len(self.pulsar_catalog),
                'threshold': self.detection_parameters['memory_effect_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Memory effect detection failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def analyze_kink_dominance(self):
        """Analyze kink dominance with aggressive detection"""
        logger.info("   Analyzing kink dominance...")
        
        try:
            kink_analysis = []
            
            for pulsar in self.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 15:
                    continue
                
                residuals = np.array([d['residual'] for d in pulsar_timing])
                times = np.array([d['time'] for d in pulsar_timing])
                
                # Analyze power spectrum for kink vs cusp signatures
                power_spectrum = np.abs(np.fft.fft(residuals))**2
                frequencies = np.fft.fftfreq(len(residuals), d=np.diff(times).mean())
                
                # Focus on positive frequencies
                pos_freq_mask = frequencies > 0
                pos_frequencies = frequencies[pos_freq_mask]
                pos_power = power_spectrum[pos_freq_mask]
                
                if len(pos_frequencies) > 5:
                    # Fit power laws for kink (-1) vs cusp (-4/3) signatures
                    log_freq = np.log10(pos_frequencies[1:])
                    log_power = np.log10(pos_power[1:])
                    
                    if len(log_freq) > 3:
                        coeffs = np.polyfit(log_freq, log_power, 1)
                        spectral_index = coeffs[0]
                        
                        # Calculate dominance
                        kink_distance = abs(spectral_index - (-1.0))
                        cusp_distance = abs(spectral_index - (-4/3))
                        
                        kink_dominance = kink_distance < cusp_distance
                        dominance_ratio = cusp_distance / kink_distance if kink_distance > 0 else float('inf')
                        
                        kink_analysis.append({
                            'pulsar': pulsar_name,
                            'spectral_index': spectral_index,
                            'kink_dominance': kink_dominance,
                            'dominance_ratio': dominance_ratio,
                            'kink_distance': kink_distance,
                            'cusp_distance': cusp_distance,
                            'data_points': len(residuals)
                        })
            
            # Calculate overall kink dominance
            total_pulsars = len(kink_analysis)
            kink_dominant_pulsars = sum(1 for a in kink_analysis if a['kink_dominance'])
            kink_dominance_fraction = kink_dominant_pulsars / total_pulsars if total_pulsars > 0 else 0
            
            logger.info(f"   Kink dominance: {kink_dominance_fraction:.1%} of pulsars")
            
            return {
                'pulsar_analysis': kink_analysis,
                'total_pulsars': total_pulsars,
                'kink_dominant_pulsars': kink_dominant_pulsars,
                'kink_dominance_fraction': kink_dominance_fraction,
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Kink dominance analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def mine_correlation_patterns(self):
        """Mine for non-standard correlation patterns with aggressive detection"""
        logger.info("   Mining correlation patterns...")
        
        try:
            # Calculate correlations between pulsars
            pulsar_residuals = {}
            
            for pulsar in self.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) >= 10:
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
            
            # Look for deviations from Hellings-Downs curve
            hellings_downs_deviations = []
            
            for i in range(n_pulsars):
                for j in range(i+1, n_pulsars):
                    # Calculate angular separation
                    pulsar_i = self.pulsar_catalog[i] if i < len(self.pulsar_catalog) else {'ra': 0, 'dec': 0}
                    pulsar_j = self.pulsar_catalog[j] if j < len(self.pulsar_catalog) else {'ra': 0, 'dec': 0}
                    
                    # Convert to radians
                    ra1, dec1 = np.radians(pulsar_i['ra']), np.radians(pulsar_i['dec'])
                    ra2, dec2 = np.radians(pulsar_j['ra']), np.radians(pulsar_j['dec'])
                    
                    # Calculate angular separation
                    cos_angle = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    
                    # Expected Hellings-Downs correlation
                    expected_corr = 0.5 * (1 + np.cos(angle)) * np.log(1 + np.cos(angle)) - 0.5
                    
                    # Observed correlation
                    observed_corr = correlation_matrix[i, j]
                    
                    # Deviation
                    deviation = abs(observed_corr - expected_corr)
                    
                    if deviation > self.detection_parameters['correlation_threshold']:  # More sensitive
                        hellings_downs_deviations.append({
                            'pulsar_pair': (pulsar_names[i], pulsar_names[j]),
                            'angular_separation': angle,
                            'expected_correlation': expected_corr,
                            'observed_correlation': observed_corr,
                            'deviation': deviation
                        })
            
            logger.info(f"   Found {len(hellings_downs_deviations)} Hellings-Downs deviations")
            
            return {
                'hellings_downs_deviations': hellings_downs_deviations,
                'total_pairs_analyzed': len(hellings_downs_deviations),
                'correlation_matrix': correlation_matrix.tolist(),
                'pulsar_names': pulsar_names,
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Correlation pattern mining failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def analyze_sky_localization(self):
        """Analyze sky localization with aggressive detection"""
        logger.info("   Analyzing sky localization...")
        
        try:
            sky_analysis = {
                'pulsar_positions': [],
                'signal_strength_by_region': {},
                'hotspots': []
            }
            
            for pulsar in self.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 10:
                    continue
                
                # Calculate signal strength
                residuals = np.array([d['residual'] for d in pulsar_timing])
                signal_strength = np.std(residuals)
                
                # Get sky position
                ra = pulsar.get('ra', 0)
                dec = pulsar.get('dec', 0)
                
                sky_analysis['pulsar_positions'].append({
                    'pulsar': pulsar_name,
                    'ra': ra,
                    'dec': dec,
                    'signal_strength': signal_strength,
                    'data_points': len(pulsar_timing)
                })
            
            # Identify hotspots (regions with high signal strength)
            if sky_analysis['pulsar_positions']:
                signal_strengths = [p['signal_strength'] for p in sky_analysis['pulsar_positions']]
                mean_signal = np.mean(signal_strengths)
                std_signal = np.std(signal_strengths)
                
                # More aggressive hotspot detection
                hotspot_threshold = mean_signal + 1.5 * std_signal  # Lowered threshold
                
                for pos in sky_analysis['pulsar_positions']:
                    if pos['signal_strength'] > hotspot_threshold:
                        sky_analysis['hotspots'].append(pos)
            
            logger.info(f"   Found {len(sky_analysis['hotspots'])} sky hotspots")
            
            return {
                'sky_analysis': sky_analysis,
                'total_pulsars': len(sky_analysis['pulsar_positions']),
                'hotspots': len(sky_analysis['hotspots']),
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Sky localization analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def compile_results(self):
        """Compile all results into a comprehensive report"""
        logger.info("📊 Compiling results...")
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pulsars': len(self.pulsar_catalog),
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
        os.makedirs('aggressive_hunt_results', exist_ok=True)
        
        # Save detailed results
        with open('aggressive_hunt_results/detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        with open('aggressive_hunt_results/summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create visualization
        self.create_aggressive_visualization()
        
        logger.info("✅ Results compiled and saved to aggressive_hunt_results/")
    
    def create_aggressive_visualization(self):
        """Create aggressive visualization of results"""
        logger.info("🎨 Creating aggressive visualization...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('🚀 Aggressive Cosmic String Hunt Results', fontsize=16, fontweight='bold')
            
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
                    axes[0, 0].set_title('Superstring Candidates')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Superstring\nCandidates Found', 
                                   ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title('Superstring Candidates')
            
            # Plot 2: Non-Gaussian Candidates
            if 'non_gaussian_analysis' in self.results and 'candidates' in self.results['non_gaussian_analysis']:
                candidates = self.results['non_gaussian_analysis']['candidates']
                if candidates:
                    scores = [c['non_gaussian_score'] for c in candidates]
                    axes[0, 1].hist(scores, bins=range(1, 7), alpha=0.7, color='orange')
                    axes[0, 1].set_xlabel('Non-Gaussian Score')
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].set_title('Non-Gaussian Candidates')
                    axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No Non-Gaussian\nCandidates Found', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('Non-Gaussian Candidates')
            
            # Plot 3: Memory Effect Candidates
            if 'memory_effect_analysis' in self.results and 'candidates' in self.results['memory_effect_analysis']:
                candidates = self.results['memory_effect_analysis']['candidates']
                if candidates:
                    step_counts = [len(c['step_detections']) for c in candidates]
                    axes[0, 2].hist(step_counts, bins=range(1, max(step_counts)+2), alpha=0.7, color='purple')
                    axes[0, 2].set_xlabel('Step Detections')
                    axes[0, 2].set_ylabel('Count')
                    axes[0, 2].set_title('Memory Effect Candidates')
                    axes[0, 2].grid(True, alpha=0.3)
                else:
                    axes[0, 2].text(0.5, 0.5, 'No Memory Effect\nCandidates Found', 
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 2].set_title('Memory Effect Candidates')
            
            # Plot 4: Kink Dominance
            if 'kink_dominance_analysis' in self.results and 'pulsar_analysis' in self.results['kink_dominance_analysis']:
                analysis = self.results['kink_dominance_analysis']['pulsar_analysis']
                if analysis:
                    spectral_indices = [a['spectral_index'] for a in analysis]
                    kink_dominant = [a['kink_dominance'] for a in analysis]
                    
                    colors = ['red' if kd else 'blue' for kd in kink_dominant]
                    axes[1, 0].scatter(range(len(spectral_indices)), spectral_indices, c=colors, alpha=0.7, s=100)
                    axes[1, 0].axhline(-1.0, color='red', linestyle='--', linewidth=2, label='Kink (-1)')
                    axes[1, 0].axhline(-4/3, color='blue', linestyle='--', linewidth=2, label='Cusp (-4/3)')
                    axes[1, 0].set_xlabel('Pulsar Index')
                    axes[1, 0].set_ylabel('Spectral Index')
                    axes[1, 0].set_title('Kink vs Cusp Dominance')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Kink Dominance\nAnalysis Available', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Kink vs Cusp Dominance')
            
            # Plot 5: Sky Hotspots
            if 'sky_localization_analysis' in self.results and 'sky_analysis' in self.results['sky_localization_analysis']:
                sky_analysis = self.results['sky_localization_analysis']['sky_analysis']
                if sky_analysis['pulsar_positions']:
                    positions = sky_analysis['pulsar_positions']
                    ras = [p['ra'] for p in positions]
                    decs = [p['dec'] for p in positions]
                    signal_strengths = [p['signal_strength'] for p in positions]
                    
                    scatter = axes[1, 1].scatter(ras, decs, c=signal_strengths, cmap='hot', alpha=0.7, s=100)
                    axes[1, 1].set_xlabel('Right Ascension (deg)')
                    axes[1, 1].set_ylabel('Declination (deg)')
                    axes[1, 1].set_title('Sky Signal Strength')
                    plt.colorbar(scatter, ax=axes[1, 1])
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Sky Analysis\nAvailable', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Sky Signal Strength')
            
            # Plot 6: Analysis Summary
            analysis_names = list(self.results.keys())
            completed_analyses = sum(1 for r in self.results.values() if 'error' not in r)
            total_analyses = len(analysis_names)
            
            axes[1, 2].bar(['Completed', 'Failed'], [completed_analyses, total_analyses - completed_analyses], 
                          color=['green', 'red'], alpha=0.7)
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Analysis Status')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('aggressive_hunt_results/aggressive_hunt_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Visualization created: aggressive_hunt_results/aggressive_hunt_visualization.png")
            
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
    """Main function to run aggressive cosmic string hunt"""
    print("🚀 AGGRESSIVE COSMIC STRING HUNT")
    print("=" * 50)
    print("High-sensitivity test with embedded cosmic string signatures:")
    print("• Superstring networks with kink dominance")
    print("• Non-Gaussian signature detection")
    print("• Memory effect detection")
    print("• Cross-correlation pattern mining")
    print("• Sky localization analysis")
    print("=" * 50)
    
    hunt = AggressiveCosmicStringHunt()
    success = hunt.run_aggressive_hunt()
    
    if success:
        print("\n🚀 AGGRESSIVE HUNT COMPLETE!")
        print("Results saved to: aggressive_hunt_results/")
        print("Check the visualization: aggressive_hunt_results/aggressive_hunt_visualization.png")
    else:
        print("\n❌ AGGRESSIVE HUNT FAILED!")
        print("Check the logs for details.")

if __name__ == "__main__":
    main()
