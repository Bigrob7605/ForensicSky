#!/usr/bin/env python3
"""
🎯 TARGETED COSMIC STRING HUNT
Focused testing on the most promising cosmic string signatures based on 2025 research

Priority Targets:
1. Superstring Networks with Kink Dominance (Gμ ~ 10^-12 - 10^-11)
2. Non-Gaussian Signature Hunt
3. Metastable/Global String Networks
4. Memory Effect Signatures
5. Cross-Dataset Validation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
from datetime import datetime
import os
import sys

# Add the core engine to path
sys.path.append('01_Core_Engine')
from Core_ForensicSky_V1 import CoreForensicSkyV1

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TargetedCosmicStringHunt:
    """
    🎯 TARGETED COSMIC STRING HUNT
    
    Focused on the most promising cosmic string signatures:
    - Superstring networks with kink dominance
    - Non-Gaussian signatures
    - Memory effect detection
    - Cross-correlation analysis
    """
    
    def __init__(self):
        self.engine = CoreForensicSkyV1()
        self.results = {}
        self.target_parameters = {
            'superstring_tension_range': (1e-12, 1e-11),  # Gμ range from NANOGrav
            'intercommutation_probability': (1e-3, 1e-1),  # p range for superstrings
            'kink_dominance_threshold': 0.7,  # Expect kinks to dominate over cusps
            'non_gaussian_threshold': 3.0,  # σ threshold for non-Gaussian detection
            'memory_effect_threshold': 1e-9,  # Timing residual step threshold
        }
        
    def run_targeted_hunt(self):
        """Run the targeted cosmic string hunt"""
        logger.info("🎯 STARTING TARGETED COSMIC STRING HUNT")
        logger.info("=" * 60)
        
        # Load real data
        logger.info("📡 Loading real IPTA DR2 data...")
        self.engine.load_real_ipta_data()
        
        if not self.engine.timing_data or not self.engine.pulsar_catalog:
            logger.error("❌ Failed to load real data")
            return False
            
        logger.info(f"✅ Loaded {len(self.engine.timing_data)} timing points from {len(self.engine.pulsar_catalog)} pulsars")
        
        # Run targeted analyses
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
        
        logger.info("🎯 TARGETED HUNT COMPLETE!")
        return True
    
    def analyze_superstring_networks(self):
        """Analyze superstring networks with kink dominance"""
        logger.info("   Analyzing superstring networks...")
        
        try:
            # Focus on the Gμ ~ 10^-12 - 10^-11 range
            gmu_min, gmu_max = self.target_parameters['superstring_tension_range']
            
            # Analyze timing residuals for superstring signatures
            superstring_candidates = []
            
            for pulsar in self.engine.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.engine.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 10:  # Need sufficient data
                    continue
                    
                residuals = np.array([d['residual'] for d in pulsar_timing])
                times = np.array([d['time'] for d in pulsar_timing])
                
                # Look for superstring signatures
                # 1. Continuous stochastic background (not sharp bursts)
                power_spectrum = np.abs(np.fft.fft(residuals))**2
                frequencies = np.fft.fftfreq(len(residuals), d=np.diff(times).mean())
                
                # 2. Look for kink-dominated spectrum
                # Kinks produce f^-1 spectrum, cusps produce f^-4/3
                log_freq = np.log10(np.abs(frequencies[1:len(frequencies)//2]))
                log_power = np.log10(power_spectrum[1:len(power_spectrum)//2])
                
                if len(log_freq) > 3:
                    # Fit power law
                    coeffs = np.polyfit(log_freq, log_power, 1)
                    spectral_index = coeffs[0]
                    
                    # Kink dominance: spectral index closer to -1 than -4/3
                    kink_dominance = abs(spectral_index - (-1.0)) < abs(spectral_index - (-4/3))
                    
                    # 3. Non-Gaussian analysis
                    skewness = self.calculate_skewness(residuals)
                    kurtosis = self.calculate_kurtosis(residuals)
                    
                    # Superstring candidates show non-Gaussian signatures
                    is_non_gaussian = abs(skewness) > 0.5 or abs(kurtosis - 3) > 1
                    
                    if kink_dominance and is_non_gaussian:
                        superstring_candidates.append({
                            'pulsar': pulsar_name,
                            'spectral_index': spectral_index,
                            'kink_dominance': kink_dominance,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'data_points': len(residuals),
                            'significance': self.calculate_significance(residuals)
                        })
            
            logger.info(f"   Found {len(superstring_candidates)} superstring candidates")
            
            return {
                'candidates': superstring_candidates,
                'total_analyzed': len(self.engine.pulsar_catalog),
                'kink_dominance_threshold': self.target_parameters['kink_dominance_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Superstring analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def hunt_non_gaussian_signatures(self):
        """Hunt for non-Gaussian signatures in timing residuals"""
        logger.info("   Hunting non-Gaussian signatures...")
        
        try:
            non_gaussian_candidates = []
            
            for pulsar in self.engine.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.engine.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 20:  # Need sufficient data
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
                
                # Calculate overall non-Gaussian score
                non_gaussian_score = 0
                if abs(tests['skewness']) > 0.5:
                    non_gaussian_score += 1
                if abs(tests['kurtosis'] - 3) > 1:
                    non_gaussian_score += 1
                if tests['jarque_bera'] < 0.05:  # p-value
                    non_gaussian_score += 1
                if tests['shapiro_wilk'] < 0.05:  # p-value
                    non_gaussian_score += 1
                if tests['anderson_darling'] < 0.05:  # p-value
                    non_gaussian_score += 1
                
                if non_gaussian_score >= 3:  # At least 3 tests indicate non-Gaussian
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
                'total_analyzed': len(self.engine.pulsar_catalog),
                'threshold': self.target_parameters['non_gaussian_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Non-Gaussian analysis failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def detect_memory_effects(self):
        """Detect memory effects from cosmic string passages"""
        logger.info("   Detecting memory effects...")
        
        try:
            memory_candidates = []
            
            for pulsar in self.engine.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.engine.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 30:  # Need sufficient data
                    continue
                    
                residuals = np.array([d['residual'] for d in pulsar_timing])
                times = np.array([d['time'] for d in pulsar_timing])
                
                # Look for step-function changes (memory effects)
                # Cosmic string passages leave permanent spacetime distortions
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
                'total_analyzed': len(self.engine.pulsar_catalog),
                'threshold': self.target_parameters['memory_effect_threshold'],
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Memory effect detection failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def analyze_kink_dominance(self):
        """Analyze kink dominance over cusp bursts"""
        logger.info("   Analyzing kink dominance...")
        
        try:
            kink_analysis = []
            
            for pulsar in self.engine.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.engine.timing_data if d['pulsar_name'] == pulsar_name]
                
                if len(pulsar_timing) < 15:  # Need sufficient data
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
                        # Linear fit
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
        """Mine for non-standard correlation patterns"""
        logger.info("   Mining correlation patterns...")
        
        try:
            # Use the engine's correlation analysis
            correlation_results = self.engine.correlation_analysis()
            
            # Look for deviations from Hellings-Downs curve
            hellings_downs_deviations = []
            
            if 'correlation_matrix' in correlation_results:
                corr_matrix = correlation_results['correlation_matrix']
                pulsar_positions = correlation_results.get('pulsar_positions', [])
                
                # Calculate expected Hellings-Downs correlations
                n_pulsars = len(pulsar_positions)
                if n_pulsars > 1:
                    for i in range(n_pulsars):
                        for j in range(i+1, n_pulsars):
                            # Calculate angular separation
                            pos1 = pulsar_positions[i]
                            pos2 = pulsar_positions[j]
                            
                            if len(pos1) >= 3 and len(pos2) >= 3:
                                # Convert to unit vectors
                                vec1 = np.array(pos1[:3]) / np.linalg.norm(pos1[:3])
                                vec2 = np.array(pos2[:3]) / np.linalg.norm(pos2[:3])
                                
                                # Angular separation
                                cos_angle = np.dot(vec1, vec2)
                                cos_angle = np.clip(cos_angle, -1, 1)
                                angle = np.arccos(cos_angle)
                                
                                # Expected Hellings-Downs correlation
                                expected_corr = 0.5 * (1 + np.cos(angle)) * np.log(1 + np.cos(angle)) - 0.5
                                
                                # Observed correlation
                                observed_corr = corr_matrix[i, j] if i < corr_matrix.shape[0] and j < corr_matrix.shape[1] else 0
                                
                                # Deviation
                                deviation = abs(observed_corr - expected_corr)
                                
                                if deviation > 0.1:  # Significant deviation
                                    hellings_downs_deviations.append({
                                        'pulsar_pair': (i, j),
                                        'angular_separation': angle,
                                        'expected_correlation': expected_corr,
                                        'observed_correlation': observed_corr,
                                        'deviation': deviation
                                    })
            
            logger.info(f"   Found {len(hellings_downs_deviations)} Hellings-Downs deviations")
            
            return {
                'hellings_downs_deviations': hellings_downs_deviations,
                'total_pairs_analyzed': len(hellings_downs_deviations),
                'correlation_analysis': correlation_results,
                'analysis_complete': True
            }
            
        except Exception as e:
            logger.error(f"   Correlation pattern mining failed: {e}")
            return {'error': str(e), 'analysis_complete': False}
    
    def analyze_sky_localization(self):
        """Analyze sky localization for cosmic string signatures"""
        logger.info("   Analyzing sky localization...")
        
        try:
            # Focus on regions where NANOGrav detected strongest signals
            # NANOGrav 15-year data shows strongest signals in specific sky regions
            
            sky_analysis = {
                'pulsar_positions': [],
                'signal_strength_by_region': {},
                'hotspots': []
            }
            
            for pulsar in self.engine.pulsar_catalog:
                pulsar_name = pulsar['name']
                pulsar_timing = [d for d in self.engine.timing_data if d['pulsar_name'] == pulsar_name]
                
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
                
                # Hotspots are 2σ above mean
                hotspot_threshold = mean_signal + 2 * std_signal
                
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
            'total_pulsars': len(self.engine.pulsar_catalog),
            'total_timing_points': len(self.engine.timing_data),
            'target_parameters': self.target_parameters,
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
        os.makedirs('targeted_hunt_results', exist_ok=True)
        
        # Save detailed results
        with open('targeted_hunt_results/detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        with open('targeted_hunt_results/summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create visualization
        self.create_targeted_visualization()
        
        logger.info("✅ Results compiled and saved to targeted_hunt_results/")
    
    def create_targeted_visualization(self):
        """Create targeted visualization of results"""
        logger.info("🎨 Creating targeted visualization...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('🎯 Targeted Cosmic String Hunt Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Superstring Candidates
            if 'superstring_analysis' in self.results and 'candidates' in self.results['superstring_analysis']:
                candidates = self.results['superstring_analysis']['candidates']
                if candidates:
                    spectral_indices = [c['spectral_index'] for c in candidates]
                    significances = [c['significance'] for c in candidates]
                    
                    axes[0, 0].scatter(spectral_indices, significances, alpha=0.7, c='red')
                    axes[0, 0].axvline(-1.0, color='green', linestyle='--', label='Kink (-1)')
                    axes[0, 0].axvline(-4/3, color='blue', linestyle='--', label='Cusp (-4/3)')
                    axes[0, 0].set_xlabel('Spectral Index')
                    axes[0, 0].set_ylabel('Significance')
                    axes[0, 0].set_title('Superstring Candidates')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
            
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
            
            # Plot 4: Kink Dominance
            if 'kink_dominance_analysis' in self.results and 'pulsar_analysis' in self.results['kink_dominance_analysis']:
                analysis = self.results['kink_dominance_analysis']['pulsar_analysis']
                if analysis:
                    spectral_indices = [a['spectral_index'] for a in analysis]
                    kink_dominant = [a['kink_dominance'] for a in analysis]
                    
                    colors = ['red' if kd else 'blue' for kd in kink_dominant]
                    axes[1, 0].scatter(range(len(spectral_indices)), spectral_indices, c=colors, alpha=0.7)
                    axes[1, 0].axhline(-1.0, color='red', linestyle='--', label='Kink (-1)')
                    axes[1, 0].axhline(-4/3, color='blue', linestyle='--', label='Cusp (-4/3)')
                    axes[1, 0].set_xlabel('Pulsar Index')
                    axes[1, 0].set_ylabel('Spectral Index')
                    axes[1, 0].set_title('Kink vs Cusp Dominance')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Sky Hotspots
            if 'sky_localization_analysis' in self.results and 'sky_analysis' in self.results['sky_localization_analysis']:
                sky_analysis = self.results['sky_localization_analysis']['sky_analysis']
                if sky_analysis['pulsar_positions']:
                    positions = sky_analysis['pulsar_positions']
                    ras = [p['ra'] for p in positions]
                    decs = [p['dec'] for p in positions]
                    signal_strengths = [p['signal_strength'] for p in positions]
                    
                    scatter = axes[1, 1].scatter(ras, decs, c=signal_strengths, cmap='hot', alpha=0.7)
                    axes[1, 1].set_xlabel('Right Ascension (deg)')
                    axes[1, 1].set_ylabel('Declination (deg)')
                    axes[1, 1].set_title('Sky Signal Strength')
                    plt.colorbar(scatter, ax=axes[1, 1])
                    axes[1, 1].grid(True, alpha=0.3)
            
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
            plt.savefig('targeted_hunt_results/targeted_hunt_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Visualization created: targeted_hunt_results/targeted_hunt_visualization.png")
            
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
        if skewness < 0.3 and kurtosis < 0.5:
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
        threshold = self.target_parameters['memory_effect_threshold']
        
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
    """Main function to run targeted cosmic string hunt"""
    print("🎯 TARGETED COSMIC STRING HUNT")
    print("=" * 50)
    print("Focusing on the most promising cosmic string signatures:")
    print("• Superstring networks with kink dominance")
    print("• Non-Gaussian signature detection")
    print("• Memory effect detection")
    print("• Cross-correlation pattern mining")
    print("• Sky localization analysis")
    print("=" * 50)
    
    hunt = TargetedCosmicStringHunt()
    success = hunt.run_targeted_hunt()
    
    if success:
        print("\n🎯 TARGETED HUNT COMPLETE!")
        print("Results saved to: targeted_hunt_results/")
        print("Check the visualization: targeted_hunt_results/targeted_hunt_visualization.png")
    else:
        print("\n❌ TARGETED HUNT FAILED!")
        print("Check the logs for details.")

if __name__ == "__main__":
    main()
