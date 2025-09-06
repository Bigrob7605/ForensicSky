#!/usr/bin/env python3
"""
🔍 PATTERN ANALYSIS SCRIPT
Analyze statistical anomalies for coherent patterns that could indicate real physics

This script examines the statistical anomalies found in the cosmic string analysis
to identify any coherent patterns that might indicate real astrophysical signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy import stats
from scipy.signal import find_peaks, correlate
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    🔍 PATTERN ANALYZER
    
    Analyzes statistical anomalies for coherent patterns that could indicate real physics
    """
    
    def __init__(self, results_file="real_ipta_hunt_results/detailed_results.json"):
        self.results_file = results_file
        self.results = None
        self.patterns = {}
        
    def load_results(self):
        """Load the detection results"""
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            logger.info("✅ Results loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load results: {e}")
            return False
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in burst events and memory effects"""
        logger.info("🕒 Analyzing temporal patterns...")
        
        patterns = {
            'burst_temporal': {},
            'memory_temporal': {},
            'cross_pulsar_timing': {}
        }
        
        # Analyze burst events
        if 'burst_analysis' in self.results:
            for candidate in self.results['burst_analysis']['candidates']:
                pulsar = candidate['pulsar']
                bursts = candidate['burst_events']
                
                if len(bursts) > 1:
                    # Extract times and amplitudes
                    times = [burst['time'] for burst in bursts]
                    amplitudes = [burst['amplitude'] for burst in bursts]
                    
                    # Calculate time intervals between bursts
                    time_intervals = np.diff(sorted(times))
                    
                    # Look for periodic patterns
                    if len(time_intervals) > 2:
                        # Check for regular intervals
                        interval_std = np.std(time_intervals)
                        interval_mean = np.mean(time_intervals)
                        regularity = 1 - (interval_std / interval_mean) if interval_mean > 0 else 0
                        
                        # Check for clustering
                        clustering_score = self._calculate_clustering_score(times)
                        
                        patterns['burst_temporal'][pulsar] = {
                            'n_bursts': len(bursts),
                            'time_span': max(times) - min(times),
                            'mean_interval': interval_mean,
                            'interval_std': interval_std,
                            'regularity': regularity,
                            'clustering_score': clustering_score,
                            'amplitude_range': [min(amplitudes), max(amplitudes)],
                            'amplitude_std': np.std(amplitudes)
                        }
        
        # Analyze memory effects
        if 'memory_effect_analysis' in self.results:
            for candidate in self.results['memory_effect_analysis']['candidates']:
                pulsar = candidate['pulsar']
                steps = candidate['step_detections']
                
                if len(steps) > 1:
                    # Extract times and step sizes
                    times = [step['time'] for step in steps]
                    step_sizes = [step['step_size'] for step in steps]
                    
                    # Calculate time intervals between steps
                    time_intervals = np.diff(sorted(times))
                    
                    # Look for patterns in step sizes
                    step_size_std = np.std(step_sizes)
                    step_size_mean = np.mean(step_sizes)
                    
                    # Check for clustering
                    clustering_score = self._calculate_clustering_score(times)
                    
                    patterns['memory_temporal'][pulsar] = {
                        'n_steps': len(steps),
                        'time_span': max(times) - min(times),
                        'mean_interval': np.mean(time_intervals) if len(time_intervals) > 0 else 0,
                        'step_size_mean': step_size_mean,
                        'step_size_std': step_size_std,
                        'clustering_score': clustering_score
                    }
        
        # Cross-pulsar timing analysis
        if 'burst_analysis' in self.results and 'memory_effect_analysis' in self.results:
            patterns['cross_pulsar_timing'] = self._analyze_cross_pulsar_timing()
        
        return patterns
    
    def _calculate_clustering_score(self, times):
        """Calculate clustering score for temporal events"""
        if len(times) < 3:
            return 0
        
        times = np.array(sorted(times))
        intervals = np.diff(times)
        
        # Calculate coefficient of variation
        if np.mean(intervals) > 0:
            cv = np.std(intervals) / np.mean(intervals)
            # Lower CV indicates more regular spacing (less clustering)
            clustering_score = 1 - min(cv, 1)
        else:
            clustering_score = 0
        
        return clustering_score
    
    def _analyze_cross_pulsar_timing(self):
        """Analyze timing relationships between pulsars"""
        cross_timing = {}
        
        # Get burst times from all pulsars
        burst_times = {}
        if 'burst_analysis' in self.results:
            for candidate in self.results['burst_analysis']['candidates']:
                pulsar = candidate['pulsar']
                bursts = candidate['burst_events']
                burst_times[pulsar] = [burst['time'] for burst in bursts]
        
        # Get memory effect times from all pulsars
        memory_times = {}
        if 'memory_effect_analysis' in self.results:
            for candidate in self.results['memory_effect_analysis']['candidates']:
                pulsar = candidate['pulsar']
                steps = candidate['step_detections']
                memory_times[pulsar] = [step['time'] for step in steps]
        
        # Look for coincident events between pulsars
        pulsars = list(burst_times.keys())
        if len(pulsars) >= 2:
            for i, p1 in enumerate(pulsars):
                for p2 in pulsars[i+1:]:
                    # Check for burst coincidences
                    coincidences = self._find_coincident_events(
                        burst_times[p1], burst_times[p2], 
                        time_window=1.0  # 1 day window
                    )
                    
                    cross_timing[f"{p1}_{p2}_bursts"] = {
                        'coincidences': coincidences,
                        'n_p1': len(burst_times[p1]),
                        'n_p2': len(burst_times[p2])
                    }
        
        return cross_timing
    
    def _find_coincident_events(self, times1, times2, time_window=1.0):
        """Find coincident events within time window"""
        coincidences = []
        
        for t1 in times1:
            for t2 in times2:
                if abs(t1 - t2) <= time_window:
                    coincidences.append({
                        'time1': t1,
                        'time2': t2,
                        'time_diff': abs(t1 - t2)
                    })
        
        return coincidences
    
    def analyze_spectral_patterns(self):
        """Analyze spectral patterns in the data"""
        logger.info("📊 Analyzing spectral patterns...")
        
        patterns = {
            'spectral_consistency': {},
            'frequency_band_analysis': {},
            'spectral_index_patterns': {}
        }
        
        # Analyze spectral indices
        if 'superstring_analysis' in self.results:
            for candidate in self.results['superstring_analysis']['candidates']:
                pulsar = candidate['pulsar']
                spectral_index = candidate.get('spectral_index', 0)
                
                patterns['spectral_consistency'][pulsar] = {
                    'spectral_index': spectral_index,
                    'kink_dominance': candidate.get('kink_dominance', False),
                    'significance': candidate.get('significance', 0)
                }
        
        # Analyze frequency band patterns
        if 'frequency_band_analysis' in self.results:
            bands = self.results['frequency_band_analysis']['frequency_bands']
            for band_name, band_data in bands.items():
                patterns['frequency_band_analysis'][band_name] = {
                    'candidates_found': len(band_data.get('candidates', [])),
                    'frequency_range': band_data.get('frequency_range', []),
                    'total_analyzed': band_data.get('total_analyzed', 0)
                }
        
        return patterns
    
    def analyze_correlation_patterns(self):
        """Analyze correlation patterns between pulsars"""
        logger.info("🔗 Analyzing correlation patterns...")
        
        patterns = {
            'correlation_strength': {},
            'correlation_significance': {},
            'correlation_matrix_analysis': {}
        }
        
        if 'correlation_analysis' in self.results:
            corr_data = self.results['correlation_analysis']
            
            # Analyze significant correlations
            if 'significant_correlations' in corr_data:
                for corr in corr_data['significant_correlations']:
                    pair = corr['pulsar_pair']
                    correlation = corr['correlation']
                    significance = corr['significance']
                    
                    pair_key = f"{pair[0]}_{pair[1]}"
                    patterns['correlation_strength'][pair_key] = {
                        'correlation': correlation,
                        'significance': significance,
                        'abs_correlation': abs(correlation)
                    }
            
            # Analyze correlation matrix
            if 'correlation_matrix' in corr_data:
                matrix = np.array(corr_data['correlation_matrix'])
                pulsar_names = corr_data.get('pulsar_names', [])
                
                patterns['correlation_matrix_analysis'] = {
                    'matrix_shape': matrix.shape,
                    'mean_abs_correlation': np.mean(np.abs(matrix)),
                    'max_abs_correlation': np.max(np.abs(matrix)),
                    'pulsar_names': pulsar_names
                }
        
        return patterns
    
    def analyze_non_gaussian_patterns(self):
        """Analyze non-Gaussian patterns"""
        logger.info("📈 Analyzing non-Gaussian patterns...")
        
        patterns = {
            'non_gaussian_consistency': {},
            'moment_patterns': {},
            'test_agreement': {}
        }
        
        if 'non_gaussian_analysis' in self.results:
            for candidate in self.results['non_gaussian_analysis']['candidates']:
                pulsar = candidate['pulsar']
                tests = candidate['tests']
                
                # Extract moment values
                skewness = tests.get('skewness', 0)
                kurtosis = tests.get('kurtosis', 0)
                
                # Check test agreement
                test_values = [tests.get(test, 0) for test in ['jarque_bera', 'shapiro_wilk', 'anderson_darling']]
                test_agreement = len([v for v in test_values if v < 0.05])  # Count significant tests
                
                patterns['non_gaussian_consistency'][pulsar] = {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'test_agreement': test_agreement,
                    'non_gaussian_score': candidate.get('non_gaussian_score', 0)
                }
                
                patterns['moment_patterns'][pulsar] = {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'skewness_abs': abs(skewness),
                    'kurtosis_excess': kurtosis - 3  # Excess kurtosis
                }
        
        return patterns
    
    def identify_coherent_patterns(self):
        """Identify coherent patterns across all analyses"""
        logger.info("🎯 Identifying coherent patterns...")
        
        coherent_patterns = {
            'temporal_coherence': {},
            'spectral_coherence': {},
            'correlation_coherence': {},
            'non_gaussian_coherence': {},
            'overall_coherence': {}
        }
        
        # Analyze temporal coherence
        temporal_patterns = self.analyze_temporal_patterns()
        
        # Check for consistent temporal patterns across pulsars
        burst_regularity = []
        memory_clustering = []
        
        for pulsar, data in temporal_patterns.get('burst_temporal', {}).items():
            burst_regularity.append(data['regularity'])
        
        for pulsar, data in temporal_patterns.get('memory_temporal', {}).items():
            memory_clustering.append(data['clustering_score'])
        
        if burst_regularity:
            coherent_patterns['temporal_coherence']['burst_regularity_consistency'] = {
                'mean_regularity': np.mean(burst_regularity),
                'std_regularity': np.std(burst_regularity),
                'consistency_score': 1 - np.std(burst_regularity) if len(burst_regularity) > 1 else 1
            }
        
        if memory_clustering:
            coherent_patterns['temporal_coherence']['memory_clustering_consistency'] = {
                'mean_clustering': np.mean(memory_clustering),
                'std_clustering': np.std(memory_clustering),
                'consistency_score': 1 - np.std(memory_clustering) if len(memory_clustering) > 1 else 1
            }
        
        # Analyze spectral coherence
        spectral_patterns = self.analyze_spectral_patterns()
        
        # Check for consistent spectral indices
        spectral_indices = []
        for pulsar, data in spectral_patterns.get('spectral_consistency', {}).items():
            spectral_indices.append(data['spectral_index'])
        
        if spectral_indices:
            coherent_patterns['spectral_coherence']['spectral_index_consistency'] = {
                'mean_spectral_index': np.mean(spectral_indices),
                'std_spectral_index': np.std(spectral_indices),
                'consistency_score': 1 - np.std(spectral_indices) if len(spectral_indices) > 1 else 1
            }
        
        # Analyze correlation coherence
        correlation_patterns = self.analyze_correlation_patterns()
        
        # Check for consistent correlation strengths
        correlations = []
        for pair, data in correlation_patterns.get('correlation_strength', {}).items():
            correlations.append(data['abs_correlation'])
        
        if correlations:
            coherent_patterns['correlation_coherence']['correlation_consistency'] = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'consistency_score': 1 - np.std(correlations) if len(correlations) > 1 else 1
            }
        
        # Analyze non-Gaussian coherence
        non_gaussian_patterns = self.analyze_non_gaussian_patterns()
        
        # Check for consistent non-Gaussian signatures
        skewness_values = []
        kurtosis_values = []
        
        for pulsar, data in non_gaussian_patterns.get('moment_patterns', {}).items():
            skewness_values.append(data['skewness'])
            kurtosis_values.append(data['kurtosis'])
        
        if skewness_values and kurtosis_values:
            coherent_patterns['non_gaussian_coherence']['moment_consistency'] = {
                'mean_skewness': np.mean(skewness_values),
                'std_skewness': np.std(skewness_values),
                'mean_kurtosis': np.mean(kurtosis_values),
                'std_kurtosis': np.std(kurtosis_values),
                'skewness_consistency': 1 - np.std(skewness_values) if len(skewness_values) > 1 else 1,
                'kurtosis_consistency': 1 - np.std(kurtosis_values) if len(kurtosis_values) > 1 else 1
            }
        
        # Calculate overall coherence score
        coherence_scores = []
        for category, data in coherent_patterns.items():
            if category != 'overall_coherence' and data:
                for metric, values in data.items():
                    if 'consistency_score' in values:
                        coherence_scores.append(values['consistency_score'])
        
        if coherence_scores:
            coherent_patterns['overall_coherence'] = {
                'mean_coherence': np.mean(coherence_scores),
                'std_coherence': np.std(coherence_scores),
                'overall_score': np.mean(coherence_scores)
            }
        
        return coherent_patterns
    
    def generate_pattern_report(self):
        """Generate comprehensive pattern analysis report"""
        logger.info("📋 Generating pattern analysis report...")
        
        # Run all analyses
        temporal_patterns = self.analyze_temporal_patterns()
        spectral_patterns = self.analyze_spectral_patterns()
        correlation_patterns = self.analyze_correlation_patterns()
        non_gaussian_patterns = self.analyze_non_gaussian_patterns()
        coherent_patterns = self.identify_coherent_patterns()
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'temporal_patterns': temporal_patterns,
            'spectral_patterns': spectral_patterns,
            'correlation_patterns': correlation_patterns,
            'non_gaussian_patterns': non_gaussian_patterns,
            'coherent_patterns': coherent_patterns,
            'summary': self._generate_summary(coherent_patterns)
        }
        
        # Save report
        with open('pattern_analysis_results.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("✅ Pattern analysis report generated")
        return report
    
    def _generate_summary(self, coherent_patterns):
        """Generate summary of pattern analysis"""
        summary = {
            'overall_coherence_score': 0,
            'key_findings': [],
            'recommendations': []
        }
        
        if 'overall_coherence' in coherent_patterns:
            summary['overall_coherence_score'] = coherent_patterns['overall_coherence'].get('overall_score', 0)
        
        # Identify key findings
        if summary['overall_coherence_score'] > 0.7:
            summary['key_findings'].append("High coherence across multiple analysis types")
        elif summary['overall_coherence_score'] > 0.4:
            summary['key_findings'].append("Moderate coherence across some analysis types")
        else:
            summary['key_findings'].append("Low coherence - patterns may be spurious")
        
        # Add specific findings
        for category, data in coherent_patterns.items():
            if category != 'overall_coherence' and data:
                for metric, values in data.items():
                    if 'consistency_score' in values and values['consistency_score'] > 0.7:
                        summary['key_findings'].append(f"High consistency in {category}: {metric}")
        
        # Generate recommendations
        if summary['overall_coherence_score'] > 0.7:
            summary['recommendations'].append("High coherence suggests potential real signal - investigate further")
        elif summary['overall_coherence_score'] > 0.4:
            summary['recommendations'].append("Moderate coherence - run additional validation tests")
        else:
            summary['recommendations'].append("Low coherence - results likely spurious, focus on systematic error correction")
        
        return summary

def main():
    """Main function to run pattern analysis"""
    print("🔍 PATTERN ANALYSIS SCRIPT")
    print("=" * 50)
    print("Analyzing statistical anomalies for coherent patterns")
    print("• Temporal pattern analysis")
    print("• Spectral pattern analysis")
    print("• Correlation pattern analysis")
    print("• Non-Gaussian pattern analysis")
    print("• Coherent pattern identification")
    print("=" * 50)
    
    analyzer = PatternAnalyzer()
    
    if not analyzer.load_results():
        print("❌ Failed to load results")
        return
    
    report = analyzer.generate_pattern_report()
    
    print("\n🔍 PATTERN ANALYSIS COMPLETE!")
    print("=" * 50)
    
    # Print summary
    summary = report['summary']
    print(f"Overall Coherence Score: {summary['overall_coherence_score']:.3f}")
    print("\nKey Findings:")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    print(f"\nDetailed results saved to: pattern_analysis_results.json")

if __name__ == "__main__":
    main()
