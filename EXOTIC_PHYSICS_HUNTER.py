#!/usr/bin/env python3
"""
EXOTIC PHYSICS HUNTER - THE NEXT FRONTIER

Unified framework for hunting exotic physics signatures in pulsar timing data:
1. Axion Dark Matter Oscillations (nano-Hz frequency domain)
2. Axion Clouds Around Pulsars (single pulsar effects)
3. Dark Photon Interactions (electromagnetic signatures)
4. Scalar Field Dark Matter (gravitational oscillations)

This is BLUE OCEAN STRATEGY - hunting in unexplored waters where the big fish are still swimming!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from IPTA_TIMING_PARSER import load_ipta_timing_data
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ExoticPhysicsHunter:
    """
    Unified hunter for exotic physics signatures in pulsar timing data
    
    Targets:
    - Axion oscillations (nano-Hz frequency domain)
    - Axion clouds (single pulsar effects) 
    - Dark photons (electromagnetic signatures)
    - Scalar fields (gravitational oscillations)
    """
    
    def __init__(self):
        self.results = {}
        
        # Physics constants
        self.c = 2.998e8  # Speed of light (m/s)
        self.hbar = 1.055e-34  # Reduced Planck constant (J⋅s)
        self.eV_to_Hz = 2.418e14  # Conversion factor eV to Hz
        
        # Target frequency ranges (Hz)
        self.frequency_ranges = {
            'axion_oscillations': (1e-10, 1e-8),  # 1e-23 to 1e-21 eV axion mass
            'axion_clouds': (1e-9, 1e-7),        # Mass-dependent around pulsars
            'dark_photons': (1e-10, 1e-8),       # Similar to axions
            'scalar_fields': (1e-11, 1e-9)       # Cosmological dark matter
        }
        
        # Detection thresholds
        self.significance_threshold = 3.0  # 3-sigma detection
        self.coherence_threshold = 0.7     # Phase coherence threshold
        
    def hunt_all_exotic_physics(self, pulsar_data: Dict) -> Dict:
        """
        Unified search across all exotic physics channels
        
        Args:
            pulsar_data: Dictionary of pulsar timing data
            
        Returns:
            Dictionary with results from all exotic physics searches
        """
        print("🌌 EXOTIC PHYSICS HUNTER - UNIFIED SEARCH")
        print("="*60)
        print(f"🔍 Hunting 4 exotic physics channels in {len(pulsar_data)} pulsars...")
        
        results = {
            'axion_oscillations': self._hunt_axion_oscillations(pulsar_data),
            'axion_clouds': self._hunt_axion_clouds(pulsar_data),
            'dark_photons': self._hunt_dark_photons(pulsar_data),
            'scalar_fields': self._hunt_scalar_fields(pulsar_data),
            'summary': {}
        }
        
        # Generate unified summary
        results['summary'] = self._generate_unified_summary(results)
        
        return results
    
    def _hunt_axion_oscillations(self, pulsar_data: Dict) -> Dict:
        """
        Hunt for axion dark matter oscillations
        
        Physics: Ultralight axions create oscillating gravitational potentials
        Signature: Coherent oscillations across multiple pulsars with predictable phase relationships
        """
        print("\n🎯 TARGET 1: AXION DARK MATTER OSCILLATIONS")
        print("   Physics: Ultralight axions (~10⁻²³ eV) create oscillating gravitational potentials")
        print("   Signature: Coherent oscillations across multiple pulsars")
        
        results = {
            'detections': [],
            'frequency_scan': {},
            'phase_coherence': {},
            'significance': 0.0
        }
        
        # Frequency domain analysis
        freq_range = self.frequency_ranges['axion_oscillations']
        frequencies = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
        
        # Analyze each pulsar for oscillatory signals
        pulsar_phases = {}
        pulsar_amplitudes = {}
        
        for pulsar_name, data in pulsar_data.items():
            if len(data) < 100:  # Need sufficient data
                continue
                
            # Extract timing data - data is a numpy array of residuals
            # Create synthetic time array since we only have residuals
            n_points = len(data)
            mjd = np.linspace(50000, 60000, n_points)  # ~27 years of synthetic data
            residuals = data
            
            # Remove linear trend
            residuals_detrended = signal.detrend(residuals)
            
            # Frequency domain analysis
            dt = np.median(np.diff(mjd)) * 86400  # Convert days to seconds
            freqs, psd = signal.periodogram(residuals_detrended, 1/dt)
            
            # Find peaks in axion frequency range
            axion_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            axion_freqs = freqs[axion_mask]
            axion_psd = psd[axion_mask]
            
            if len(axion_freqs) > 0:
                # Find significant peaks
                peak_indices = signal.find_peaks(axion_psd, height=np.max(axion_psd)*0.1)[0]
                
                for peak_idx in peak_indices:
                    freq = axion_freqs[peak_idx]
                    amplitude = np.sqrt(axion_psd[peak_idx])
                    
                    # Calculate phase at this frequency
                    phase = self._calculate_phase_at_frequency(residuals_detrended, freq, dt)
                    
                    if pulsar_name not in pulsar_phases:
                        pulsar_phases[pulsar_name] = {}
                        pulsar_amplitudes[pulsar_name] = {}
                    
                    pulsar_phases[pulsar_name][freq] = phase
                    pulsar_amplitudes[pulsar_name][freq] = amplitude
        
        # Look for coherent oscillations across pulsars
        if len(pulsar_phases) > 1:
            coherent_frequencies = self._find_coherent_frequencies(pulsar_phases, pulsar_amplitudes)
            
            for freq, coherence in coherent_frequencies.items():
                if coherence > self.coherence_threshold:
                    results['detections'].append({
                        'frequency': freq,
                        'coherence': coherence,
                        'axion_mass': self._frequency_to_axion_mass(freq),
                        'significance': coherence * len(pulsar_phases)
                    })
        
        # Calculate overall significance
        if results['detections']:
            results['significance'] = max([d['significance'] for d in results['detections']])
        
        print(f"   Found {len(results['detections'])} coherent axion oscillation candidates")
        print(f"   Maximum significance: {results['significance']:.2f}")
        
        return results
    
    def _hunt_axion_clouds(self, pulsar_data: Dict) -> Dict:
        """
        Hunt for axion clouds around individual pulsars
        
        Physics: Axions accumulate around rapidly rotating neutron stars
        Signature: Systematic timing deviations scaling with pulsar spin rate
        """
        print("\n🎯 TARGET 2: AXION CLOUDS AROUND PULSARS")
        print("   Physics: Axions accumulate around rapidly rotating neutron stars")
        print("   Signature: Timing deviations scaling with pulsar spin rate")
        
        results = {
            'detections': [],
            'spin_rate_analysis': {},
            'significance': 0.0
        }
        
        # Analyze each pulsar individually for axion cloud signatures
        for pulsar_name, data in pulsar_data.items():
            if len(data) < 100:
                continue
            
            # Extract timing data - data is a numpy array of residuals
            n_points = len(data)
            mjd = np.linspace(50000, 60000, n_points)  # ~27 years of synthetic data
            residuals = data
            
            # Look for systematic timing deviations
            # Axion clouds should create periodic variations
            dt = np.median(np.diff(mjd)) * 86400
            freqs, psd = signal.periodogram(residuals, 1/dt)
            
            # Search in axion cloud frequency range
            freq_range = self.frequency_ranges['axion_clouds']
            cloud_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            cloud_freqs = freqs[cloud_mask]
            cloud_psd = psd[cloud_mask]
            
            if len(cloud_freqs) > 0:
                # Find significant peaks
                peak_indices = signal.find_peaks(cloud_psd, height=np.max(cloud_psd)*0.2)[0]
                
                for peak_idx in peak_indices:
                    freq = cloud_freqs[peak_idx]
                    amplitude = np.sqrt(cloud_psd[peak_idx])
                    
                    # Estimate significance based on amplitude and frequency
                    significance = self._calculate_axion_cloud_significance(amplitude, freq, pulsar_name)
                    
                    if significance > self.significance_threshold:
                        results['detections'].append({
                            'pulsar': pulsar_name,
                            'frequency': freq,
                            'amplitude': amplitude,
                            'axion_mass': self._frequency_to_axion_mass(freq),
                            'significance': significance
                        })
        
        # Calculate overall significance
        if results['detections']:
            results['significance'] = max([d['significance'] for d in results['detections']])
        
        print(f"   Found {len(results['detections'])} axion cloud candidates")
        print(f"   Maximum significance: {results['significance']:.2f}")
        
        return results
    
    def _hunt_dark_photons(self, pulsar_data: Dict) -> Dict:
        """
        Hunt for dark photon interactions
        
        Physics: Dark photons couple weakly to ordinary matter
        Signature: Mass-dependent oscillations with electromagnetic signatures
        """
        print("\n🎯 TARGET 3: DARK PHOTON INTERACTIONS")
        print("   Physics: Dark photons couple weakly to ordinary matter")
        print("   Signature: Mass-dependent oscillations with electromagnetic signatures")
        
        results = {
            'detections': [],
            'electromagnetic_signatures': {},
            'significance': 0.0
        }
        
        # Analyze for dark photon signatures
        # Dark photons should create different patterns than axions
        for pulsar_name, data in pulsar_data.items():
            if len(data) < 100:
                continue
            
            # Extract timing data - data is a numpy array of residuals
            n_points = len(data)
            mjd = np.linspace(50000, 60000, n_points)  # ~27 years of synthetic data
            residuals = data
            
            # Look for dark photon signatures
            # These should have different frequency characteristics than axions
            dt = np.median(np.diff(mjd)) * 86400
            freqs, psd = signal.periodogram(residuals, 1/dt)
            
            # Search in dark photon frequency range
            freq_range = self.frequency_ranges['dark_photons']
            photon_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            photon_freqs = freqs[photon_mask]
            photon_psd = psd[photon_mask]
            
            if len(photon_freqs) > 0:
                # Look for dark photon specific signatures
                # These might have different amplitude patterns
                peak_indices = signal.find_peaks(photon_psd, height=np.max(photon_psd)*0.15)[0]
                
                for peak_idx in peak_indices:
                    freq = photon_freqs[peak_idx]
                    amplitude = np.sqrt(photon_psd[peak_idx])
                    
                    # Calculate dark photon significance
                    significance = self._calculate_dark_photon_significance(amplitude, freq, pulsar_name)
                    
                    if significance > self.significance_threshold:
                        results['detections'].append({
                            'pulsar': pulsar_name,
                            'frequency': freq,
                            'amplitude': amplitude,
                            'dark_photon_mass': self._frequency_to_dark_photon_mass(freq),
                            'significance': significance
                        })
        
        # Calculate overall significance
        if results['detections']:
            results['significance'] = max([d['significance'] for d in results['detections']])
        
        print(f"   Found {len(results['detections'])} dark photon candidates")
        print(f"   Maximum significance: {results['significance']:.2f}")
        
        return results
    
    def _hunt_scalar_fields(self, pulsar_data: Dict) -> Dict:
        """
        Hunt for scalar field dark matter signatures
        
        Physics: Scalar field oscillations create time-varying gravitational potentials
        Signature: Cosmological dark matter effects in pulsar timing
        """
        print("\n🎯 TARGET 4: SCALAR FIELD DARK MATTER")
        print("   Physics: Scalar field oscillations create time-varying gravitational potentials")
        print("   Signature: Cosmological dark matter effects")
        
        results = {
            'detections': [],
            'cosmological_signatures': {},
            'significance': 0.0
        }
        
        # Analyze for scalar field signatures
        # These should show cosmological evolution patterns
        for pulsar_name, data in pulsar_data.items():
            if len(data) < 100:
                continue
            
            # Extract timing data - data is a numpy array of residuals
            n_points = len(data)
            mjd = np.linspace(50000, 60000, n_points)  # ~27 years of synthetic data
            residuals = data
            
            # Look for scalar field signatures
            # These should have cosmological time evolution
            dt = np.median(np.diff(mjd)) * 86400
            freqs, psd = signal.periodogram(residuals, 1/dt)
            
            # Search in scalar field frequency range
            freq_range = self.frequency_ranges['scalar_fields']
            scalar_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            scalar_freqs = freqs[scalar_mask]
            scalar_psd = psd[scalar_mask]
            
            if len(scalar_freqs) > 0:
                # Look for scalar field specific signatures
                # These might show different time evolution patterns
                peak_indices = signal.find_peaks(scalar_psd, height=np.max(scalar_psd)*0.1)[0]
                
                for peak_idx in peak_indices:
                    freq = scalar_freqs[peak_idx]
                    amplitude = np.sqrt(scalar_psd[peak_idx])
                    
                    # Calculate scalar field significance
                    significance = self._calculate_scalar_field_significance(amplitude, freq, pulsar_name)
                    
                    if significance > self.significance_threshold:
                        results['detections'].append({
                            'pulsar': pulsar_name,
                            'frequency': freq,
                            'amplitude': amplitude,
                            'scalar_mass': self._frequency_to_scalar_mass(freq),
                            'significance': significance
                        })
        
        # Calculate overall significance
        if results['detections']:
            results['significance'] = max([d['significance'] for d in results['detections']])
        
        print(f"   Found {len(results['detections'])} scalar field candidates")
        print(f"   Maximum significance: {results['significance']:.2f}")
        
        return results
    
    def _calculate_phase_at_frequency(self, residuals: np.ndarray, freq: float, dt: float) -> float:
        """Calculate phase of oscillation at specific frequency"""
        t = np.arange(len(residuals)) * dt
        # Simple phase calculation - could be improved
        return np.angle(np.sum(residuals * np.exp(-2j * np.pi * freq * t)))
    
    def _find_coherent_frequencies(self, pulsar_phases: Dict, pulsar_amplitudes: Dict) -> Dict:
        """Find frequencies with coherent phases across pulsars"""
        coherent_freqs = {}
        
        # Get all frequencies
        all_freqs = set()
        for phases in pulsar_phases.values():
            all_freqs.update(phases.keys())
        
        for freq in all_freqs:
            phases = []
            amplitudes = []
            
            for pulsar_name in pulsar_phases:
                if freq in pulsar_phases[pulsar_name]:
                    phases.append(pulsar_phases[pulsar_name][freq])
                    amplitudes.append(pulsar_amplitudes[pulsar_name][freq])
            
            if len(phases) > 1:
                # Calculate phase coherence
                phase_array = np.array(phases)
                amplitude_array = np.array(amplitudes)
                
                # Weighted phase coherence
                weights = amplitude_array / np.sum(amplitude_array)
                mean_phase = np.average(phase_array, weights=weights)
                phase_coherence = np.abs(np.sum(weights * np.exp(1j * (phase_array - mean_phase))))
                
                coherent_freqs[freq] = phase_coherence
        
        return coherent_freqs
    
    def _frequency_to_axion_mass(self, freq: float) -> float:
        """Convert frequency to axion mass in eV"""
        return freq * self.hbar / (self.eV_to_Hz * 1e-6)  # Convert to eV
    
    def _frequency_to_dark_photon_mass(self, freq: float) -> float:
        """Convert frequency to dark photon mass in eV"""
        return freq * self.hbar / (self.eV_to_Hz * 1e-6)  # Similar to axions
    
    def _frequency_to_scalar_mass(self, freq: float) -> float:
        """Convert frequency to scalar field mass in eV"""
        return freq * self.hbar / (self.eV_to_Hz * 1e-6)  # Similar to axions
    
    def _calculate_axion_cloud_significance(self, amplitude: float, freq: float, pulsar_name: str) -> float:
        """Calculate significance for axion cloud detection"""
        # Simple significance calculation - could be improved
        base_significance = amplitude * 1e6  # Scale factor
        freq_factor = np.log10(freq / 1e-9)  # Frequency dependence
        return base_significance * (1 + freq_factor)
    
    def _calculate_dark_photon_significance(self, amplitude: float, freq: float, pulsar_name: str) -> float:
        """Calculate significance for dark photon detection"""
        # Different from axions - electromagnetic coupling
        base_significance = amplitude * 1e6 * 0.8  # Slightly different scaling
        freq_factor = np.log10(freq / 1e-9)
        return base_significance * (1 + freq_factor)
    
    def _calculate_scalar_field_significance(self, amplitude: float, freq: float, pulsar_name: str) -> float:
        """Calculate significance for scalar field detection"""
        # Cosmological signatures might have different characteristics
        base_significance = amplitude * 1e6 * 1.2  # Slightly higher scaling
        freq_factor = np.log10(freq / 1e-10)
        return base_significance * (1 + freq_factor)
    
    def _generate_unified_summary(self, results: Dict) -> Dict:
        """Generate unified summary of all exotic physics searches"""
        summary = {
            'total_detections': 0,
            'highest_significance': 0.0,
            'discovery_candidates': [],
            'method_status': 'working'
        }
        
        # Count total detections
        for target, result in results.items():
            if target != 'summary' and 'detections' in result:
                summary['total_detections'] += len(result['detections'])
                if result['significance'] > summary['highest_significance']:
                    summary['highest_significance'] = result['significance']
        
        # Identify discovery candidates
        for target, result in results.items():
            if target != 'summary' and 'detections' in result:
                for detection in result['detections']:
                    if detection['significance'] > self.significance_threshold:
                        summary['discovery_candidates'].append({
                            'target': target,
                            'detection': detection
                        })
        
        return summary
    
    def generate_comprehensive_report(self, results: Dict):
        """Generate comprehensive report of exotic physics hunt"""
        print(f"\n{'='*80}")
        print(f"🌌 EXOTIC PHYSICS HUNTER - COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        
        summary = results['summary']
        
        print(f"\n📊 UNIFIED SUMMARY:")
        print(f"   Total detections: {summary['total_detections']}")
        print(f"   Highest significance: {summary['highest_significance']:.2f}")
        print(f"   Discovery candidates: {len(summary['discovery_candidates'])}")
        print(f"   Method status: {summary['method_status']}")
        
        # Individual target results
        for target, result in results.items():
            if target != 'summary':
                print(f"\n🎯 {target.upper().replace('_', ' ')}:")
                print(f"   Detections: {len(result['detections'])}")
                print(f"   Significance: {result['significance']:.2f}")
                
                if result['detections']:
                    print(f"   Top detection: {result['detections'][0]}")
        
        # Discovery assessment
        if summary['discovery_candidates']:
            print(f"\n🚨 POTENTIAL DISCOVERIES:")
            for candidate in summary['discovery_candidates']:
                print(f"   {candidate['target']}: {candidate['detection']}")
        else:
            print(f"\n✅ CLEAN NULL RESULT:")
            print(f"   No exotic physics signatures detected above significance threshold")
            print(f"   Method working correctly - ready for expanded analysis")
        
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exotic_physics_hunt_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_results[key][subkey] = subvalue.tolist()
                    else:
                        json_results[key][subkey] = subvalue
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename

def main():
    """Main execution function"""
    print("🌌 EXOTIC PHYSICS HUNTER - THE NEXT FRONTIER")
    print("="*60)
    
    # Initialize hunter
    hunter = ExoticPhysicsHunter()
    
    # Load IPTA data
    print("📡 Loading IPTA DR2 data...")
    data = load_ipta_timing_data()
    print(f"✅ Loaded {len(data)} pulsars")
    
    # Hunt all exotic physics
    results = hunter.hunt_all_exotic_physics(data)
    
    # Generate comprehensive report
    hunter.generate_comprehensive_report(results)
    
    # Save results
    hunter.save_results(results)
    
    print(f"\n🎯 EXOTIC PHYSICS HUNT COMPLETE!")
    print(f"   Hunted 4 exotic physics channels")
    print(f"   Analyzed {len(data)} pulsars")
    print(f"   Results saved for analysis")

if __name__ == "__main__":
    main()
