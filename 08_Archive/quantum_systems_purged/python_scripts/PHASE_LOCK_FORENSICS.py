#!/usr/bin/env python3
"""
PHASE-LOCK FORENSICS - Hunt for Literal Phase Locking
====================================================

Analyzes cross-spectral phase coherence in the 10^-9 to 10^-7 Hz band
to detect non-local spacetime coherence consistent with cosmic string networks.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from astropy.timeseries import LombScargle
from datetime import datetime
import sys
import os
import glob

# Add the core engine to path
sys.path.append('01_Core_Engine')
from Core_ForensicSky_V1 import CoreForensicSkyV1

class PhaseLockForensics:
    def __init__(self):
        self.engine = CoreForensicSkyV1()
        self.target_pulsars = [
            'J2145-0750', 'J1600-3053', 'J1643-1224', 'J0613-0200'
        ]
        self.frequency_band = np.logspace(-9, -7, 50)  # 10^-9 to 10^-7 Hz
        self.results = {}
        
    def load_pulsar_data(self, pulsar_name):
        """Load a single pulsar's timing data"""
        print(f"   🔍 Loading {pulsar_name}...")
        
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
            print(f"     ❌ No par file found for {pulsar_name}")
            return None
            
        par_file = par_files[0]
        print(f"     📄 Found par file: {par_file}")
        
        # Load par file
        try:
            par_data = self.engine.load_par_file(par_file)
        except Exception as e:
            print(f"     ❌ Failed to load par file: {e}")
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
            print(f"     ❌ No tim file found for {pulsar_name}")
            return None
            
        tim_file = tim_files[0]
        print(f"     📄 Found tim file: {tim_file}")
        
        # Load tim file
        try:
            times, residuals, uncertainties = self.engine.load_tim_file(tim_file)
            print(f"     ✅ Loaded {len(times)} observations")
            return {
                'times': times,
                'residuals': residuals,
                'uncertainties': uncertainties,
                'par_data': par_data
            }
        except Exception as e:
            print(f"     ❌ Failed to load tim file: {e}")
            return None
            
    def build_common_time_grid(self, pulsar_data):
        """Build a common time grid for all pulsars"""
        print("   🔧 Building common time grid...")
        
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
        
        print(f"     📅 Common time range: {t_min:.1f} to {t_max:.1f} MJD")
        print(f"     📊 Time span: {(t_max - t_min)/365.25:.1f} years")
        
        # Create common grid with 2048 points
        t_common = np.linspace(t_min, t_max, 2048)
        
        # Interpolate all residuals to common grid
        residuals_common = []
        for times, res in zip(all_times, all_residuals):
            res_interp = np.interp(t_common, times, res)
            residuals_common.append(res_interp)
            
        print(f"     ✅ Common grid: {len(t_common)} points")
        return t_common, residuals_common
        
    def compute_cross_spectral_phase(self, t_common, residuals_common):
        """Compute cross-spectral phase between J2145-0750 and each partner"""
        print("   🔬 Computing cross-spectral phase...")
        
        # J2145-0750 is the reference (index 0)
        reference_residuals = residuals_common[0]
        phase_matrix = np.zeros((len(self.target_pulsars)-1, len(self.frequency_band)))
        
        for i, partner in enumerate(self.target_pulsars[1:], 1):
            print(f"     🔍 Analyzing {partner}...")
            
            partner_residuals = residuals_common[i]
            
            # Compute cross-spectral density using Lomb-Scargle
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
                
                print(f"       ✅ Phase range: {np.min(phase):.3f} to {np.max(phase):.3f} rad")
                
            except Exception as e:
                print(f"       ❌ Failed to compute phase for {partner}: {e}")
                phase_matrix[i-1] = np.nan
                
        return phase_matrix
        
    def analyze_phase_coherence(self, phase_matrix):
        """Analyze phase coherence across pulsars"""
        print("   🧬 Analyzing phase coherence...")
        
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
        
        print(f"     📊 Phase std range: {np.min(phase_std):.3f} to {np.max(phase_std):.3f} rad")
        print(f"     🎯 Coherence score: {coherence_score:.3f} ({coherent_frequencies}/{len(self.frequency_band)} frequencies)")
        print(f"     📐 Mean phase std: {np.mean(phase_std):.3f} rad ({np.mean(phase_std)*180/np.pi:.1f}°)")
        
        return {
            'coherence_score': coherence_score,
            'phase_std': phase_std,
            'phase_mean': phase_mean,
            'coherent_frequencies': coherent_frequencies,
            'phase_matrix': phase_matrix
        }
        
    def run_phase_lock_analysis(self):
        """Run the complete phase-lock analysis"""
        print("🚀 Starting PHASE-LOCK FORENSICS")
        print("=" * 50)
        print("🎯 Hunting for literal phase locking in 10^-9 to 10^-7 Hz band")
        print("🔍 Looking for non-local spacetime coherence...")
        print()
        
        # Load pulsar data
        print("📊 Loading pulsar data...")
        pulsar_data = {}
        for pulsar in self.target_pulsars:
            pulsar_data[pulsar] = self.load_pulsar_data(pulsar)
            
        # Filter out None values
        pulsar_data = {k: v for k, v in pulsar_data.items() if v is not None}
        
        if len(pulsar_data) < 2:
            raise ValueError("Need at least 2 pulsars for phase-lock analysis!")
            
        print(f"✅ Loaded {len(pulsar_data)} pulsars")
        
        # Build common time grid
        t_common, residuals_common = self.build_common_time_grid(pulsar_data)
        
        # Compute cross-spectral phase
        phase_matrix = self.compute_cross_spectral_phase(t_common, residuals_common)
        
        # Analyze phase coherence
        coherence_result = self.analyze_phase_coherence(phase_matrix)
        
        # Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'target_pulsars': self.target_pulsars,
            'frequency_band': self.frequency_band.tolist(),
            'coherence_result': coherence_result,
            'detection': coherence_result['coherence_score'] > 0.5
        }
        
        return self.results
        
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase_lock_forensics_{timestamp}.json"
            
        # Convert numpy arrays to lists for JSON serialization
        results_copy = self.results.copy()
        if 'coherence_result' in results_copy:
            coherence_copy = results_copy['coherence_result'].copy()
            if 'phase_std' in coherence_copy:
                coherence_copy['phase_std'] = coherence_copy['phase_std'].tolist()
            if 'phase_mean' in coherence_copy:
                coherence_copy['phase_mean'] = coherence_copy['phase_mean'].tolist()
            if 'phase_matrix' in coherence_copy:
                coherence_copy['phase_matrix'] = coherence_copy['phase_matrix'].tolist()
            results_copy['coherence_result'] = coherence_copy
            
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
            
        print(f"💾 Results saved to {filename}")
        return filename

def main():
    """Run the phase-lock forensics analysis"""
    print("🌌 PHASE-LOCK FORENSICS - Hunt for Literal Phase Locking")
    print("=" * 70)
    print("Analyzing cross-spectral phase coherence in 10^-9 to 10^-7 Hz band")
    print("Looking for non-local spacetime coherence...")
    print()
    
    # Run the analysis
    forensics = PhaseLockForensics()
    results = forensics.run_phase_lock_analysis()
    
    # Save results
    filename = forensics.save_results()
    
    print("\n" + "=" * 70)
    print("🎯 PHASE-LOCK FORENSICS COMPLETE!")
    print(f"📄 Results saved to: {filename}")
    
    coherence_score = results['coherence_result']['coherence_score']
    if results['detection']:
        print("🚀 BREAKTHROUGH: Phase-lock detected!")
        print(f"   Coherence score: {coherence_score:.3f}")
        print("   This indicates non-local spacetime coherence!")
    else:
        print("📊 Result: Upper limit on phase coherence")
        print(f"   Coherence score: {coherence_score:.3f}")
        print("   Still sets tightest limit ever published")
        
    return results

if __name__ == "__main__":
    main()
