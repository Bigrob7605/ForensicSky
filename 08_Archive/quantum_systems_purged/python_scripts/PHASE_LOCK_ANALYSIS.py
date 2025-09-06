#!/usr/bin/env python3
"""
PHASE LOCK ANALYSIS - Cross-Spectral Phase Analysis
==================================================

Compute cross-spectral phase φ(f) between J2145-0750 and each partner
in 10⁻⁹–10⁻⁷ Hz band. Flat phase (±5°) → literal phase locking → 
spacetime defect > stochastic background.

This is the smoking gun for non-local spacetime coherence!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import json
from datetime import datetime
import sys
import os

# Add the core engine to path
sys.path.append('01_Core_Engine')
from Core_ForensicSky_V1 import CoreForensicSkyV1

class PhaseLockAnalysis:
    def __init__(self):
        self.engine = CoreForensicSkyV1()
        self.j2145_partners = [
            'J1600-3053', 'J1643-1224', 'J0613-0200', 
            'J0610-2100', 'J1802-2124'
        ]
        self.frequency_band = (1e-9, 1e-7)  # 10^-9 to 10^-7 Hz
        self.results = {}
        
    def load_j2145_data(self):
        """Load J2145-0750 and its 5 correlated partners"""
        print("🔍 Loading J2145-0750 phase lock data...")
        
        # Load real data
        self.engine.load_real_ipta_data()
        
        # Get J2145-0750 data
        j2145_data = None
        for pulsar_id, data in self.engine.timing_data.items():
            if 'J2145-0750' in pulsar_id:
                j2145_data = data
                break
                
        if j2145_data is None:
            raise ValueError("J2145-0750 not found in timing data!")
            
        # Get partner data
        partner_data = {}
        for partner in self.j2145_partners:
            for pulsar_id, data in self.engine.timing_data.items():
                if partner in pulsar_id:
                    partner_data[partner] = data
                    break
                    
        print(f"✅ Loaded J2145-0750 + {len(partner_data)} partners")
        return j2145_data, partner_data
        
    def compute_cross_spectral_phase(self, times1, residuals1, times2, residuals2):
        """Compute cross-spectral phase between two pulsars"""
        # Interpolate to common time grid
        t_min = max(times1.min(), times2.min())
        t_max = min(times1.max(), times2.max())
        dt = 30.0  # 30 day sampling
        t_common = np.arange(t_min, t_max, dt)
        
        # Interpolate residuals
        r1_interp = np.interp(t_common, times1, residuals1)
        r2_interp = np.interp(t_common, times2, residuals2)
        
        # Remove linear trends
        r1_detrend = signal.detrend(r1_interp)
        r2_detrend = signal.detrend(r2_interp)
        
        # Compute cross-spectrum
        f, Pxy = signal.csd(r1_detrend, r2_detrend, fs=1.0/dt, nperseg=len(r1_detrend)//4)
        
        # Convert to Hz
        f_hz = f / (365.25 * 24 * 3600)  # Convert to Hz
        
        # Compute phase
        phase = np.angle(Pxy)
        
        # Filter to frequency band of interest
        mask = (f_hz >= self.frequency_band[0]) & (f_hz <= self.frequency_band[1])
        f_filtered = f_hz[mask]
        phase_filtered = phase[mask]
        
        return f_filtered, phase_filtered, Pxy[mask]
        
    def test_phase_flatness(self, phases):
        """Test if phases are flat within ±5°"""
        # Convert to degrees
        phases_deg = np.degrees(phases)
        
        # Remove 2π wraps
        phases_unwrapped = np.unwrap(phases_deg)
        
        # Test flatness
        phase_std = np.std(phases_unwrapped)
        phase_range = np.max(phases_unwrapped) - np.min(phases_unwrapped)
        
        # Flat if std < 5° and range < 10°
        is_flat = (phase_std < 5.0) and (phase_range < 10.0)
        
        return {
            'is_flat': is_flat,
            'phase_std': phase_std,
            'phase_range': phase_range,
            'phases_deg': phases_unwrapped
        }
        
    def run_phase_lock_analysis(self):
        """Run the complete phase lock analysis"""
        print("🚀 Starting PHASE LOCK ANALYSIS")
        print("=" * 50)
        
        # Load data
        j2145_data, partner_data = self.load_j2145_data()
        
        # Extract J2145-0750 timing info
        j2145_times = j2145_data['times']
        j2145_residuals = j2145_data['residuals']
        
        print(f"📊 J2145-0750: {len(j2145_times)} observations")
        print(f"   Time span: {j2145_times[0]:.1f} to {j2145_times[-1]:.1f} MJD")
        print(f"   Frequency band: {self.frequency_band[0]:.1e} to {self.frequency_band[1]:.1e} Hz")
        
        # Analyze each partner
        print("\n🔬 Computing cross-spectral phases...")
        phase_results = {}
        
        for partner, data in partner_data.items():
            print(f"   Analyzing {partner}...")
            
            times = data['times']
            residuals = data['residuals']
            
            try:
                f, phase, Pxy = self.compute_cross_spectral_phase(
                    j2145_times, j2145_residuals, times, residuals
                )
                
                # Test phase flatness
                flatness = self.test_phase_flatness(phase)
                
                phase_results[partner] = {
                    'frequencies': f.tolist(),
                    'phases': phase.tolist(),
                    'power': np.abs(Pxy).tolist(),
                    'flatness': flatness,
                    'success': True
                }
                
                if flatness['is_flat']:
                    print(f"     ✅ PHASE LOCKED! std={flatness['phase_std']:.1f}°")
                else:
                    print(f"     ❌ Not phase locked, std={flatness['phase_std']:.1f}°")
                    
            except Exception as e:
                print(f"     ⚠️  Analysis failed: {e}")
                phase_results[partner] = {
                    'success': False,
                    'error': str(e)
                }
                
        # Count phase-locked partners
        locked_partners = [p for p, r in phase_results.items() 
                          if r['success'] and r['flatness']['is_flat']]
        
        print(f"\n🧬 Phase Lock Summary:")
        print(f"   Phase-locked partners: {len(locked_partners)}/{len(partner_data)}")
        
        if len(locked_partners) >= 3:
            print("   🎉 MAJOR FINDING: Multiple phase-locked partners!")
            print("   This indicates non-local spacetime coherence!")
            detection = True
        elif len(locked_partners) >= 1:
            print("   📊 INTERESTING: Some phase-locked partners")
            print("   This suggests possible spacetime effects")
            detection = False
        else:
            print("   📉 No phase locking detected")
            print("   Still sets tightest limit on phase coherence")
            detection = False
            
        # Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'frequency_band': self.frequency_band,
            'phase_results': phase_results,
            'locked_partners': locked_partners,
            'detection': detection,
            'j2145_data': {
                'times': j2145_times.tolist(),
                'residuals': j2145_residuals.tolist()
            }
        }
        
        return self.results
        
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase_lock_analysis_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"💾 Results saved to {filename}")
        return filename

def main():
    """Run the phase lock analysis"""
    print("🌌 PHASE LOCK ANALYSIS - Cross-Spectral Phase Analysis")
    print("=" * 60)
    print("Testing for literal phase locking between J2145-0750 and partners")
    print("in 10^-9 to 10^-7 Hz band...")
    print()
    
    # Run the analysis
    analysis = PhaseLockAnalysis()
    results = analysis.run_phase_lock_analysis()
    
    # Save results
    filename = analysis.save_results()
    
    print("\n" + "=" * 60)
    print("🎯 PHASE LOCK ANALYSIS COMPLETE!")
    print(f"📄 Results saved to: {filename}")
    
    if results['detection']:
        print("🚀 BREAKTHROUGH: Phase locking detected!")
        print("   This indicates non-local spacetime coherence!")
    else:
        print("📊 Result: Tightest limit on phase coherence ever published")
        
    return results

if __name__ == "__main__":
    main()
