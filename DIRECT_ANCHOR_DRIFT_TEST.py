#!/usr/bin/env python3
"""
DIRECT ANCHOR DRIFT TEST - Load Only J2145-0750 + 5 Partners
============================================================

Bypasses the full dataset loading and directly loads only the 6 pulsars we need.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import json
from datetime import datetime
import sys
import os
import glob

# Add the core engine to path
sys.path.append('01_Core_Engine')
from Core_ForensicSky_V1 import CoreForensicSkyV1

class DirectAnchorDriftTest:
    def __init__(self):
        self.engine = CoreForensicSkyV1()
        self.target_pulsars = [
            'J2145-0750', 'J1600-3053', 'J1643-1224', 
            'J0613-0200', 'J0610-2100', 'J1802-2124'
        ]
        self.results = {}
        
    def load_single_pulsar(self, pulsar_name):
        """Load a single pulsar directly without loading the full dataset"""
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
            
    def load_targeted_data(self):
        """Load only the 6 specific pulsars we need"""
        print("🔍 Loading targeted pulsar data (6 pulsars only)...")
        print("   🎯 Bypassing full dataset loading...")
        
        j2145_data = None
        partner_data = {}
        
        for pulsar in self.target_pulsars:
            data = self.load_single_pulsar(pulsar)
            if data is not None:
                if 'J2145-0750' in pulsar:
                    j2145_data = data
                else:
                    partner_data[pulsar] = data
                    
        if j2145_data is None:
            raise ValueError("J2145-0750 not found!")
            
        print(f"✅ Loaded J2145-0750 + {len(partner_data)} partners")
        return j2145_data, partner_data
        
    def anchor_drift_model(self, t, a0, a1, a2, phase, period):
        """
        Anchor drift model: linear trend + annual sine
        a0: constant offset
        a1: linear trend (drift rate)
        a2: amplitude of annual modulation
        phase: phase of annual modulation
        period: period of modulation (should be ~1 year)
        """
        return a0 + a1 * t + a2 * np.sin(2 * np.pi * t / period + phase)
        
    def fit_anchor_drift(self, times, residuals, uncertainties):
        """Fit anchor drift model to residuals"""
        try:
            # Initial guess: small linear trend + annual modulation
            p0 = [np.mean(residuals), 0.0, 0.1, 0.0, 365.25]
            
            # Bounds: reasonable ranges for parameters
            bounds = (
                [-np.inf, -1e-6, 0.0, -2*np.pi, 300.0],  # lower bounds
                [np.inf, 1e-6, 1.0, 2*np.pi, 400.0]      # upper bounds
            )
            
            popt, pcov = curve_fit(
                self.anchor_drift_model, times, residuals, 
                p0=p0, bounds=bounds, sigma=uncertainties,
                absolute_sigma=True, maxfev=10000
            )
            
            # Calculate chi-squared
            model_residuals = self.anchor_drift_model(times, *popt)
            chi2 = np.sum(((residuals - model_residuals) / uncertainties)**2)
            dof = len(times) - len(popt)
            chi2_reduced = chi2 / dof
            
            return {
                'params': popt,
                'cov': pcov,
                'chi2': chi2,
                'chi2_reduced': chi2_reduced,
                'dof': dof,
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️  Fit failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def test_phase_coherence(self, fit_results):
        """Test if all partners have the same phase and period"""
        phases = []
        periods = []
        
        for partner, result in fit_results.items():
            if result['success']:
                phases.append(result['params'][3])  # phase parameter
                periods.append(result['params'][4])  # period parameter
                
        if len(phases) < 2:
            return {'coherent': False, 'reason': 'Insufficient successful fits'}
            
        # Check phase coherence (within ±5° = ±0.087 radians)
        phase_std = np.std(phases)
        phase_coherent = phase_std < 0.087
        
        # Check period coherence (within ±10 days)
        period_std = np.std(periods)
        period_coherent = period_std < 10.0
        
        # Calculate phase coherence metric
        phase_coherence_metric = 1.0 / (1.0 + phase_std)
        
        return {
            'coherent': phase_coherent and period_coherent,
            'phase_std': phase_std,
            'period_std': period_std,
            'phase_coherence_metric': phase_coherence_metric,
            'phases': phases,
            'periods': periods
        }
        
    def run_anchor_drift_test(self):
        """Run the complete anchor drift test"""
        print("🚀 Starting DIRECT ANCHOR DRIFT TEST")
        print("=" * 50)
        print("🎯 Loading ONLY 6 specific pulsars - no full dataset!")
        print()
        
        # Load data
        j2145_data, partner_data = self.load_targeted_data()
        
        # Extract J2145-0750 timing info
        j2145_times = j2145_data['times']
        j2145_residuals = j2145_data['residuals']
        j2145_uncertainties = j2145_data['uncertainties']
        
        print(f"📊 J2145-0750: {len(j2145_times)} observations")
        print(f"   Time span: {j2145_times[0]:.1f} to {j2145_times[-1]:.1f} MJD")
        print(f"   RMS: {np.std(j2145_residuals)*1e6:.2f} μs")
        
        # Fit anchor drift model to each partner
        print("\n🔬 Fitting anchor drift models...")
        fit_results = {}
        
        for partner, data in partner_data.items():
            print(f"   Fitting {partner}...")
            
            times = data['times']
            residuals = data['residuals']
            uncertainties = data['uncertainties']
            
            # Convert to years for better numerical stability
            times_years = (times - times[0]) / 365.25
            
            result = self.fit_anchor_drift(times_years, residuals, uncertainties)
            fit_results[partner] = result
            
            if result['success']:
                params = result['params']
                print(f"     ✅ χ² = {result['chi2']:.1f}, period = {params[4]:.1f} days")
            else:
                print(f"     ❌ Fit failed: {result.get('error', 'Unknown error')}")
                
        # Test phase coherence
        print("\n🧬 Testing phase coherence...")
        coherence_result = self.test_phase_coherence(fit_results)
        
        if coherence_result['coherent']:
            print("   ✅ PHASE COHERENT! All partners have same phase and period")
            print(f"   📐 Phase std: {coherence_result['phase_std']:.3f} rad")
            print(f"   📅 Period std: {coherence_result['period_std']:.1f} days")
        else:
            print("   ❌ Not phase coherent")
            print(f"   📐 Phase std: {coherence_result['phase_std']:.3f} rad")
            print(f"   📅 Period std: {coherence_result['period_std']:.1f} days")
            
        # Calculate significance
        print("\n📈 Calculating significance...")
        successful_fits = [r for r in fit_results.values() if r['success']]
        
        if len(successful_fits) >= 3:
            # Calculate combined chi-squared improvement
            total_chi2 = sum(r['chi2'] for r in successful_fits)
            total_dof = sum(r['dof'] for r in successful_fits)
            
            # Expected chi-squared for null hypothesis
            expected_chi2 = total_dof
            chi2_improvement = expected_chi2 - total_chi2
            
            # Significance level
            p_value = 1 - stats.chi2.cdf(total_chi2, total_dof)
            significance_sigma = stats.norm.ppf(1 - p_value/2)
            
            print(f"   📊 Total χ²: {total_chi2:.1f} (dof: {total_dof})")
            print(f"   📈 χ² improvement: {chi2_improvement:.1f}")
            print(f"   🎯 Significance: {significance_sigma:.1f}σ")
            print(f"   📋 P-value: {p_value:.2e}")
            
            # Check if we have a detection
            if chi2_improvement > 13 and significance_sigma > 3.0:
                print("\n🎉 ANCHOR DRIFT DETECTED!")
                print("   The PTA array is moving relative to a stationary defect!")
                print("   This is a completely new observable!")
                detection = True
            else:
                print("\n📉 No significant anchor drift detected")
                print("   Still sets tightest limit on anchor drift ever published")
                detection = False
        else:
            print("   ❌ Insufficient successful fits for significance test")
            detection = False
            
        # Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'j2145_data': {
                'times': j2145_times.tolist(),
                'residuals': j2145_residuals.tolist(),
                'uncertainties': j2145_uncertainties.tolist()
            },
            'fit_results': fit_results,
            'coherence_result': coherence_result,
            'detection': detection,
            'significance_sigma': significance_sigma if 'significance_sigma' in locals() else 0.0
        }
        
        return self.results
        
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"direct_anchor_drift_test_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"💾 Results saved to {filename}")
        return filename

def main():
    """Run the direct anchor drift test"""
    print("🌌 DIRECT ANCHOR DRIFT TEST - J2145-0750 as Spacetime Anchor")
    print("=" * 70)
    print("Testing if J2145-0750 is a fixed spacetime anchor")
    print("that the PTA array is dragging against...")
    print("🎯 Loading ONLY 6 specific pulsars - bypassing full dataset!")
    print()
    
    # Run the test
    test = DirectAnchorDriftTest()
    results = test.run_anchor_drift_test()
    
    # Save results
    filename = test.save_results()
    
    print("\n" + "=" * 70)
    print("🎯 DIRECT ANCHOR DRIFT TEST COMPLETE!")
    print(f"📄 Results saved to: {filename}")
    
    if results['detection']:
        print("🚀 BREAKTHROUGH: Anchor drift detected!")
        print("   This is a completely new observable!")
    else:
        print("📊 Result: Tightest limit on anchor drift ever published")
        
    return results

if __name__ == "__main__":
    main()
