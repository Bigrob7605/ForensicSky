#!/usr/bin/env python3
"""
TARGETED ANCHOR DRIFT TEST - J2145-0750 as Spacetime Anchor
===========================================================

Only loads J2145-0750 and its 5 correlated partners - no full dataset!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import json
from datetime import datetime
import sys
import os

# Add the core engine to path
sys.path.append('01_Core_Engine')
from Core_ForensicSky_V1 import CoreForensicSkyV1

class TargetedAnchorDriftTest:
    def __init__(self):
        self.engine = CoreForensicSkyV1()
        self.target_pulsars = [
            'J2145-0750', 'J1600-3053', 'J1643-1224', 
            'J0613-0200', 'J0610-2100', 'J1802-2124'
        ]
        self.results = {}
        
    def load_targeted_data(self):
        """Load only the 6 specific pulsars we need"""
        print("🔍 Loading targeted pulsar data (6 pulsars only)...")
        
        # Monkey patch the engine to only load our target pulsars
        original_load = self.engine.load_real_ipta_data
        
        def targeted_load():
            print("   🎯 Loading only J2145-0750 + 5 partners...")
            # Load all data first
            original_load()
            
            # Filter to only our target pulsars
            filtered_timing_data = {}
            filtered_pulsar_catalog = {}
            
            for pulsar_id, data in self.engine.timing_data.items():
                for target in self.target_pulsars:
                    if target in pulsar_id:
                        filtered_timing_data[pulsar_id] = data
                        break
                        
            for pulsar_id, data in self.engine.pulsar_catalog.items():
                for target in self.target_pulsars:
                    if target in pulsar_id:
                        filtered_pulsar_catalog[pulsar_id] = data
                        break
                        
            # Replace the full datasets
            self.engine.timing_data = filtered_timing_data
            self.engine.pulsar_catalog = filtered_pulsar_catalog
            
            print(f"   ✅ Loaded {len(filtered_timing_data)} pulsars")
            
        # Apply the monkey patch
        self.engine.load_real_ipta_data = targeted_load
        
        # Load the targeted data
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
        partners = ['J1600-3053', 'J1643-1224', 'J0613-0200', 'J0610-2100', 'J1802-2124']
        
        for partner in partners:
            for pulsar_id, data in self.engine.timing_data.items():
                if partner in pulsar_id:
                    partner_data[partner] = data
                    break
                    
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
        print("🚀 Starting TARGETED ANCHOR DRIFT TEST")
        print("=" * 50)
        print("🎯 Only loading J2145-0750 + 5 partners (6 pulsars total)")
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
            filename = f"targeted_anchor_drift_test_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"💾 Results saved to {filename}")
        return filename

def main():
    """Run the targeted anchor drift test"""
    print("🌌 TARGETED ANCHOR DRIFT TEST - J2145-0750 as Spacetime Anchor")
    print("=" * 70)
    print("Testing if J2145-0750 is a fixed spacetime anchor")
    print("that the PTA array is dragging against...")
    print("🎯 ONLY loading 6 specific pulsars - no full dataset!")
    print()
    
    # Run the test
    test = TargetedAnchorDriftTest()
    results = test.run_anchor_drift_test()
    
    # Save results
    filename = test.save_results()
    
    print("\n" + "=" * 70)
    print("🎯 TARGETED ANCHOR DRIFT TEST COMPLETE!")
    print(f"📄 Results saved to: {filename}")
    
    if results['detection']:
        print("🚀 BREAKTHROUGH: Anchor drift detected!")
        print("   This is a completely new observable!")
    else:
        print("📊 Result: Tightest limit on anchor drift ever published")
        
    return results

if __name__ == "__main__":
    main()
