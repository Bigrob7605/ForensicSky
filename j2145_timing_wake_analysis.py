#!/usr/bin/env python3
"""
J2145-0750 TIMING-RESIDUAL STRING WAKE ANALYSIS
==============================================

Fit simultaneous quadratic+linear drift with common crossing time
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

def load_timing_data():
    """Load timing data for the 6 correlated pulsars"""
    print("🔬 Loading timing data for string wake analysis...")
    
    try:
        # Import the Core Forensic Sky V1 engine
        sys.path.append('01_Core_Engine')
        from Core_ForensicSky_V1 import CoreForensicSkyV1
        
        # Initialize engine
        engine = CoreForensicSkyV1()
        
        # Load real IPTA DR2 data
        loading_stats = engine.load_real_ipta_data()
        
        if not hasattr(engine, 'timing_data') or not engine.timing_data:
            print("❌ No timing data loaded from engine!")
            return None
        
        # The 6 pulsars in our hub-and-spoke pattern
        target_pulsars = ['J2145-0750', 'J1600-3053', 'J1643-1224', 'J0613-0200', 'J0610-2100', 'J1802-2124']
        
        timing_data = {}
        
        for pulsar_id in target_pulsars:
            if pulsar_id in engine.timing_data:
                timing_info = engine.timing_data[pulsar_id]
                if 'residuals' in timing_info and len(timing_info['residuals']) > 0:
                    # Get times and residuals
                    times = timing_info.get('times', [])
                    residuals = timing_info['residuals']
                    
                    if len(times) > 0 and len(residuals) > 0:
                        # Ensure same length
                        min_len = min(len(times), len(residuals))
                        timing_data[pulsar_id] = {
                            'times': np.array(times[:min_len]),
                            'residuals': np.array(residuals[:min_len])
                        }
                        print(f"✅ {pulsar_id}: {min_len} data points")
                    else:
                        print(f"⚠️ {pulsar_id}: No timing data")
                else:
                    print(f"⚠️ {pulsar_id}: No residuals")
            else:
                print(f"⚠️ {pulsar_id}: Not found in timing data")
        
        print(f"📊 Loaded timing data for {len(timing_data)} pulsars")
        return timing_data
        
    except Exception as e:
        print(f"❌ Error loading timing data: {e}")
        return None

def string_wake_template(t, t0, a, b, c):
    """String wake template: r(t) = a*(t-t0)^2 + b*(t-t0) + c"""
    return a * (t - t0)**2 + b * (t - t0) + c

def fit_individual_pulsars(timing_data):
    """Fit each pulsar individually (no common t0)"""
    print("\n🔍 FITTING INDIVIDUAL PULSARS (NO COMMON T0)")
    print("-" * 60)
    
    individual_fits = {}
    total_chi2_individual = 0
    total_dof = 0
    
    for pulsar_id, data in timing_data.items():
        times = data['times']
        residuals = data['residuals']
        
        if len(times) < 3:
            print(f"⚠️ {pulsar_id}: Not enough data points")
            continue
        
        # Fit individual quadratic+linear drift
        def chi2_func(params):
            t0, a, b, c = params
            model = string_wake_template(times, t0, a, b, c)
            return np.sum((residuals - model)**2)
        
        # Initial guess
        t0_init = np.mean(times)
        a_init = 0.0
        b_init = 0.0
        c_init = np.mean(residuals)
        
        result = minimize(chi2_func, [t0_init, a_init, b_init, c_init], method='Nelder-Mead')
        
        if result.success:
            t0, a, b, c = result.x
            chi2_val = result.fun
            dof = len(times) - 4  # 4 parameters
            
            individual_fits[pulsar_id] = {
                't0': t0,
                'a': a,
                'b': b,
                'c': c,
                'chi2': chi2_val,
                'dof': dof
            }
            
            total_chi2_individual += chi2_val
            total_dof += dof
            
            print(f"{pulsar_id}: t0={t0:.2f}, a={a:.2e}, b={b:.2e}, c={c:.2e}, χ²={chi2_val:.2f}")
        else:
            print(f"❌ {pulsar_id}: Fit failed")
    
    print(f"\nTotal χ² (individual): {total_chi2_individual:.2f}")
    print(f"Total DOF: {total_dof}")
    
    return individual_fits, total_chi2_individual, total_dof

def fit_common_t0(timing_data):
    """Fit all pulsars with common crossing time t0"""
    print("\n🔍 FITTING WITH COMMON T0 (STRING WAKE)")
    print("-" * 60)
    
    # Prepare data
    all_times = []
    all_residuals = []
    pulsar_indices = []
    
    for i, (pulsar_id, data) in enumerate(timing_data.items()):
        times = data['times']
        residuals = data['residuals']
        
        if len(times) < 3:
            continue
        
        all_times.extend(times)
        all_residuals.extend(residuals)
        pulsar_indices.extend([i] * len(times))
    
    all_times = np.array(all_times)
    all_residuals = np.array(all_residuals)
    pulsar_indices = np.array(pulsar_indices)
    
    n_pulsars = len(timing_data)
    
    def chi2_func(params):
        t0 = params[0]  # Common crossing time
        a_params = params[1:n_pulsars+1]  # Quadratic coefficients
        b_params = params[n_pulsars+1:2*n_pulsars+1]  # Linear coefficients
        c_params = params[2*n_pulsars+1:3*n_pulsars+1]  # Constant offsets
        
        chi2_total = 0
        for i, (pulsar_id, data) in enumerate(timing_data.items()):
            if i >= n_pulsars:
                break
                
            times = data['times']
            residuals = data['residuals']
            
            if len(times) < 3:
                continue
            
            model = string_wake_template(times, t0, a_params[i], b_params[i], c_params[i])
            chi2_total += np.sum((residuals - model)**2)
        
        return chi2_total
    
    # Initial guess
    t0_init = np.mean(all_times)
    a_init = np.zeros(n_pulsars)
    b_init = np.zeros(n_pulsars)
    c_init = np.array([np.mean(timing_data[list(timing_data.keys())[i]]['residuals']) for i in range(n_pulsars)])
    
    initial_params = np.concatenate([[t0_init], a_init, b_init, c_init])
    
    result = minimize(chi2_func, initial_params, method='Nelder-Mead')
    
    if result.success:
        t0 = result.x[0]
        a_params = result.x[1:n_pulsars+1]
        b_params = result.x[n_pulsars+1:2*n_pulsars+1]
        c_params = result.x[2*n_pulsars+1:3*n_pulsars+1]
        
        chi2_common = result.fun
        dof_common = len(all_times) - (1 + 3*n_pulsars)  # 1 common t0 + 3 params per pulsar
        
        print(f"Common t0: {t0:.2f}")
        print(f"Total χ² (common t0): {chi2_common:.2f}")
        print(f"Total DOF: {dof_common}")
        
        # Individual parameters
        for i, pulsar_id in enumerate(timing_data.keys()):
            if i < n_pulsars:
                print(f"{pulsar_id}: a={a_params[i]:.2e}, b={b_params[i]:.2e}, c={c_params[i]:.2e}")
        
        return t0, a_params, b_params, c_params, chi2_common, dof_common
    else:
        print("❌ Common t0 fit failed")
        return None, None, None, None, None, None

def main():
    """Run the timing-residual string wake analysis"""
    print("🚀 J2145-0750 TIMING-RESIDUAL STRING WAKE ANALYSIS")
    print("=" * 70)
    print("Fit simultaneous quadratic+linear drift with common crossing time")
    print()
    
    # Load timing data
    timing_data = load_timing_data()
    
    if timing_data is None or len(timing_data) == 0:
        print("❌ Cannot proceed without timing data")
        return
    
    # Fit individual pulsars
    individual_fits, chi2_individual, dof_individual = fit_individual_pulsars(timing_data)
    
    # Fit with common t0
    t0, a_params, b_params, c_params, chi2_common, dof_common = fit_common_t0(timing_data)
    
    if t0 is not None:
        # Compare fits
        print(f"\n🎯 STRING WAKE SIGNIFICANCE TEST")
        print("-" * 50)
        
        delta_chi2 = chi2_individual - chi2_common
        delta_dof = dof_individual - dof_common
        
        print(f"Δχ² = {delta_chi2:.2f}")
        print(f"ΔDOF = {delta_dof}")
        
        if delta_dof > 0:
            # Calculate p-value
            p_value = 1 - chi2.cdf(delta_chi2, delta_dof)
            print(f"P-value: {p_value:.2e}")
            
            if delta_chi2 > 9:
                print("✅ STRING WAKE DETECTED! χ² improvement > 9")
                print("   - Common crossing time significantly improves fit")
                print("   - Consistent with string wake lensing")
            elif delta_chi2 > 4:
                print("⚠️  MODERATE STRING WAKE EVIDENCE (χ² improvement > 4)")
            else:
                print("❌ NO STRING WAKE EVIDENCE (χ² improvement < 4)")
        else:
            print("⚠️  Cannot calculate significance (ΔDOF ≤ 0)")
        
        # Save results
        results = {
            'common_t0': float(t0),
            'chi2_individual': float(chi2_individual),
            'chi2_common': float(chi2_common),
            'delta_chi2': float(delta_chi2),
            'delta_dof': int(delta_dof),
            'p_value': float(p_value) if delta_dof > 0 else None,
            'individual_fits': individual_fits,
            'common_fit_params': {
                'a_params': a_params.tolist(),
                'b_params': b_params.tolist(),
                'c_params': c_params.tolist()
            }
        }
        
        with open('j2145_string_wake_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved to 'j2145_string_wake_results.json'")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
