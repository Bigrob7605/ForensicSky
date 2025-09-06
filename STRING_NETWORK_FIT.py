#!/usr/bin/env python3
"""
STRING NETWORK FIT - Global Quantum Coherence Analysis
====================================================

Fit global string network coherence across >15° baselines
using quantum-verified correlations from J2145-0750 hub
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Try to import emcee for MCMC, fall back to simple optimization
try:
    import emcee
    MCMC_AVAILABLE = True
    print("✅ Emcee available - MCMC sampling ready!")
except ImportError:
    MCMC_AVAILABLE = False
    print("⚠️ Emcee not available - using simple optimization")

def load_quantum_verified_data():
    """Load the quantum-verified correlation data"""
    print("🔬 Loading quantum-verified correlation data...")
    
    # Load quantum results
    with open('quantum_50_premium_pulsars_20250905_193411.json', 'r') as f:
        quantum_data = json.load(f)
    
    # The 6 pulsars in our hub-and-spoke pattern
    psr_names = ['J2145-0750', 'J1600-3053', 'J1643-1224', 'J0613-0200', 'J0610-2100', 'J1802-2124']
    
    # Get their indices in the quantum data
    all_pulsars = quantum_data['pulsar_ids']
    psr_indices = [all_pulsars.index(name) for name in psr_names if name in all_pulsars]
    
    # Extract quantum kernel values for these pulsars
    kernels = np.array(quantum_data['kernels'])
    quantum_kernels = kernels[np.ix_(psr_indices, psr_indices)]
    
    print(f"✅ Loaded quantum kernels for {len(psr_names)} pulsars")
    print(f"Quantum kernel matrix shape: {quantum_kernels.shape}")
    
    return psr_names, quantum_kernels

def string_wake_template(toas, t_k, A, w=30):
    """String wake template: |t - t_k|^(4/3) with width w"""
    dt = (toas - t_k) / w
    mask = np.abs(dt) < 1
    tmpl = np.zeros_like(toas)
    tmpl[mask] = A * np.abs(dt[mask])**(4/3)
    return tmpl

def angular_separation(ra1, dec1, ra2, dec2):
    """Calculate angular separation between two points on the sky"""
    # Convert to radians
    ra1, dec1, ra2, dec2 = np.radians([ra1, dec1, ra2, dec2])
    
    # Angular separation formula
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    cos_sep = np.clip(cos_sep, -1, 1)  # Avoid numerical errors
    return np.degrees(np.arccos(cos_sep))

def log_likelihood_string_network(theta, resids, toas, psr_positions, white_var=1e-12):
    """Log likelihood for string network model"""
    # theta = [log_Gμ, t1, t2, t3, ra1, dec1, ra2, dec2, ra3, dec3, log_theta_kern]
    log_Gμ = theta[0]
    Gμ = 10**log_Gμ
    
    t_k = theta[1:4]  # 3 string crossing times
    ra_k = theta[4:7]  # 3 string node RA
    dec_k = theta[7:10]  # 3 string node DEC
    log_theta_kern = theta[10]
    theta_kern = 10**log_theta_kern
    
    # String node positions
    pos_k = np.array([ra_k, dec_k]).T  # 3 string nodes
    
    ll = 0
    for i, (r, t, pos_p) in enumerate(zip(resids, toas, psr_positions)):
        # Calculate angular separations from pulsar to all string nodes
        dtheta = np.array([angular_separation(pos_p[0], pos_p[1], ra, dec) for ra, dec in zip(ra_k, dec_k)])
        
        # Kernel weights
        kern = np.exp(-(dtheta / theta_kern)**2)
        
        # String wake model
        model = np.zeros_like(t)
        for tk, k in zip(t_k, kern):
            model += string_wake_template(t, tk, Gμ * k)
        
        # Log likelihood
        ll += -0.5 * np.sum((r - model)**2 / white_var)
    
    return ll

def log_likelihood_null(theta, resids, toas, white_var=1e-12):
    """Log likelihood for null model (independent red noise)"""
    # theta = [log_red_var] for each pulsar
    n_pulsars = len(resids)
    log_red_var = theta[:n_pulsars]
    red_var = 10**log_red_var
    
    ll = 0
    for i, (r, t) in enumerate(zip(resids, toas)):
        # Simple red noise model
        ll += -0.5 * np.sum(r**2 / (red_var[i] + white_var))
    
    return ll

def fit_string_network_simple(resids, toas, psr_positions):
    """Simple optimization-based string network fit"""
    print("\n🔍 FITTING STRING NETWORK MODEL (SIMPLE OPTIMIZATION)")
    print("-" * 70)
    
    n_pulsars = len(resids)
    
    # Initial guess
    log_Gμ_init = -11  # GUT scale
    t_k_init = [np.mean(toas[0]), np.mean(toas[1]), np.mean(toas[2])]  # 3 crossing times
    ra_k_init = [psr_positions[0][0], psr_positions[1][0], psr_positions[2][0]]  # 3 RA
    dec_k_init = [psr_positions[0][1], psr_positions[1][1], psr_positions[2][1]]  # 3 DEC
    log_theta_kern_init = 1.0  # 10 degrees
    
    initial_params = np.concatenate([[log_Gμ_init], t_k_init, ra_k_init, dec_k_init, [log_theta_kern_init]])
    
    def neg_log_likelihood(theta):
        return -log_likelihood_string_network(theta, resids, toas, psr_positions)
    
    result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead')
    
    if result.success:
        theta_opt = result.x
        log_Gμ = theta_opt[0]
        t_k = theta_opt[1:4]
        ra_k = theta_opt[4:7]
        dec_k = theta_opt[7:10]
        log_theta_kern = theta_opt[10]
        
        Gμ = 10**log_Gμ
        theta_kern = 10**log_theta_kern
        
        print(f"✅ String network fit converged!")
        print(f"Gμ = {Gμ:.2e}")
        print(f"String crossing times: {t_k}")
        print(f"String node positions: RA={ra_k}, DEC={dec_k}")
        print(f"Kernel width: {theta_kern:.1f}°")
        
        return theta_opt, result.fun
    else:
        print("❌ String network fit failed to converge")
        return None, None

def fit_null_model(resids, toas):
    """Fit null model (independent red noise)"""
    print("\n🔍 FITTING NULL MODEL (INDEPENDENT RED NOISE)")
    print("-" * 60)
    
    n_pulsars = len(resids)
    
    # Initial guess
    log_red_var_init = np.full(n_pulsars, -12)  # 1e-12 variance
    
    def neg_log_likelihood(theta):
        return -log_likelihood_null(theta, resids, toas)
    
    result = minimize(neg_log_likelihood, log_red_var_init, method='Nelder-Mead')
    
    if result.success:
        log_red_var = result.x
        red_var = 10**log_red_var
        
        print(f"✅ Null model fit converged!")
        print(f"Red noise variances: {red_var}")
        
        return result.x, result.fun
    else:
        print("❌ Null model fit failed to converge")
        return None, None

def create_synthetic_data():
    """Create synthetic data for testing"""
    print("🧪 Creating synthetic data for testing...")
    
    # 6 pulsars with known positions (approximate)
    psr_positions = [
        (326.8, -7.8),   # J2145-0750
        (240.0, -30.9),  # J1600-3053
        (251.0, -12.4),  # J1643-1224
        (93.3, -2.0),    # J0613-0200
        (92.5, -21.0),   # J0610-2100
        (270.5, -21.4)   # J1802-2124
    ]
    
    # Generate synthetic timing data
    n_points = 100
    mjd_start = 50000
    mjd_end = 60000
    
    resids = []
    toas = []
    
    for i, (ra, dec) in enumerate(psr_positions):
        # Generate TOAs
        t = np.linspace(mjd_start, mjd_end, n_points)
        
        # Generate residuals with some structure
        # Add a string wake signature
        t_string = 55000  # String crossing time
        A_string = 1e-6  # String amplitude
        w_string = 30    # String width
        
        r = np.random.normal(0, 1e-6, n_points)
        
        # Add string wake
        dt = (t - t_string) / w_string
        mask = np.abs(dt) < 1
        r[mask] += A_string * np.abs(dt[mask])**(4/3)
        
        resids.append(r)
        toas.append(t)
    
    print(f"✅ Generated synthetic data for {len(psr_positions)} pulsars")
    return resids, toas, psr_positions

def main():
    """Run the string network fit analysis"""
    print("🚀 STRING NETWORK FIT - GLOBAL QUANTUM COHERENCE")
    print("=" * 70)
    print("Fit global string network coherence across >15° baselines")
    print()
    
    # For now, use synthetic data to test the framework
    # In production, this would load real timing data
    resids, toas, psr_positions = create_synthetic_data()
    
    print(f"📊 Data summary:")
    print(f"  Pulsars: {len(resids)}")
    print(f"  Data points per pulsar: {len(resids[0])}")
    print(f"  Time range: {toas[0][0]:.0f} - {toas[0][-1]:.0f} MJD")
    
    # Fit null model
    null_params, null_ll = fit_null_model(resids, toas)
    
    # Fit string network model
    string_params, string_ll = fit_string_network_simple(resids, toas, psr_positions)
    
    if null_params is not None and string_params is not None:
        # Compare models
        print(f"\n🎯 MODEL COMPARISON")
        print("-" * 40)
        
        delta_ll = string_ll - null_ll  # Note: this is negative log likelihood
        delta_ll = -delta_ll  # Convert to positive for string network
        
        print(f"Null model log-likelihood: {null_ll:.2f}")
        print(f"String network log-likelihood: {string_ll:.2f}")
        print(f"Δlog-likelihood: {delta_ll:.2f}")
        
        if delta_ll > 10:
            print("✅ STRING NETWORK STRONGLY FAVORED! ΔlogL > 10")
        elif delta_ll > 5:
            print("⚠️  STRING NETWORK MODERATELY FAVORED (ΔlogL > 5)")
        else:
            print("❌ NO STRING NETWORK EVIDENCE (ΔlogL < 5)")
        
        # Save results
        results = {
            'null_model': {
                'params': null_params.tolist(),
                'log_likelihood': float(null_ll)
            },
            'string_network': {
                'params': string_params.tolist(),
                'log_likelihood': float(string_ll)
            },
            'delta_log_likelihood': float(delta_ll),
            'psr_positions': psr_positions
        }
        
        with open('string_network_fit_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved to 'string_network_fit_results.json'")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
