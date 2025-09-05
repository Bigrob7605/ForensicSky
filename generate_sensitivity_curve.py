#!/usr/bin/env python3
"""
SENSITIVITY CURVE GENERATOR
==========================
Generate sensitivity curve with error bars for tomorrow's sprint
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def generate_sensitivity_curve():
    """Generate sensitivity curve with error bars"""
    
    log.info("🎯 GENERATING SENSITIVITY CURVE")
    log.info("=" * 50)
    
    # Gμ values to test (5e-12 to 2e-11 in 5 steps)
    Gmu_values = np.logspace(np.log10(5e-12), np.log10(2e-11), 5)
    n_monte_carlo = 100  # 100 skies per point
    
    log.info(f"📊 Testing {len(Gmu_values)} Gμ values")
    log.info(f"🎲 {n_monte_carlo} Monte-Carlo realizations per point")
    
    # Store results
    all_results = []
    
    for i, Gmu in enumerate(Gmu_values):
        log.info(f"\n🌌 Testing Gμ = {Gmu:.2e} ({i+1}/{len(Gmu_values)})")
        
        detection_fractions = []
        
        for j in range(n_monte_carlo):
            # Create injection with this Gμ and seed
            injection_file = f"temp_injection_Gmu_{Gmu:.0e}_seed_{j}.json"
            
            # Use enhanced injection script
            cmd = f"python enhanced_inject_cosmic_string_skies.py REAL_ENHANCED_COSMIC_STRING_RESULTS.json --Gmu {Gmu} --seed {j} -o {injection_file}"
            subprocess.run(cmd, shell=True, capture_output=True)
            
            # Run forensic disproof
            cmd = f"python disprove_cosmic_strings_forensic.py {injection_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            verdict = result.stdout.strip()
            
            # Calculate detection fraction
            if verdict == "STRONG":
                detection_fraction = 1.0
            elif verdict == "WEAK":
                detection_fraction = 0.5
            else:  # TOY_DATA
                detection_fraction = 0.0
                
            detection_fractions.append(detection_fraction)
            
            # Clean up temp file
            Path(injection_file).unlink(missing_ok=True)
        
        # Calculate statistics
        mean_detection = np.mean(detection_fractions)
        std_detection = np.std(detection_fractions)
        
        all_results.append({
            'Gmu': Gmu,
            'detection_fractions': detection_fractions,
            'mean_detection': mean_detection,
            'std_detection': std_detection
        })
        
        log.info(f"   Mean detection: {mean_detection:.1%} ± {std_detection:.1%}")
    
    # Create sensitivity curve plot
    log.info("\n📊 Creating sensitivity curve plot...")
    
    Gmu_vals = [r['Gmu'] for r in all_results]
    mean_detections = [r['mean_detection'] for r in all_results]
    std_detections = [r['std_detection'] for r in all_results]
    
    plt.figure(figsize=(10, 8))
    
    # Plot with error bars
    plt.errorbar(Gmu_vals, mean_detections, yerr=std_detections, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    # Add 90% detection threshold line
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Detection Threshold')
    
    # Formatting
    plt.xscale('log')
    plt.xlabel('Cosmic String Tension Gμ', fontsize=14, fontweight='bold')
    plt.ylabel('Detection Fraction', fontsize=14, fontweight='bold')
    plt.title('Cosmic String Detection Sensitivity Curve\n' + 
              f'Monte-Carlo Analysis ({n_monte_carlo} realizations per point)', 
              fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('SENSITIVITY_CURVE_v0.3.0.png', dpi=300, bbox_inches='tight')
    plt.savefig('SENSITIVITY_CURVE_v0.3.0.pdf', bbox_inches='tight')
    
    # Save CSV data
    csv_data = []
    for r in all_results:
        csv_data.append({
            'Gmu': r['Gmu'],
            'mean_detection': r['mean_detection'],
            'std_detection': r['std_detection']
        })
    
    with open('SENSITIVITY_CURVE_v0.3.0.csv', 'w') as f:
        f.write('Gmu,mean_detection,std_detection\n')
        for row in csv_data:
            f.write(f"{row['Gmu']:.2e},{row['mean_detection']:.4f},{row['std_detection']:.4f}\n")
    
    # Save JSON results
    with open('SENSITIVITY_CURVE_v0.3.0.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    log.info("✅ Sensitivity curve generated!")
    log.info("📁 Files saved:")
    log.info("   - SENSITIVITY_CURVE_v0.3.0.png")
    log.info("   - SENSITIVITY_CURVE_v0.3.0.pdf")
    log.info("   - SENSITIVITY_CURVE_v0.3.0.csv")
    log.info("   - SENSITIVITY_CURVE_v0.3.0.json")
    
    return all_results

if __name__ == "__main__":
    generate_sensitivity_curve()
