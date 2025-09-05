#!/usr/bin/env python3
"""
SENSITIVITY CURVE TEST
=====================
Test detection fraction vs. Gμ to build sensitivity curve
"""

import json
import numpy as np
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def test_sensitivity_curve():
    """Test detection fraction vs. Gμ"""
    
    log.info("🎯 SENSITIVITY CURVE TEST - Detection Fraction vs. Gμ")
    log.info("=" * 60)
    
    # Test different Gμ values
    Gmu_values = [1e-12, 5e-12, 1e-11, 5e-11, 1e-10]
    detection_fractions = []
    
    for Gmu in Gmu_values:
        log.info(f"\n🌌 Testing Gμ = {Gmu:.2e}")
        
        # Create injection with this Gμ
        injection_file = f"test_injection_Gmu_{Gmu:.0e}.json"
        
        # Use our injection script
        cmd = f"python inject_cosmic_string_skies.py REAL_ENHANCED_COSMIC_STRING_RESULTS.json --Gmu {Gmu} -o {injection_file}"
        subprocess.run(cmd, shell=True, capture_output=True)
        
        # Run forensic disproof
        cmd = f"python disprove_cosmic_strings_forensic.py {injection_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        verdict = result.stdout.strip()
        log.info(f"   Verdict: {verdict}")
        
        # Calculate detection fraction
        if verdict == "STRONG":
            detection_fraction = 1.0
        elif verdict == "WEAK":
            detection_fraction = 0.5
        else:  # TOY_DATA
            detection_fraction = 0.0
            
        detection_fractions.append(detection_fraction)
        log.info(f"   Detection fraction: {detection_fraction:.1%}")
    
    # Create sensitivity curve data
    sensitivity_data = {
        'Gmu_values': Gmu_values,
        'detection_fractions': detection_fractions,
        'description': 'Cosmic string detection sensitivity curve'
    }
    
    # Save results
    with open('SENSITIVITY_CURVE_DATA.json', 'w') as f:
        json.dump(sensitivity_data, f, indent=2)
    
    log.info(f"\n📊 SENSITIVITY CURVE RESULTS:")
    for i, (Gmu, frac) in enumerate(zip(Gmu_values, detection_fractions)):
        log.info(f"   Gμ = {Gmu:.2e}: {frac:.1%} detection")
    
    # Find 90% detection threshold
    try:
        idx_90 = next(i for i, frac in enumerate(detection_fractions) if frac >= 0.9)
        threshold_Gmu = Gmu_values[idx_90]
        log.info(f"\n🎯 90% Detection Threshold: Gμ = {threshold_Gmu:.2e}")
    except StopIteration:
        log.info(f"\n⚠️  90% detection threshold not reached in tested range")
    
    log.info(f"\n📁 Sensitivity curve data saved to SENSITIVITY_CURVE_DATA.json")
    
    return sensitivity_data

if __name__ == "__main__":
    test_sensitivity_curve()
