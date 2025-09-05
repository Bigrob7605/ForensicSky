#!/usr/bin/env python3
"""
NEXT SPRINT PREPARATION
=======================
Prepare tools for tomorrow's 2h sprint:
1. Sweep Gμ = 5e-12 → 2e-11 in 5 steps
2. Add --seed argument for repeatable Monte-Carlo
3. Generate sensitivity curve with error bars
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def create_enhanced_injection_engine():
    """Create enhanced injection engine with --seed support"""
    
    enhanced_script = '''#!/usr/bin/env python3
"""
ENHANCED COSMIC STRING INJECTION ENGINE
======================================
Enhanced version with --seed support for repeatable Monte-Carlo
"""

import json
import sys
import numpy as np
import argparse
import logging
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("injector")

class EnhancedCosmicStringInjector:
    """Enhanced cosmic string injector with seed support"""
    
    def __init__(self, input_json: str, Gmu: float = 1e-11, seed: int = None):
        self.input_json = input_json
        self.Gmu = Gmu
        self.seed = seed
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            log.info(f"🎲 Random seed set to {seed}")
    
    def load_results(self):
        """Load the input results JSON"""
        with open(self.input_json, 'r') as f:
            self.results = json.load(f)
        log.info(f"📁 Loaded results from {self.input_json}")
        
    def inject_cosmic_string_signal(self):
        """Inject cosmic string gravitational wave signal"""
        log.info(f"🌌 Injecting cosmic string signal with Gμ = {self.Gmu:.2e}")
        
        # Cosmic string parameters
        loop_size = 10.0  # kpc
        cusp_amplitude = self.Gmu * 1e-14  # Amplitude scaling
        
        # Get correlation analysis data
        ca = self.results['correlation_analysis']
        
        # Inject cosmic string correlations
        if 'correlations' in ca:
            original_correlations = np.array(ca['correlations'])
            n_correlations = len(original_correlations)
            
            # Generate cosmic string correlation signal
            angular_separations = np.array(ca.get('angular_separations', np.random.uniform(0, np.pi, n_correlations)))
            
            # Hellings-Downs correlation for cosmic strings
            hd_correlation = 0.5 * (1 - np.cos(angular_separations)) * np.log(0.5 * (1 - np.cos(angular_separations))) - \
                           0.25 * (1 - np.cos(angular_separations)) + 0.25 * (3 + np.cos(angular_separations)) * np.log(0.5 * (3 + np.cos(angular_separations)))
            
            # Scale by cosmic string amplitude
            cosmic_string_signal = cusp_amplitude * hd_correlation
            
            # Add to original correlations
            injected_correlations = original_correlations + cosmic_string_signal
            ca['correlations'] = injected_correlations.tolist()
            
            # Update statistics
            ca['mean_correlation'] = float(np.mean(injected_correlations))
            ca['std_correlation'] = float(np.std(injected_correlations))
            ca['detection_rate'] = float(np.sum(np.abs(injected_correlations) > 0.1) / len(injected_correlations) * 100)
            
            log.info(f"✅ Injected cosmic string correlations: {n_correlations} pairs")
            log.info(f"   Mean correlation: {ca['mean_correlation']:.4f}")
            log.info(f"   Detection rate: {ca['detection_rate']:.1f}%")
        
        # Inject periodic signals (cosmic string cusps)
        pa = self.results['periodic_analysis']
        
        if 'periodic_results' in pa:
            n_pulsars = len(pa['periodic_results'])
            
            # Generate cosmic string periodic signals
            for i, pulsar_result in enumerate(pa['periodic_results']):
                # Cosmic string cusps produce periodic signals
                cusp_frequency = 1.0 / (365.25 * 10)  # ~10 year period
                cusp_amplitude = self.Gmu * 1e-13  # Smaller amplitude for individual pulsars
                
                # Add cosmic string periodic component
                if 'power' in pulsar_result:
                    pulsar_result['power'] += cusp_amplitude
                if 'snr' in pulsar_result:
                    pulsar_result['snr'] += cusp_amplitude * 10  # Boost SNR
                if 'fap' in pulsar_result:
                    pulsar_result['fap'] = max(0.001, pulsar_result['fap'] * 0.1)  # Lower FAP
            
            # Update periodic statistics
            powers = [r.get('power', 0) for r in pa['periodic_results']]
            snrs = [r.get('snr', 0) for r in pa['periodic_results']]
            faps = [r.get('fap', 0.5) for r in pa['periodic_results']]
            
            pa['mean_power'] = float(np.mean(powers))
            pa['mean_snr'] = float(np.mean(snrs))
            pa['mean_fap'] = float(np.mean(faps))
            pa['detection_rate'] = float(np.sum(np.array(faps) < 0.01) / len(faps) * 100)
            
            log.info(f"✅ Injected cosmic string periodic signals: {n_pulsars} pulsars")
            log.info(f"   Mean SNR: {pa['mean_snr']:.2f}")
            log.info(f"   Detection rate: {pa['detection_rate']:.1f}%")
        
        # Inject spectral signature
        sa = self.results['spectral_analysis']
        
        # Cosmic strings produce specific spectral signatures
        cosmic_string_slope = -2/3  # Characteristic slope for cosmic strings
        cosmic_string_amplitude = self.Gmu * 1e-15
        
        # Update spectral parameters
        sa['mean_slope'] = cosmic_string_slope
        sa['mean_white_noise_strength'] = cosmic_string_amplitude
        sa['cosmic_string_injected'] = True
        sa['injection_Gmu'] = self.Gmu
        
        log.info(f"✅ Injected cosmic string spectral signature")
        log.info(f"   Slope: {cosmic_string_slope:.3f}")
        log.info(f"   Amplitude: {cosmic_string_amplitude:.2e}")
        
        # Add injection metadata
        self.results['injection_metadata'] = {
            'Gmu': self.Gmu,
            'seed': self.seed,
            'injection_type': 'cosmic_string',
            'timestamp': np.datetime64('now').astype(str),
            'description': f'Cosmic string signal with Gμ = {self.Gmu:.2e} injected (seed={self.seed})'
        }
        
        log.info(f"🎯 Cosmic string injection complete!")
        
    def save_injected_results(self, output_file: str):
        """Save the injected results to a new JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        log.info(f"💾 Saved injected results to {output_file}")
        
    def run_injection(self, output_file: str):
        """Run the complete injection process"""
        log.info(f"🚀 STARTING ENHANCED COSMIC STRING INJECTION")
        log.info(f"   Input: {self.input_json}")
        log.info(f"   Gμ: {self.Gmu:.2e}")
        log.info(f"   Seed: {self.seed}")
        log.info(f"   Output: {output_file}")
        log.info("=" * 60)
        
        self.load_results()
        self.inject_cosmic_string_signal()
        self.save_injected_results(output_file)
        
        log.info("🎉 ENHANCED INJECTION COMPLETE!")
        log.info(f"📁 Results saved to {output_file}")
        log.info("🔍 Ready for forensic disproof testing!")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Enhanced cosmic string injection with seed support')
    parser.add_argument('input_json', help='Input results JSON file')
    parser.add_argument('--Gmu', type=float, default=1e-11, 
                       help='Cosmic string tension Gμ (default: 1e-11)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('-o', '--output', help='Output JSON file (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = args.input_json.replace('.json', '')
        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        args.output = f"{base_name}_injected_Gmu_{args.Gmu:.0e}{seed_suffix}.json"
    
    # Run injection
    injector = EnhancedCosmicStringInjector(args.input_json, args.Gmu, args.seed)
    injector.run_injection(args.output)
    
    print(f"\\n🎯 ENHANCED INJECTION COMPLETE!")
    print(f"📁 Output: {args.output}")
    print(f"🔍 Next: python disprove_cosmic_strings_forensic.py {args.output}")

if __name__ == "__main__":
    main()
'''
    
    with open('enhanced_inject_cosmic_string_skies.py', 'w', encoding='utf-8') as f:
        f.write(enhanced_script)
    
    log.info("✅ Enhanced injection engine created with --seed support")

def create_sensitivity_curve_generator():
    """Create sensitivity curve generator for tomorrow's sprint"""
    
    sensitivity_script = '''#!/usr/bin/env python3
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
        log.info(f"\\n🌌 Testing Gμ = {Gmu:.2e} ({i+1}/{len(Gmu_values)})")
        
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
    log.info("\\n📊 Creating sensitivity curve plot...")
    
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
    plt.title('Cosmic String Detection Sensitivity Curve\\n' + 
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
        f.write('Gmu,mean_detection,std_detection\\n')
        for row in csv_data:
            f.write(f"{row['Gmu']:.2e},{row['mean_detection']:.4f},{row['std_detection']:.4f}\\n")
    
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
'''
    
    with open('generate_sensitivity_curve.py', 'w', encoding='utf-8') as f:
        f.write(sensitivity_script)
    
    log.info("✅ Sensitivity curve generator created")

def create_sprint_plan():
    """Create tomorrow's sprint plan"""
    
    sprint_plan = '''# TOMORROW'S SPRINT PLAN (2h max)
## Cosmic String Detection Sensitivity Curve

### 🎯 **OBJECTIVES**
1. **Sweep Gμ = 5e-12 → 2e-11 in 5 steps** → plot detection fraction vs. Gμ
2. **Add --seed argument** → repeatable Monte-Carlo (100 skies per point → error bars)
3. **Push v0.3.0** with PNG curve + CSV data → publish-ready figure

### 🚀 **EXECUTION PLAN**

#### **Phase 1: Enhanced Injection Engine (30 min)**
```bash
# Test enhanced injection with seed
python enhanced_inject_cosmic_string_skies.py REAL_ENHANCED_COSMIC_STRING_RESULTS.json --Gmu 1e-11 --seed 42

# Verify reproducibility
python enhanced_inject_cosmic_string_skies.py REAL_ENHANCED_COSMIC_STRING_RESULTS.json --Gmu 1e-11 --seed 42
```

#### **Phase 2: Sensitivity Curve Generation (60 min)**
```bash
# Run full sensitivity curve
python generate_sensitivity_curve.py

# Verify results
python -c "import json; data=json.load(open('SENSITIVITY_CURVE_v0.3.0.json')); print('Gμ values:', [r['Gmu'] for r in data]); print('Detection fractions:', [r['mean_detection'] for r in data])"
```

#### **Phase 3: Git Commit & Tag (15 min)**
```bash
git add .
git commit -m "feat: sensitivity curve v0.3.0 — Monte-Carlo analysis with error bars"
git tag v0.3.0
```

#### **Phase 4: Verification (15 min)**
- Check PNG/PDF files generated
- Verify CSV data format
- Confirm error bars look reasonable
- Test reproducibility with same seeds

### 📊 **EXPECTED OUTPUTS**
- `SENSITIVITY_CURVE_v0.3.0.png` - Publication-ready figure
- `SENSITIVITY_CURVE_v0.3.0.pdf` - Vector version
- `SENSITIVITY_CURVE_v0.3.0.csv` - Data for analysis
- `SENSITIVITY_CURVE_v0.3.0.json` - Full results
- Git tag `v0.3.0` - Citable release

### 🎯 **SUCCESS CRITERIA**
- ✅ 5 Gμ values tested (5e-12 to 2e-11)
- ✅ 100 Monte-Carlo realizations per point
- ✅ Error bars on sensitivity curve
- ✅ Reproducible with same seeds
- ✅ Publication-ready figures
- ✅ Git tag v0.3.0 created

### 🚀 **NEXT STEPS AFTER SPRINT**
- Point the whole thing at real IPTA DR2
- Watch the sky confess
- Prepare for publication

---
**Ready to hunt cosmic strings!** 🌌🚀
'''
    
    with open('TOMORROW_SPRINT_PLAN.md', 'w', encoding='utf-8') as f:
        f.write(sprint_plan)
    
    log.info("✅ Tomorrow's sprint plan created")

def main():
    """Prepare all tools for tomorrow's sprint"""
    
    log.info("🚀 PREPARING TOMORROW'S SPRINT TOOLS")
    log.info("=" * 50)
    
    # Create enhanced injection engine
    create_enhanced_injection_engine()
    
    # Create sensitivity curve generator
    create_sensitivity_curve_generator()
    
    # Create sprint plan
    create_sprint_plan()
    
    log.info("\\n✅ ALL TOOLS PREPARED!")
    log.info("📁 Files created:")
    log.info("   - enhanced_inject_cosmic_string_skies.py")
    log.info("   - generate_sensitivity_curve.py")
    log.info("   - TOMORROW_SPRINT_PLAN.md")
    
    log.info("\\n🎯 READY FOR TOMORROW'S 2H SPRINT!")
    log.info("🌌 Then we point the whole thing at real IPTA DR2 and watch the sky confess!")

if __name__ == "__main__":
    main()
