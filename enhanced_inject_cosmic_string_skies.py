#!/usr/bin/env python3
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
            hd_correlation = 0.5 * (1 - np.cos(angular_separations)) * np.log(0.5 * (1 - np.cos(angular_separations))) -                            0.25 * (1 - np.cos(angular_separations)) + 0.25 * (3 + np.cos(angular_separations)) * np.log(0.5 * (3 + np.cos(angular_separations)))
            
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
    
    print(f"\n🎯 ENHANCED INJECTION COMPLETE!")
    print(f"📁 Output: {args.output}")
    print(f"🔍 Next: python disprove_cosmic_strings_forensic.py {args.output}")

if __name__ == "__main__":
    main()
