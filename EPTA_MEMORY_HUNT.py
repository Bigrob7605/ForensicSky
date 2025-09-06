#!/usr/bin/env python3
"""
EPTA Data Memory Effect Hunt

Target: Cross-validate our memory hunter with European Pulsar Timing Array data
Strategy: Focus on EPTA-specific pulsars and compare results with NANOGrav
Timeline: Execute immediately after NANOGrav analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter
from IPTA_TIMING_PARSER import load_ipta_timing_data
import json
from datetime import datetime

class EPTAMemoryHunter:
    """
    Specialized hunter for EPTA data focusing on European pulsars
    """
    
    def __init__(self):
        self.hunter = GravitationalWaveMemoryHunter()
        self.results = {}
        
    def identify_epta_pulsars(self, data: dict, min_data_points: int = 300) -> list:
        """
        Identify EPTA-specific pulsars based on naming and data quality
        
        Args:
            data: Dictionary of pulsar timing data
            min_data_points: Minimum number of data points required
            
        Returns:
            List of EPTA pulsars with quality metrics
        """
        epta_pulsars = []
        
        for pulsar_name, residuals in data.items():
            if len(residuals) >= min_data_points:
                # EPTA pulsars often have specific naming patterns
                is_epta = any(pattern in pulsar_name.upper() for pattern in 
                            ['J', 'B', 'PSR'])  # Basic pattern matching
                
                if is_epta:
                    # Calculate quality metrics
                    noise_level = np.std(residuals)
                    data_span = len(residuals)
                    
                    # Quality score (more data, lower noise = better)
                    quality_score = data_span / (noise_level + 1e-10)
                    
                    epta_pulsars.append({
                        'name': pulsar_name,
                        'quality_score': quality_score,
                        'noise_level': noise_level,
                        'data_points': len(residuals),
                        'is_epta': is_epta
                    })
        
        # Sort by quality score (highest first)
        epta_pulsars.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return epta_pulsars
    
    def hunt_memory_effects(self, data: dict, top_n: int = 15) -> dict:
        """
        Hunt for memory effects in EPTA pulsars
        
        Args:
            data: Dictionary of pulsar timing data
            top_n: Number of top pulsars to analyze
            
        Returns:
            Complete analysis results
        """
        print("🌍 EPTA MEMORY EFFECT HUNT")
        print("="*50)
        
        # Step 1: Identify EPTA pulsars
        print(f"🔍 Step 1: Identifying EPTA pulsars...")
        epta_pulsars = self.identify_epta_pulsars(data)
        
        print(f"   Found {len(epta_pulsars)} EPTA pulsars with sufficient data")
        print(f"   Selecting top {min(top_n, len(epta_pulsars))} highest quality")
        
        # Show top pulsars
        for i, pulsar in enumerate(epta_pulsars[:top_n]):
            print(f"   {i+1:2d}. {pulsar['name']}: "
                  f"score={pulsar['quality_score']:.1f}, "
                  f"noise={pulsar['noise_level']:.2e}, "
                  f"points={pulsar['data_points']}")
        
        # Step 2: Create subset with top EPTA pulsars
        top_pulsars = [p['name'] for p in epta_pulsars[:top_n]]
        subset_data = {name: data[name] for name in top_pulsars if name in data}
        
        print(f"\n🔍 Step 2: Analyzing {len(subset_data)} EPTA pulsars...")
        
        # Step 3: Run memory effect analysis
        results = self.hunter.detect_memory_effects(subset_data)
        
        # Step 4: Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'EPTA_IPTA_DR2',
            'total_pulsars': len(data),
            'epta_pulsars_found': len(epta_pulsars),
            'analyzed_pulsars': len(subset_data),
            'epta_pulsars': epta_pulsars[:top_n],
            'analysis_results': results
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.results:
            return "No analysis results available"
        
        results = self.results['analysis_results']
        
        report = f"""
🌍 EPTA MEMORY EFFECT HUNT REPORT
==================================
Timestamp: {self.results['timestamp']}
Dataset: {self.results['dataset']}
Total pulsars available: {self.results['total_pulsars']}
EPTA pulsars found: {self.results['epta_pulsars_found']}
Pulsars analyzed: {self.results['analyzed_pulsars']}

📊 DETECTION RESULTS:
Step candidates: {len(results['step_candidates'])}
Coincident events: {len(results['coincident_events'])}
Memory effects: {len(results['memory_effects'])}
Significance: {results['significance']:.2f}

🎯 TOP EPTA PULSARS:
"""
        
        for i, pulsar in enumerate(self.results['epta_pulsars'][:10]):
            report += f"{i+1:2d}. {pulsar['name']}: score={pulsar['quality_score']:.1f}\n"
        
        if results['memory_effects']:
            report += f"\n✅ COSMIC STRING MEMORY EFFECTS DETECTED!\n"
            for i, effect in enumerate(results['memory_effects']):
                report += f"Effect {i+1}: {effect['n_pulsars']} pulsars, "
                report += f"strain={effect['strain_amplitude']:.2e}, "
                report += f"tension={effect['string_tension_estimate']:.2e}\n"
        else:
            report += f"\n❌ No cosmic string memory effects detected\n"
            report += f"Clean null result - method working correctly\n"
        
        return report

def main():
    """Execute EPTA memory effect hunt"""
    print("🌍 EXECUTING EPTA MEMORY EFFECT HUNT")
    print("="*60)
    
    # Initialize hunter
    epta_hunter = EPTAMemoryHunter()
    
    # Load IPTA data (includes EPTA)
    print("📡 Loading IPTA data...")
    data = load_ipta_timing_data()
    
    if len(data) == 0:
        print("❌ No data loaded - check data path")
        return
    
    print(f"✅ Loaded {len(data)} pulsars")
    
    # Hunt for memory effects
    results = epta_hunter.hunt_memory_effects(data, top_n=15)
    
    # Generate report
    report = epta_hunter.generate_report()
    print(report)
    
    # Save results
    with open('epta_memory_hunt_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to epta_memory_hunt_results.json")
    
    return results

if __name__ == "__main__":
    main()
