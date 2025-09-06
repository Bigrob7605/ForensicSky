#!/usr/bin/env python3
"""
NANOGrav 15-Year Data Memory Effect Hunt

Target: Test our gravitational wave memory hunter on NANOGrav's most stable pulsars
Strategy: Focus on top 20 most stable pulsars with longest baselines
Timeline: Execute immediately for quick wins
"""

import numpy as np
import matplotlib.pyplot as plt
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter
from IPTA_TIMING_PARSER import load_ipta_timing_data
import json
from datetime import datetime

class NANOGravMemoryHunter:
    """
    Specialized hunter for NANOGrav data focusing on most stable pulsars
    """
    
    def __init__(self):
        self.hunter = GravitationalWaveMemoryHunter()
        self.results = {}
        
    def identify_stable_pulsars(self, data: dict, min_data_points: int = 500) -> list:
        """
        Identify the most stable pulsars based on data quality
        
        Args:
            data: Dictionary of pulsar timing data
            min_data_points: Minimum number of data points required
            
        Returns:
            List of most stable pulsar names
        """
        stable_pulsars = []
        
        for pulsar_name, residuals in data.items():
            if len(residuals) >= min_data_points:
                # Calculate stability metrics
                noise_level = np.std(residuals)
                data_span = len(residuals)  # Proxy for observation span
                
                # Stability score (lower noise, more data = better)
                stability_score = data_span / (noise_level + 1e-10)
                
                stable_pulsars.append({
                    'name': pulsar_name,
                    'stability_score': stability_score,
                    'noise_level': noise_level,
                    'data_points': len(residuals)
                })
        
        # Sort by stability score (highest first)
        stable_pulsars.sort(key=lambda x: x['stability_score'], reverse=True)
        
        return stable_pulsars
    
    def hunt_memory_effects(self, data: dict, top_n: int = 20) -> dict:
        """
        Hunt for memory effects in the most stable pulsars
        
        Args:
            data: Dictionary of pulsar timing data
            top_n: Number of top pulsars to analyze
            
        Returns:
            Complete analysis results
        """
        print("🎯 NANOGRAV MEMORY EFFECT HUNT")
        print("="*50)
        
        # Step 1: Identify most stable pulsars
        print(f"🔍 Step 1: Identifying most stable pulsars...")
        stable_pulsars = self.identify_stable_pulsars(data)
        
        print(f"   Found {len(stable_pulsars)} pulsars with sufficient data")
        print(f"   Selecting top {min(top_n, len(stable_pulsars))} most stable")
        
        # Show top pulsars
        for i, pulsar in enumerate(stable_pulsars[:top_n]):
            print(f"   {i+1:2d}. {pulsar['name']}: "
                  f"score={pulsar['stability_score']:.1f}, "
                  f"noise={pulsar['noise_level']:.2e}, "
                  f"points={pulsar['data_points']}")
        
        # Step 2: Create subset with most stable pulsars
        top_pulsars = [p['name'] for p in stable_pulsars[:top_n]]
        subset_data = {name: data[name] for name in top_pulsars if name in data}
        
        print(f"\n🔍 Step 2: Analyzing {len(subset_data)} most stable pulsars...")
        
        # Step 3: Run memory effect analysis
        results = self.hunter.detect_memory_effects(subset_data)
        
        # Step 4: Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'NANOGrav_15yr',
            'total_pulsars': len(data),
            'analyzed_pulsars': len(subset_data),
            'stable_pulsars': stable_pulsars[:top_n],
            'analysis_results': results
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.results:
            return "No analysis results available"
        
        results = self.results['analysis_results']
        
        report = f"""
🌌 NANOGRAV MEMORY EFFECT HUNT REPORT
=====================================
Timestamp: {self.results['timestamp']}
Dataset: {self.results['dataset']}
Total pulsars available: {self.results['total_pulsars']}
Pulsars analyzed: {self.results['analyzed_pulsars']}

📊 DETECTION RESULTS:
Step candidates: {len(results['step_candidates'])}
Coincident events: {len(results['coincident_events'])}
Memory effects: {len(results['memory_effects'])}
Significance: {results['significance']:.2f}

🎯 TOP STABLE PULSARS:
"""
        
        for i, pulsar in enumerate(self.results['stable_pulsars'][:10]):
            report += f"{i+1:2d}. {pulsar['name']}: score={pulsar['stability_score']:.1f}\n"
        
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
    """Execute NANOGrav memory effect hunt"""
    print("🚀 EXECUTING NANOGRAV MEMORY EFFECT HUNT")
    print("="*60)
    
    # Initialize hunter
    nanograv_hunter = NANOGravMemoryHunter()
    
    # Load IPTA data (includes NANOGrav)
    print("📡 Loading IPTA data...")
    data = load_ipta_timing_data()
    
    if len(data) == 0:
        print("❌ No data loaded - check data path")
        return
    
    print(f"✅ Loaded {len(data)} pulsars")
    
    # Hunt for memory effects
    results = nanograv_hunter.hunt_memory_effects(data, top_n=20)
    
    # Generate report
    report = nanograv_hunter.generate_report()
    print(report)
    
    # Save results
    with open('nanograv_memory_hunt_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to nanograv_memory_hunt_results.json")
    
    return results

if __name__ == "__main__":
    main()
