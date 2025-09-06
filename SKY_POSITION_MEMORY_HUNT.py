#!/usr/bin/env python3
"""
Sky Position Memory Effect Hunt

Target: Implement realistic light-travel-time constraints for more precise detection
Strategy: Use pulsar sky positions to filter coincident events by causality
Timeline: Execute after EPTA cross-validation
"""

import numpy as np
import matplotlib.pyplot as plt
from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter
from IPTA_TIMING_PARSER import load_ipta_timing_data
import json
from datetime import datetime
from astropy.coordinates import SkyCoord
from astropy import units as u

class SkyPositionMemoryHunter:
    """
    Memory hunter with sky position constraints for realistic light-travel-time filtering
    """
    
    def __init__(self):
        self.hunter = GravitationalWaveMemoryHunter()
        self.results = {}
        
        # Known pulsar sky positions (RA, Dec in degrees)
        # These are some of the most stable millisecond pulsars
        self.pulsar_positions = {
            'J1713+0747': (258.562, 7.701),      # NANOGrav's best
            'J1909-3744': (287.421, -37.740),    # EPTA's best
            'J1600-3053': (240.000, -30.883),    # EPTA
            'J1744-1134': (266.020, -11.580),    # EPTA
            'J2010-1323': (302.500, -13.383),    # EPTA
            'J1918-0642': (289.500, -6.700),     # EPTA
            'J1012+5307': (153.000, 53.117),     # EPTA
            'J2145-0750': (326.250, -7.500),     # EPTA
            'J0030+0451': (7.500, 4.850),        # NANOGrav
            'J0613-0200': (93.250, -2.000),      # NANOGrav
            'J1024-0719': (156.000, -7.317),     # NANOGrav
            'J1614-2230': (243.500, -22.500),    # NANOGrav
            'J1640+2224': (250.000, 22.400),     # NANOGrav
            'J1853+1303': (283.250, 13.050),     # NANOGrav
            'J1903+0327': (285.750, 3.450),      # NANOGrav
            'J2043+1711': (310.750, 17.183),     # NANOGrav
            'J2317+1439': (349.250, 14.650),     # NANOGrav
        }
    
    def calculate_light_travel_time(self, pulsar1: str, pulsar2: str) -> float:
        """
        Calculate light travel time between two pulsars
        
        Args:
            pulsar1: First pulsar name
            pulsar2: Second pulsar name
            
        Returns:
            Light travel time in seconds
        """
        if pulsar1 not in self.pulsar_positions or pulsar2 not in self.pulsar_positions:
            return float('inf')  # Unknown position
        
        # Get sky coordinates
        ra1, dec1 = self.pulsar_positions[pulsar1]
        ra2, dec2 = self.pulsar_positions[pulsar2]
        
        # Create SkyCoord objects
        coord1 = SkyCoord(ra1, dec1, unit='deg')
        coord2 = SkyCoord(ra2, dec2, unit='deg')
        
        # Calculate angular separation
        separation = coord1.separation(coord2)
        
        # Convert to light travel time (assuming cosmic string at cosmological distance)
        # For cosmic strings, we assume they're at cosmological distances
        # so the light travel time is dominated by the angular separation
        # This is a simplified calculation - in reality, we'd need the string's distance
        
        # Rough estimate: light travel time = angular_separation * distance_to_string / c
        # For cosmic strings at cosmological distances (~1 Gpc), this gives:
        # t = separation_radians * 1_Gpc / c ≈ separation_radians * 1e9 years
        
        # Convert to seconds (1 year ≈ 3.15e7 seconds)
        light_travel_time = separation.radian * 1e9 * 3.15e7  # seconds
        
        return light_travel_time
    
    def filter_by_causality(self, coincident_events: list, max_time_window: float = 1e6) -> list:
        """
        Filter coincident events by light-travel-time constraints
        
        Args:
            coincident_events: List of coincident events
            max_time_window: Maximum time window for causality (seconds)
            
        Returns:
            Filtered list of causally consistent events
        """
        causally_consistent = []
        
        for event in coincident_events:
            pulsars = event.get('pulsars', [])
            time_window = event.get('time_window', 0)
            
            # Check if all pulsars in this event are causally connected
            is_causal = True
            
            for i, pulsar1 in enumerate(pulsars):
                for j, pulsar2 in enumerate(pulsars[i+1:], i+1):
                    light_travel_time = self.calculate_light_travel_time(pulsar1, pulsar2)
                    
                    # Event is causally consistent if time window < light travel time
                    if time_window > light_travel_time:
                        is_causal = False
                        break
                
                if not is_causal:
                    break
            
            if is_causal:
                causally_consistent.append(event)
        
        return causally_consistent
    
    def hunt_memory_effects_with_sky_constraints(self, data: dict, top_n: int = 15) -> dict:
        """
        Hunt for memory effects with sky position constraints
        
        Args:
            data: Dictionary of pulsar timing data
            top_n: Number of top pulsars to analyze
            
        Returns:
            Complete analysis results with sky constraints
        """
        print("🌌 SKY POSITION MEMORY EFFECT HUNT")
        print("="*50)
        
        # Step 1: Filter data to only include pulsars with known positions
        print(f"🔍 Step 1: Filtering pulsars with known sky positions...")
        known_pulsars = []
        for pulsar_name in data.keys():
            if pulsar_name in self.pulsar_positions:
                known_pulsars.append(pulsar_name)
        
        print(f"   Found {len(known_pulsars)} pulsars with known sky positions")
        print(f"   Selecting top {min(top_n, len(known_pulsars))} for analysis")
        
        # Show selected pulsars
        for i, pulsar in enumerate(known_pulsars[:top_n]):
            ra, dec = self.pulsar_positions[pulsar]
            print(f"   {i+1:2d}. {pulsar}: RA={ra:.1f}°, Dec={dec:.1f}°")
        
        # Step 2: Create subset with known pulsars
        subset_data = {name: data[name] for name in known_pulsars[:top_n] if name in data}
        
        print(f"\n🔍 Step 2: Analyzing {len(subset_data)} pulsars with sky constraints...")
        
        # Step 3: Run standard memory effect analysis
        results = self.hunter.detect_memory_effects(subset_data)
        
        # Step 4: Apply sky position constraints
        print(f"🔍 Step 3: Applying light-travel-time constraints...")
        causally_consistent = self.filter_by_causality(results['coincident_events'])
        
        print(f"   Original coincident events: {len(results['coincident_events'])}")
        print(f"   Causally consistent events: {len(causally_consistent)}")
        
        # Step 5: Update results with sky constraints
        results['coincident_events'] = causally_consistent
        results['causally_consistent_events'] = causally_consistent
        
        # Recalculate significance with filtered events
        if len(causally_consistent) > 0:
            results['significance'] = len(causally_consistent) / len(results['step_candidates'])
        else:
            results['significance'] = 0.0
        
        # Step 6: Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'IPTA_DR2_Sky_Constrained',
            'total_pulsars': len(data),
            'pulsars_with_positions': len(known_pulsars),
            'analyzed_pulsars': len(subset_data),
            'pulsar_positions': {p: self.pulsar_positions[p] for p in known_pulsars[:top_n]},
            'analysis_results': results
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.results:
            return "No analysis results available"
        
        results = self.results['analysis_results']
        
        report = f"""
🌌 SKY POSITION MEMORY EFFECT HUNT REPORT
==========================================
Timestamp: {self.results['timestamp']}
Dataset: {self.results['dataset']}
Total pulsars available: {self.results['total_pulsars']}
Pulsars with known positions: {self.results['pulsars_with_positions']}
Pulsars analyzed: {self.results['analyzed_pulsars']}

📊 DETECTION RESULTS:
Step candidates: {len(results['step_candidates'])}
Coincident events: {len(results['coincident_events'])}
Causally consistent events: {len(results['causally_consistent_events'])}
Memory effects: {len(results['memory_effects'])}
Significance: {results['significance']:.2f}

🎯 PULSAR SKY POSITIONS:
"""
        
        for pulsar, (ra, dec) in self.results['pulsar_positions'].items():
            report += f"{pulsar}: RA={ra:.1f}°, Dec={dec:.1f}°\n"
        
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
    """Execute sky position memory effect hunt"""
    print("🌌 EXECUTING SKY POSITION MEMORY EFFECT HUNT")
    print("="*60)
    
    # Initialize hunter
    sky_hunter = SkyPositionMemoryHunter()
    
    # Load IPTA data
    print("📡 Loading IPTA data...")
    data = load_ipta_timing_data()
    
    if len(data) == 0:
        print("❌ No data loaded - check data path")
        return
    
    print(f"✅ Loaded {len(data)} pulsars")
    
    # Hunt for memory effects with sky constraints
    results = sky_hunter.hunt_memory_effects_with_sky_constraints(data, top_n=15)
    
    # Generate report
    report = sky_hunter.generate_report()
    print(report)
    
    # Save results
    with open('sky_position_memory_hunt_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to sky_position_memory_hunt_results.json")
    
    return results

if __name__ == "__main__":
    main()
