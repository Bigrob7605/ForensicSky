#!/usr/bin/env python3
"""
J2145-0750 STRING BEACON CANDIDATE ANALYSIS
==========================================

Hub-and-spoke pattern analysis for cosmic string wake detection
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')

def sky_geometry_reality_check():
    """Check if J2145-0750 and correlated pulsars form a linear filament"""
    print("🔍 SKY-GEOMETRY REALITY CHECK")
    print("=" * 50)
    
    # The hub-and-spoke pattern from our quantum analysis
    psr_names = ['J2145-0750', 'J1600-3053', 'J1643-1224', 'J0613-0200', 'J0610-2100', 'J1802-2124']
    
    print(f"Analyzing {len(psr_names)} pulsars in hub-and-spoke pattern:")
    for i, name in enumerate(psr_names):
        print(f"  {i+1}. {name}")
    
    # Convert to SkyCoord format
    # J2145-0750 = 21:45:xx, -07:50:xx
    # J1600-3053 = 16:00:xx, -30:53:xx  
    # J1643-1224 = 16:43:xx, -12:24:xx
    # J0613-0200 = 06:13:xx, -02:00:xx
    # J0610-2100 = 06:10:xx, -21:00:xx
    # J1802-2124 = 18:02:xx, -21:24:xx
    
    # Parse coordinates (approximate for now)
    coords = []
    for name in psr_names:
        if name == 'J2145-0750':
            coords.append('21h45m00s -07d50m00s')
        elif name == 'J1600-3053':
            coords.append('16h00m00s -30d53m00s')
        elif name == 'J1643-1224':
            coords.append('16h43m00s -12d24m00s')
        elif name == 'J0613-0200':
            coords.append('06h13m00s -02d00m00s')
        elif name == 'J0610-2100':
            coords.append('06h10m00s -21d00m00s')
        elif name == 'J1802-2124':
            coords.append('18h02m00s -21d24m00s')
    
    try:
        # Create SkyCoord objects
        c = SkyCoord(coords, frame='icrs')
        
        print(f"\n📍 PULSAR COORDINATES:")
        for i, (name, coord) in enumerate(zip(psr_names, coords)):
            print(f"  {name}: {coord}")
        
        # Calculate great-circle distances from J2145-0750 (center)
        center = c[0]  # J2145-0750
        separations = center.separation(c)
        
        print(f"\n📏 ANGULAR SEPARATIONS FROM J2145-0750:")
        print("Pulsar | Separation (deg) | Status")
        print("-" * 40)
        
        for i, (name, sep) in enumerate(zip(psr_names, separations)):
            sep_deg = sep.value
            if i == 0:
                status = "CENTER"
            elif sep_deg < 15:
                status = "✅ CLOSE (< 15°)"
            else:
                status = "❌ FAR (> 15°)"
            print(f"{name} | {sep_deg:8.2f} | {status}")
        
        # Check if separations are reasonable for string filament
        close_pulsars = separations[1:] < 15 * u.deg  # Exclude center
        n_close = np.sum(close_pulsars)
        
        print(f"\n🎯 FILAMENT CANDIDATE CHECK:")
        print(f"Pulsars within 15° of J2145-0750: {n_close}/5")
        
        if n_close >= 3:
            print("✅ STRONG FILAMENT CANDIDATE - Multiple pulsars in close proximity!")
        elif n_close >= 2:
            print("⚠️  WEAK FILAMENT CANDIDATE - Some pulsars in close proximity")
        else:
            print("❌ NO FILAMENT CANDIDATE - Pulsars too widely separated")
        
        # Project onto galactic plane
        print(f"\n🌌 GALACTIC COORDINATES:")
        l = c.galactic.l.value
        b = c.galactic.b.value
        
        print("Pulsar | Galactic l (deg) | Galactic b (deg)")
        print("-" * 45)
        for i, name in enumerate(psr_names):
            print(f"{name} | {l[i]:12.2f} | {b[i]:12.2f}")
        
        # Check if galactic longitudes line up (string filament should be roughly straight)
        l_center = l[0]
        l_diffs = np.abs(l - l_center)
        l_aligned = l_diffs < 5  # Within 5 degrees
        
        print(f"\n🔍 GALACTIC LONGITUDE ALIGNMENT:")
        print(f"Longitude differences from J2145-0750:")
        for i, (name, diff) in enumerate(zip(psr_names, l_diffs)):
            status = "✅ ALIGNED" if diff < 5 else "❌ NOT ALIGNED"
            print(f"  {name}: {diff:6.2f}° | {status}")
        
        n_aligned = np.sum(l_aligned[1:])  # Exclude center
        print(f"\nPulsars aligned within 5°: {n_aligned}/5")
        
        if n_aligned >= 3:
            print("✅ STRONG LINEAR FILAMENT - Galactic longitudes align!")
        elif n_aligned >= 2:
            print("⚠️  WEAK LINEAR FILAMENT - Some alignment detected")
        else:
            print("❌ NO LINEAR FILAMENT - Longitudes don't align")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if n_close >= 3 and n_aligned >= 3:
            print("🚀 STRONG COSMIC STRING CANDIDATE!")
            print("   - Multiple pulsars in close proximity")
            print("   - Galactic longitudes align")
            print("   - Hub-and-spoke pattern consistent with string wake")
        elif n_close >= 2 or n_aligned >= 2:
            print("⚠️  MODERATE COSMIC STRING CANDIDATE")
            print("   - Some geometric consistency")
            print("   - Worth further investigation")
        else:
            print("❌ WEAK COSMIC STRING CANDIDATE")
            print("   - Limited geometric consistency")
            print("   - May be background correlation")
        
        return c, separations, l, b
        
    except Exception as e:
        print(f"❌ Error in sky geometry analysis: {e}")
        return None, None, None, None

def create_string_beacon_plot(c, separations, l, b):
    """Create the one-plot kill-shot visualization"""
    print(f"\n📊 CREATING STRING BEACON VISUALIZATION")
    print("=" * 50)
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Panel A: Sky map (galactic coords)
        ax1.scatter(l, b, s=100, c='red', alpha=0.7, label='Correlated Pulsars')
        ax1.scatter(l[0], b[0], s=200, c='gold', marker='*', label='J2145-0750 (Hub)')
        
        # Draw lines from hub to spokes
        for i in range(1, len(l)):
            ax1.plot([l[0], l[i]], [b[0], b[i]], 'k--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Galactic Longitude (deg)')
        ax1.set_ylabel('Galactic Latitude (deg)')
        ax1.set_title('Hub-and-Spoke Pattern: J2145-0750 String Beacon Candidate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Panel B: Angular separations
        psr_names = ['J2145-0750', 'J1600-3053', 'J1643-1224', 'J0613-0200', 'J0610-2100', 'J1802-2124']
        separations_deg = separations.value
        
        bars = ax2.bar(range(len(psr_names)), separations_deg, color=['gold' if i == 0 else 'red' for i in range(len(psr_names))])
        ax2.set_xlabel('Pulsar Index')
        ax2.set_ylabel('Angular Separation from J2145-0750 (deg)')
        ax2.set_title('Angular Separations from Hub')
        ax2.set_xticks(range(len(psr_names)))
        ax2.set_xticklabels([name.replace('J', '') for name in psr_names], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add threshold line
        ax2.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='15° Threshold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('j2145_string_beacon_candidate.png', dpi=300, bbox_inches='tight')
        print("✅ Plot saved as 'j2145_string_beacon_candidate.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        return False

def main():
    """Run the J2145-0750 string beacon analysis"""
    print("🚀 J2145-0750 STRING BEACON CANDIDATE ANALYSIS")
    print("=" * 60)
    print("Hub-and-spoke pattern analysis for cosmic string wake detection")
    print()
    
    # Run sky geometry reality check
    c, separations, l, b = sky_geometry_reality_check()
    
    if c is not None:
        # Create visualization
        create_string_beacon_plot(c, separations, l, b)
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. ✅ Sky geometry check completed")
        print("2. 🔄 Run timing-residual string wake template fit")
        print("3. 🔄 Cross-match with CHIME FRB sky")
        print("4. 📝 Draft ApJL note if geometry holds")
        
        # Save results
        results = {
            'pulsar_names': ['J2145-0750', 'J1600-3053', 'J1643-1224', 'J0613-0200', 'J0610-2100', 'J1802-2124'],
            'angular_separations': separations.value.tolist(),
            'galactic_longitudes': l.tolist(),
            'galactic_latitudes': b.tolist(),
            'analysis_timestamp': '2025-09-05T20:00:00Z'
        }
        
        with open('j2145_string_beacon_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("📊 Results saved to 'j2145_string_beacon_results.json'")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
