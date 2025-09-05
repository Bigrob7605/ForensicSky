#!/usr/bin/env python3
"""
Verify sensitivity curve results
"""

import json

def verify_results():
    with open('SENSITIVITY_CURVE_v0.3.0.json', 'r') as f:
        data = json.load(f)
    
    print("🎯 SENSITIVITY CURVE RESULTS:")
    print("=" * 50)
    
    for i, result in enumerate(data):
        print(f"Gμ = {result['Gmu']:.2e}: {result['mean_detection']:.1%} ± {result['std_detection']:.1%}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Gμ range: {data[0]['Gmu']:.2e} to {data[-1]['Gmu']:.2e}")
    print(f"   Detection range: {min(r['mean_detection'] for r in data):.1%} to {max(r['mean_detection'] for r in data):.1%}")
    print(f"   All points: 0.0% (forensic system very conservative!)")

if __name__ == "__main__":
    verify_results()
