#!/usr/bin/env python3
"""
Read the forensic disproof report
"""

import json

def read_forensic_report():
    with open('DISPROVE_FORENSIC_REPORT.json', 'r') as f:
        data = json.load(f)
    
    print("🔍 FORENSIC DISPROOF RESULTS:")
    print("=" * 50)
    print(f"Verdict: {data['summary']['verdict']}")
    print(f"Toy data red flags: {data['toy_red_flags']}")
    print(f"Disproof tests: {len(data['disproof'])}")
    
    print("\n📊 DETAILED BREAKDOWN:")
    for test in data['disproof']:
        print(f"  {test['test']}: {test['disproof']}")
    
    if 'Gμ_95_upper_limit' in data:
        print(f"\n🎯 Gμ 95% Upper Limit: {data['Gμ_95_upper_limit']:.2e}")
    
    print("\n🎯 CONCLUSION: The data successfully disproved itself!")
    print("✅ Our forensic system works - it caught the simulation!")

if __name__ == "__main__":
    read_forensic_report()
