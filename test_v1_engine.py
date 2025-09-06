#!/usr/bin/env python3
"""
TEST V1 ENGINE - Test the Core ForensicSky V1 engine methods
"""

import sys
sys.path.append('01_Core_Engine')

from Core_ForensicSky_V1 import CoreForensicSkyV1

def main():
    print("🚀 TESTING V1 ENGINE METHODS...")
    print("=" * 50)
    
    # Initialize the V1 engine
    engine = CoreForensicSkyV1()
    
    print("📊 Testing correlation analysis...")
    try:
        engine.correlation_analysis()
        print("✅ Correlation analysis works!")
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
    
    print("\n📈 Testing spectral analysis...")
    try:
        engine.spectral_analysis()
        print("✅ Spectral analysis works!")
    except Exception as e:
        print(f"❌ Spectral analysis failed: {e}")
    
    print("\n🤖 Testing ML analysis...")
    try:
        engine.ml_analysis()
        print("✅ ML analysis works!")
    except Exception as e:
        print(f"❌ ML analysis failed: {e}")
    
    print("\n🔬 Testing forensic disproof analysis...")
    try:
        engine.forensic_disproof_analysis()
        print("✅ Forensic disproof analysis works!")
    except Exception as e:
        print(f"❌ Forensic disproof analysis failed: {e}")
    
    print("\n🎉 V1 ENGINE TEST COMPLETE!")

if __name__ == "__main__":
    main()
