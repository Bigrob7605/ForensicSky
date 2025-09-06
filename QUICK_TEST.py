#!/usr/bin/env python3
"""
Quick test to verify everything is working
"""

print("🧪 QUICK TEST - VERIFYING SYSTEM")
print("="*40)

try:
    from GRAVITATIONAL_WAVE_MEMORY_HUNTER import GravitationalWaveMemoryHunter
    print("✅ Memory hunter imported successfully")
    
    from IPTA_TIMING_PARSER import load_ipta_timing_data
    print("✅ IPTA parser imported successfully")
    
    # Test memory hunter
    hunter = GravitationalWaveMemoryHunter()
    print("✅ Memory hunter initialized")
    
    # Test data loading
    print("📡 Testing data loading...")
    data = load_ipta_timing_data()
    print(f"✅ Loaded {len(data)} pulsars")
    
    if len(data) > 0:
        # Test analysis on first few pulsars
        test_data = dict(list(data.items())[:5])
        print(f"🔍 Testing analysis on {len(test_data)} pulsars...")
        
        results = hunter.detect_memory_effects(test_data)
        print(f"✅ Analysis completed")
        print(f"   Step candidates: {len(results['step_candidates'])}")
        print(f"   Coincident events: {len(results['coincident_events'])}")
        print(f"   Memory effects: {len(results['memory_effects'])}")
        print(f"   Significance: {results['significance']:.2f}")
    
    print("\n🎯 SYSTEM READY FOR COSMIC STRING HUNTING!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
