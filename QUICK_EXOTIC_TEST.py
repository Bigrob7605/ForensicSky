#!/usr/bin/env python3
"""
Quick test of the Exotic Physics Hunter
"""

print("🌌 EXOTIC PHYSICS HUNTER - QUICK TEST")
print("="*50)

try:
    from EXOTIC_PHYSICS_HUNTER import ExoticPhysicsHunter
    print("✅ Exotic Physics Hunter imported successfully")
    
    from IPTA_TIMING_PARSER import load_ipta_timing_data
    print("✅ IPTA parser imported successfully")
    
    # Initialize hunter
    hunter = ExoticPhysicsHunter()
    print("✅ Exotic Physics Hunter initialized")
    
    # Load data
    print("📡 Loading IPTA data...")
    data = load_ipta_timing_data()
    print(f"✅ Loaded {len(data)} pulsars")
    
    # Test on first few pulsars
    if len(data) > 0:
        test_data = dict(list(data.items())[:3])  # Test on first 3 pulsars
        print(f"🔍 Testing exotic physics hunt on {len(test_data)} pulsars...")
        
        results = hunter.hunt_all_exotic_physics(test_data)
        print("✅ Exotic physics hunt completed!")
        
        # Show summary
        summary = results['summary']
        print(f"\n📊 QUICK RESULTS:")
        print(f"   Total detections: {summary['total_detections']}")
        print(f"   Highest significance: {summary['highest_significance']:.2f}")
        print(f"   Discovery candidates: {len(summary['discovery_candidates'])}")
        
        # Show individual results
        for target, result in results.items():
            if target != 'summary':
                print(f"   {target}: {len(result['detections'])} detections, significance {result['significance']:.2f}")
    
    print("\n🎯 EXOTIC PHYSICS HUNTER READY FOR FULL ANALYSIS!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
