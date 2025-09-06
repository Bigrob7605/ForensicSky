#!/usr/bin/env python3
"""
Test Project Functionality
Verify that the cosmic strings project is working after cleanup
"""

import sys
import traceback
from COSMIC_STRINGS_TOOLKIT import CosmicStringsToolkit

def test_basic_functionality():
    """Test basic functionality of the cosmic strings toolkit."""
    print("🧪 TESTING PROJECT FUNCTIONALITY AFTER CLEANUP")
    print("=" * 60)
    
    try:
        # Test 1: Import and initialization
        print("1. Testing toolkit initialization...")
        toolkit = CosmicStringsToolkit()
        print("   ✅ Toolkit initialized successfully")
        
        # Test 2: Data loading
        print("2. Testing data loading...")
        data = toolkit.load_ipta_data()
        print(f"   ✅ Data loaded: {len(data['pulsar_names'])} pulsars")
        
        # Test 3: Signal calculation
        print("3. Testing cosmic string signal calculation...")
        positions = data['pulsar_positions'][:3]  # Use first 3 pulsars
        distances = data['pulsar_distances'][:3]
        signal = toolkit.calculate_cosmic_string_signal(1e-10, positions, distances)
        print(f"   ✅ Signal calculated: {signal.shape}")
        
        # Test 4: Upper limits
        print("4. Testing upper limit calculation...")
        limits = toolkit.calculate_upper_limits()
        print(f"   ✅ Upper limits: {limits['upper_limit_95']:.2e}")
        
        # Test 5: Comprehensive analysis
        print("5. Testing comprehensive analysis...")
        results = toolkit.run_comprehensive_analysis()
        print(f"   ✅ Analysis completed: {len(results)} result categories")
        
        # Test 6: Report generation
        print("6. Testing report generation...")
        report = toolkit.generate_report()
        print(f"   ✅ Report generated: {len(report)} characters")
        
        # Test 7: Results saving
        print("7. Testing results saving...")
        filename = toolkit.save_results("test_results.npz")
        if filename:
            print(f"   ✅ Results saved to: {filename}")
        else:
            print("   ⚠️  No results to save")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("Status: Project is functional after cleanup")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_data_structure():
    """Test the data structure and organization."""
    print("\n📁 TESTING DATA STRUCTURE")
    print("=" * 40)
    
    import os
    
    # Check main directories
    directories = [
        "01_Core_Engine",
        "02_Data", 
        "03_Analysis",
        "04_Results",
        "05_Visualizations",
        "06_Documentation",
        "07_Tests",
        "08_Archive",
        "09_Config",
        "10_Notebooks"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/ exists")
        else:
            print(f"   ❌ {directory}/ missing")
    
    # Check essential files
    essential_files = [
        "COSMIC_STRINGS_TOOLKIT.py",
        "README.md"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            print(f"   ✅ {file} exists")
        else:
            print(f"   ❌ {file} missing")

def main():
    """Main test function."""
    print("🚀 COSMIC STRINGS PROJECT - POST-CLEANUP VERIFICATION")
    print("=" * 70)
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test data structure
    test_data_structure()
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 20)
    if functionality_ok:
        print("✅ Project is functional after cleanup")
        print("✅ Core toolkit works correctly")
        print("✅ Basic analysis pipeline operational")
        print("\n🎯 NEXT STEPS:")
        print("   - Run production gold standard tests")
        print("   - Validate with real IPTA DR2 data")
        print("   - Complete comprehensive analysis")
    else:
        print("❌ Project has issues after cleanup")
        print("❌ Core functionality needs repair")
        print("\n🔧 REQUIRED ACTIONS:")
        print("   - Fix identified issues")
        print("   - Restore missing components")
        print("   - Re-run functionality tests")

if __name__ == "__main__":
    main()
