#!/usr/bin/env python3
"""
Debug IPTA Data Loading

This script helps debug why timing data is not loading properly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def debug_data_loading():
    """Debug the data loading process"""
    
    print("🔍 DEBUGGING IPTA DATA LOADING")
    print("="*50)
    
    try:
        from COMPREHENSIVE_IPTA_DATA_LOADER import ComprehensiveIPTADataLoader
        
        # Initialize loader
        loader = ComprehensiveIPTADataLoader()
        
        # Check data directory
        data_root = Path("02_Data/ipta_dr2")
        print(f"📁 Data root: {data_root}")
        print(f"   Exists: {data_root.exists()}")
        
        if data_root.exists():
            # List all .tim files
            tim_files = list(data_root.glob("**/*.tim"))
            print(f"📄 Found {len(tim_files)} .tim files")
            
            if len(tim_files) > 0:
                # Show first few files
                print("   First 5 .tim files:")
                for i, tim_file in enumerate(tim_files[:5]):
                    print(f"     {i+1}. {tim_file}")
                
                # Try to load first file manually
                print(f"\n🧪 Testing first file: {tim_files[0]}")
                times, residuals, uncertainties = loader._load_tim_file(tim_files[0])
                print(f"   Times loaded: {len(times)}")
                print(f"   Residuals loaded: {len(residuals)}")
                print(f"   Uncertainties loaded: {len(uncertainties)}")
                
                if len(times) > 0:
                    print(f"   First few times: {times[:5]}")
                    print(f"   First few residuals: {residuals[:5]}")
                    print(f"   First few uncertainties: {uncertainties[:5]}")
                
                # Check pulsar name extraction
                pulsar_name = loader._extract_pulsar_name(tim_files[0])
                print(f"   Extracted pulsar name: {pulsar_name}")
                
            else:
                print("❌ No .tim files found!")
                
        else:
            print("❌ Data directory does not exist!")
            
        # Try to run the actual loader
        print(f"\n🚀 Running actual data loader...")
        data = loader.load_all_data()
        
        print(f"\n📊 RESULTS:")
        print(f"   Pulsars loaded: {len(data['pulsar_catalog'])}")
        print(f"   Timing datasets loaded: {len(data['timing_data'])}")
        print(f"   Clock files loaded: {len(data['clock_data'])}")
        
        if len(data['timing_data']) > 0:
            print(f"\n✅ SUCCESS! First timing dataset:")
            first_timing = data['timing_data'][0]
            print(f"   Pulsar: {first_timing['pulsar_name']}")
            print(f"   Observations: {first_timing['n_observations']}")
            print(f"   Source: {first_timing['source']}")
        else:
            print(f"\n❌ FAILED! No timing data loaded")
            
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_loading()
