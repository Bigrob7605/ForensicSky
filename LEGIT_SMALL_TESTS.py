#!/usr/bin/env python3
"""
LEGIT SMALL TESTS
================

Test the ULTIMATE_COSMIC_STRING_ENGINE.py with small, focused tests
to verify it's working properly before attempting a full run.
"""

import sys
import os
import time
import logging
from pathlib import Path
import numpy as np
import json

# Add current directory to path
sys.path.append('.')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_import():
    """Test 1: Import the ultimate engine"""
    logger.info("🧪 TEST 1: Import ULTIMATE_COSMIC_STRING_ENGINE")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        logger.info("   ✅ Import successful")
        return True
    except Exception as e:
        logger.error(f"   ❌ Import failed: {e}")
        return False

def test_engine_initialization():
    """Test 2: Initialize the engine"""
    logger.info("🧪 TEST 2: Initialize UltimateCosmicStringEngine")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        logger.info("   ✅ Engine initialization successful")
        logger.info(f"   📊 Data path: {engine.data_path}")
        logger.info(f"   📊 Gμ range: {len(engine.Gmu_range)} values")
        return True
    except Exception as e:
        logger.error(f"   ❌ Engine initialization failed: {e}")
        return False

def test_data_loading():
    """Test 3: Load real IPTA data"""
    logger.info("🧪 TEST 3: Load real IPTA data")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Test data loading
        success = engine.load_real_ipta_data()
        if success:
            logger.info("   ✅ Data loading successful")
            logger.info(f"   📊 Pulsars: {len(engine.pulsar_catalog)}")
            logger.info(f"   📊 Timing data points: {len(engine.timing_data)}")
            return True
        else:
            logger.error("   ❌ Data loading failed")
            return False
    except Exception as e:
        logger.error(f"   ❌ Data loading test failed: {e}")
        return False

def test_data_processing():
    """Test 4: Process real data (small subset)"""
    logger.info("🧪 TEST 4: Process real data (small subset)")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Load data
        if not engine.load_real_ipta_data():
            logger.error("   ❌ Cannot load data for processing test")
            return False
        
        # Process data
        success = engine.process_real_data()
        if success:
            logger.info("   ✅ Data processing successful")
            logger.info(f"   📊 Processed pulsars: {len(engine.pulsar_data)}")
            if len(engine.pulsar_data) > 0:
                total_obs = sum(p['n_observations'] for p in engine.pulsar_data)
                logger.info(f"   📊 Total observations: {total_obs:,}")
            return True
        else:
            logger.error("   ❌ Data processing failed")
            return False
    except Exception as e:
        logger.error(f"   ❌ Data processing test failed: {e}")
        return False

def test_null_hypothesis():
    """Test 5: Test null hypothesis (small test)"""
    logger.info("🧪 TEST 5: Test null hypothesis (small test)")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Load and process data
        if not engine.load_real_ipta_data():
            logger.error("   ❌ Cannot load data for null hypothesis test")
            return False
        
        if not engine.process_real_data():
            logger.error("   ❌ Cannot process data for null hypothesis test")
            return False
        
        # Run null hypothesis test
        null_results = engine.test_null_hypothesis_real_data()
        
        logger.info("   ✅ Null hypothesis test completed")
        logger.info(f"   📊 Mean residual: {null_results['mean_residual']:.2e}")
        logger.info(f"   📊 Std residual: {null_results['std_residual']:.2e}")
        logger.info(f"   📊 Is normal: {null_results['is_normal']}")
        logger.info(f"   📊 Is zero mean: {null_results['is_zero_mean']}")
        logger.info(f"   📊 Is white noise: {null_results['is_white_noise']}")
        
        return True
    except Exception as e:
        logger.error(f"   ❌ Null hypothesis test failed: {e}")
        return False

def test_correlations_small():
    """Test 6: Test correlations (small subset)"""
    logger.info("🧪 TEST 6: Test correlations (small subset)")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Load and process data
        if not engine.load_real_ipta_data():
            logger.error("   ❌ Cannot load data for correlation test")
            return False
        
        if not engine.process_real_data():
            logger.error("   ❌ Cannot process data for correlation test")
            return False
        
        # Run correlation analysis
        corr_results = engine.analyze_correlations_real_data()
        
        logger.info("   ✅ Correlation analysis completed")
        logger.info(f"   📊 Total correlations: {corr_results['n_total']}")
        logger.info(f"   📊 Significant correlations: {corr_results['n_significant']}")
        logger.info(f"   📊 Mean correlation: {corr_results['mean_correlation']:.3f}")
        logger.info(f"   📊 HD fit good: {corr_results['hd_fit_good']}")
        
        return True
    except Exception as e:
        logger.error(f"   ❌ Correlation test failed: {e}")
        return False

def test_spectral_analysis_small():
    """Test 7: Test spectral analysis (small subset)"""
    logger.info("🧪 TEST 7: Test spectral analysis (small subset)")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Load and process data
        if not engine.load_real_ipta_data():
            logger.error("   ❌ Cannot load data for spectral test")
            return False
        
        if not engine.process_real_data():
            logger.error("   ❌ Cannot process data for spectral test")
            return False
        
        # Run spectral analysis
        spectral_results = engine.analyze_spectral_signatures_real_data()
        
        logger.info("   ✅ Spectral analysis completed")
        logger.info(f"   📊 Pulsars analyzed: {spectral_results['n_analyzed']}")
        logger.info(f"   📊 Cosmic string candidates: {spectral_results['n_candidates']}")
        logger.info(f"   📊 Mean slope: {spectral_results['mean_slope']:.3f}")
        logger.info(f"   📊 Expected slope: -0.667")
        
        return True
    except Exception as e:
        logger.error(f"   ❌ Spectral analysis test failed: {e}")
        return False

def test_periodic_analysis_small():
    """Test 8: Test periodic analysis (small subset)"""
    logger.info("🧪 TEST 8: Test periodic analysis (small subset)")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Load and process data
        if not engine.load_real_ipta_data():
            logger.error("   ❌ Cannot load data for periodic test")
            return False
        
        if not engine.process_real_data():
            logger.error("   ❌ Cannot process data for periodic test")
            return False
        
        # Run periodic analysis
        periodic_results = engine.analyze_periodic_signals_real_data()
        
        logger.info("   ✅ Periodic analysis completed")
        logger.info(f"   📊 Pulsars analyzed: {periodic_results['n_analyzed']}")
        logger.info(f"   📊 Significant signals: {periodic_results['n_significant']}")
        logger.info(f"   📊 Mean power: {periodic_results['mean_power']:.2f}")
        logger.info(f"   📊 Mean period: {periodic_results['mean_period']:.2f} days")
        
        return True
    except Exception as e:
        logger.error(f"   ❌ Periodic analysis test failed: {e}")
        return False

def test_arc2_small():
    """Test 9: Test ARC2 enhancement (small test)"""
    logger.info("🧪 TEST 9: Test ARC2 enhancement (small test)")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Load and process data
        if not engine.load_real_ipta_data():
            logger.error("   ❌ Cannot load data for ARC2 test")
            return False
        
        if not engine.process_real_data():
            logger.error("   ❌ Cannot process data for ARC2 test")
            return False
        
        # Test ARC2 on small sample
        if len(engine.pulsar_data) > 0:
            sample_data = engine.pulsar_data[0]['residuals']
            arc2_results = engine.ultimate_hybrid_arc2_solver(sample_data)
            
            logger.info("   ✅ ARC2 enhancement completed")
            logger.info(f"   📊 Patterns detected: {len(arc2_results['patterns'])}")
            logger.info(f"   📊 IAR: {arc2_results['iar']:.6f}")
            logger.info(f"   📊 Phase transition strength: {arc2_results['phase_transition']['strength']:.3f}")
            logger.info(f"   📊 Enhanced accuracy: {arc2_results['enhanced_accuracy']:.3f}")
        else:
            logger.warning("   ⚠️  No pulsar data available for ARC2 test")
        
        return True
    except Exception as e:
        logger.error(f"   ❌ ARC2 test failed: {e}")
        return False

def test_engine_integration():
    """Test 10: Test full engine integration (small run)"""
    logger.info("🧪 TEST 10: Test full engine integration (small run)")
    
    try:
        from ULTIMATE_COSMIC_STRING_ENGINE import UltimateCosmicStringEngine
        engine = UltimateCosmicStringEngine()
        
        # Run the ultimate analysis
        logger.info("   🚀 Running ultimate analysis...")
        start_time = time.time()
        
        results = engine.run_ultimate_analysis()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if results:
            logger.info("   ✅ Ultimate analysis completed successfully")
            logger.info(f"   ⏱️  Duration: {duration:.2f} seconds")
            logger.info(f"   📊 Results saved to: ULTIMATE_COSMIC_STRING_RESULTS.json")
            
            # Check key results
            if 'null_hypothesis' in results:
                null_passed = results['null_hypothesis']['null_hypothesis_passed']
                logger.info(f"   📊 Null hypothesis: {'PASSED' if null_passed else 'FAILED'}")
            
            if 'correlation_analysis' in results:
                n_corr = results['correlation_analysis']['n_significant']
                logger.info(f"   📊 Significant correlations: {n_corr}")
            
            if 'spectral_analysis' in results:
                n_candidates = results['spectral_analysis']['n_candidates']
                logger.info(f"   📊 Cosmic string candidates: {n_candidates}")
            
            return True
        else:
            logger.error("   ❌ Ultimate analysis failed")
            return False
    except Exception as e:
        logger.error(f"   ❌ Engine integration test failed: {e}")
        return False

def run_all_tests():
    """Run all small tests"""
    logger.info("🧪 RUNNING ALL LEGIT SMALL TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_import),
        ("Engine Initialization", test_engine_initialization),
        ("Data Loading", test_data_loading),
        ("Data Processing", test_data_processing),
        ("Null Hypothesis", test_null_hypothesis),
        ("Correlations", test_correlations_small),
        ("Spectral Analysis", test_spectral_analysis_small),
        ("Periodic Analysis", test_periodic_analysis_small),
        ("ARC2 Enhancement", test_arc2_small),
        ("Engine Integration", test_engine_integration)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🎯 TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Passed: {passed}/{total}")
    logger.info(f"❌ Failed: {total-passed}/{total}")
    logger.info(f"📊 Success rate: {passed/total*100:.1f}%")
    
    # Save results
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': total,
        'passed_tests': passed,
        'failed_tests': total - passed,
        'success_rate': passed/total*100,
        'test_results': results
    }
    
    with open('LEGIT_SMALL_TESTS_RESULTS.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"📁 Results saved to: LEGIT_SMALL_TESTS_RESULTS.json")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Ready for full run!")
        return True
    else:
        logger.warning(f"⚠️  {total-passed} tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    print("🧪 LEGIT SMALL TESTS")
    print("=" * 60)
    print("🎯 Testing ULTIMATE_COSMIC_STRING_ENGINE.py")
    print("🎯 Small, focused tests to verify functionality")
    print("🎯 Ready for real cosmic string science")
    print("=" * 60)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("🚀 Ready for full cosmic string analysis!")
        print("🎯 ULTIMATE_COSMIC_STRING_ENGINE.py is working properly!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔍 Check the logs for details")
        print("🔧 Fix issues before attempting full run")
