#!/usr/bin/env python3
"""
TEST CORE FORENSIC SKY V1
=========================

Test script for the consolidated Core ForensicSky V1 engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_ForensicSky_V1 import CoreForensicSkyV1
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_engine():
    """Test the Core ForensicSky V1 engine"""
    logger.info("🧪 TESTING CORE FORENSIC SKY V1 ENGINE")
    
    # Initialize engine
    engine = CoreForensicSkyV1()
    
    # Run complete analysis
    results = engine.run_complete_analysis()
    
    # Run comprehensive tests
    test_results = engine.run_comprehensive_tests()
    
    if results:
        logger.info("✅ TEST PASSED - Core engine working!")
        logger.info(f"   Final Verdict: {results['final_verdict']}")
        logger.info(f"   Pulsars Loaded: {results['loading_stats']['successful_loads']}")
        logger.info(f"   GPU Available: {engine.gpu_available}")
        logger.info(f"   Healpy Available: {engine.healpy_available}")
        
        # Check test results
        for test_name, test_result in test_results.items():
            status = test_result.get('status', 'UNKNOWN')
            logger.info(f"   {test_name}: {status}")
        
        return True
    else:
        logger.error("❌ TEST FAILED - Core engine not working!")
        return False

if __name__ == "__main__":
    success = test_core_engine()
    sys.exit(0 if success else 1)
