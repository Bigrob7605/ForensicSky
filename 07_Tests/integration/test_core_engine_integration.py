#!/usr/bin/env python3
"""
Integration Test for Core ForensicSky V1
Test the core engine functionality
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestCoreEngineIntegration:
    """Integration tests for Core ForensicSky V1"""
    
    def test_core_engine_initialization(self):
        """Test that the core engine can be initialized"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            assert engine is not None
            assert hasattr(engine, 'run_complete_analysis')
            assert hasattr(engine, 'gpu_available')
            assert hasattr(engine, 'pulsar_catalog')
        except Exception as e:
            pytest.fail(f"Failed to initialize CoreForensicSkyV1: {e}")
    
    def test_advanced_technologies_available(self):
        """Test that all advanced technologies are available"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Check for key advanced technology components
            advanced_components = [
                'ultimate_visualization',
                'enhanced_gpu_pta',
                'world_shattering_pta',
                'cosmic_string_gold',
                'perfect_detector'
            ]
            
            for component in advanced_components:
                assert hasattr(engine, component), f"Advanced component {component} not found"
                
        except Exception as e:
            pytest.fail(f"Failed to check advanced technologies: {e}")
    
    def test_analysis_methods_available(self):
        """Test that all analysis methods are available"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Check for key analysis methods
            analysis_methods = [
                'run_ultimate_visualization_analysis',
                'run_perfect_detector_analysis',
                'correlation_analysis',
                'spectral_analysis',
                'ml_analysis'
            ]
            
            for method in analysis_methods:
                assert hasattr(engine, method), f"Analysis method {method} not found"
                assert callable(getattr(engine, method)), f"Analysis method {method} is not callable"
                
        except Exception as e:
            pytest.fail(f"Failed to check analysis methods: {e}")
    
    def test_data_loading_capability(self):
        """Test that the engine can handle data loading"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Test data loading method exists
            assert hasattr(engine, 'load_real_ipta_data'), "Data loading method not found"
            assert callable(engine.load_real_ipta_data), "Data loading method is not callable"
            
        except Exception as e:
            pytest.fail(f"Failed to check data loading capability: {e}")
    
    def test_gpu_detection(self):
        """Test GPU detection capability"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Check GPU availability detection
            assert hasattr(engine, 'gpu_available'), "GPU availability detection not found"
            assert isinstance(engine.gpu_available, bool), "GPU availability should be boolean"
            
        except Exception as e:
            pytest.fail(f"Failed to check GPU detection: {e}")
    
    def test_ml_integration(self):
        """Test machine learning integration"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Check ML components
            ml_components = [
                'ml_noise',
                'neural_detector',
                'bayesian_analyzer'
            ]
            
            for component in ml_components:
                assert hasattr(engine, component), f"ML component {component} not found"
                
        except Exception as e:
            pytest.fail(f"Failed to check ML integration: {e}")
    
    def test_visualization_capability(self):
        """Test visualization capability"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Check visualization components
            assert hasattr(engine, 'ultimate_visualization'), "Ultimate visualization not found"
            assert hasattr(engine.ultimate_visualization, 'create_correlation_network_plot'), "Correlation network plot method not found"
            assert hasattr(engine.ultimate_visualization, 'create_spectral_signature_plot'), "Spectral signature plot method not found"
            
        except Exception as e:
            pytest.fail(f"Failed to check visualization capability: {e}")
    
    def test_forensic_protection(self):
        """Test forensic protection capability"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Check forensic protection methods
            assert hasattr(engine, 'forensic_disproof_analysis'), "Forensic disproof analysis not found"
            assert callable(engine.forensic_disproof_analysis), "Forensic disproof analysis is not callable"
            
        except Exception as e:
            pytest.fail(f"Failed to check forensic protection: {e}")
    
    def test_engine_attributes(self):
        """Test that the engine has all required attributes"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            engine = CoreForensicSkyV1()
            
            # Check required attributes
            required_attributes = [
                'data_path',
                'pulsar_data',
                'timing_data',
                'pulsar_catalog',
                'gpu_available',
                'healpy_available',
                'G',
                'c',
                'H0',
                'Gmu_range',
                'string_spectral_index',
                'expected_limit',
                'correlation_threshold',
                'spectral_slope_tolerance',
                'periodic_power_threshold',
                'fap_threshold',
                'results',
                'forensic_report'
            ]
            
            for attr in required_attributes:
                assert hasattr(engine, attr), f"Required attribute {attr} not found"
                
        except Exception as e:
            pytest.fail(f"Failed to check engine attributes: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
