#!/usr/bin/env python3
"""
Test Project Organization
Verify that all files are in their correct locations
"""

import pytest
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestProjectOrganization:
    """Test that the project is properly organized"""
    
    def test_project_structure_exists(self):
        """Test that all main directories exist"""
        required_dirs = [
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
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} does not exist"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    def test_core_engine_exists(self):
        """Test that the core engine exists"""
        core_engine_path = project_root / "01_Core_Engine" / "Core_ForensicSky_V1.py"
        assert core_engine_path.exists(), "Core_ForensicSky_V1.py does not exist"
        assert core_engine_path.is_file(), "Core_ForensicSky_V1.py is not a file"
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        config_files = [
            "09_Config/requirements.txt",
            "09_Config/setup.py",
            "09_Config/settings/project_config.py"
        ]
        
        for config_file in config_files:
            file_path = project_root / config_file
            assert file_path.exists(), f"Config file {config_file} does not exist"
            assert file_path.is_file(), f"{config_file} is not a file"
    
    def test_documentation_exists(self):
        """Test that documentation files exist"""
        doc_files = [
            "README.md",
            "PROJECT_STRUCTURE.md"
        ]
        
        for doc_file in doc_files:
            file_path = project_root / doc_file
            assert file_path.exists(), f"Documentation file {doc_file} does not exist"
            assert file_path.is_file(), f"{doc_file} is not a file"
    
    def test_data_directories_exist(self):
        """Test that data directories exist"""
        data_dirs = [
            "02_Data/ipta_dr2",
            "02_Data/processed",
            "02_Data/raw"
        ]
        
        for data_dir in data_dirs:
            dir_path = project_root / data_dir
            assert dir_path.exists(), f"Data directory {data_dir} does not exist"
            assert dir_path.is_dir(), f"{data_dir} is not a directory"
    
    def test_analysis_directories_exist(self):
        """Test that analysis directories exist"""
        analysis_dirs = [
            "03_Analysis/correlation",
            "03_Analysis/spectral",
            "03_Analysis/ml"
        ]
        
        for analysis_dir in analysis_dirs:
            dir_path = project_root / analysis_dir
            assert dir_path.exists(), f"Analysis directory {analysis_dir} does not exist"
            assert dir_path.is_dir(), f"{analysis_dir} is not a directory"
    
    def test_results_directories_exist(self):
        """Test that results directories exist"""
        results_dirs = [
            "04_Results/json",
            "04_Results/npz",
            "04_Results/logs"
        ]
        
        for results_dir in results_dirs:
            dir_path = project_root / results_dir
            assert dir_path.exists(), f"Results directory {results_dir} does not exist"
            assert dir_path.is_dir(), f"{results_dir} is not a directory"
    
    def test_visualization_directories_exist(self):
        """Test that visualization directories exist"""
        viz_dirs = [
            "05_Visualizations/plots",
            "05_Visualizations/figures"
        ]
        
        for viz_dir in viz_dirs:
            dir_path = project_root / viz_dir
            assert dir_path.exists(), f"Visualization directory {viz_dir} does not exist"
            assert dir_path.is_dir(), f"{viz_dir} is not a directory"
    
    def test_test_directories_exist(self):
        """Test that test directories exist"""
        test_dirs = [
            "07_Tests/unit",
            "07_Tests/integration"
        ]
        
        for test_dir in test_dirs:
            dir_path = project_root / test_dir
            assert dir_path.exists(), f"Test directory {test_dir} does not exist"
            assert dir_path.is_dir(), f"{test_dir} is not a directory"
    
    def test_archive_directories_exist(self):
        """Test that archive directories exist"""
        archive_dirs = [
            "08_Archive/old_engines",
            "08_Archive/backup"
        ]
        
        for archive_dir in archive_dirs:
            dir_path = project_root / archive_dir
            assert dir_path.exists(), f"Archive directory {archive_dir} does not exist"
            assert dir_path.is_dir(), f"{archive_dir} is not a directory"
    
    def test_config_directories_exist(self):
        """Test that config directories exist"""
        config_dirs = [
            "09_Config/settings",
            "09_Config/parameters"
        ]
        
        for config_dir in config_dirs:
            dir_path = project_root / config_dir
            assert dir_path.exists(), f"Config directory {config_dir} does not exist"
            assert dir_path.is_dir(), f"{config_dir} is not a directory"
    
    def test_core_engine_importable(self):
        """Test that the core engine can be imported"""
        try:
            from 01_Core_Engine.Core_ForensicSky_V1 import CoreForensicSkyV1
            assert CoreForensicSkyV1 is not None
        except ImportError as e:
            pytest.fail(f"Failed to import CoreForensicSkyV1: {e}")
    
    def test_project_config_importable(self):
        """Test that the project config can be imported"""
        try:
            from 09_Config.settings.project_config import PROJECT_ROOT
            assert PROJECT_ROOT is not None
        except ImportError as e:
            pytest.fail(f"Failed to import project config: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
