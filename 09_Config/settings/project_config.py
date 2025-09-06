#!/usr/bin/env python3
"""
Project Configuration Settings
Core ForensicSky V1 - Cosmic String Detection Engine
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CORE_ENGINE_DIR = PROJECT_ROOT / "01_Core_Engine"
DATA_DIR = PROJECT_ROOT / "02_Data"
ANALYSIS_DIR = PROJECT_ROOT / "03_Analysis"
RESULTS_DIR = PROJECT_ROOT / "04_Results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "05_Visualizations"
DOCUMENTATION_DIR = PROJECT_ROOT / "06_Documentation"
TESTS_DIR = PROJECT_ROOT / "07_Tests"
ARCHIVE_DIR = PROJECT_ROOT / "08_Archive"
CONFIG_DIR = PROJECT_ROOT / "09_Config"
NOTEBOOKS_DIR = PROJECT_ROOT / "10_Notebooks"

# Data paths
IPTA_DATA_DIR = DATA_DIR / "ipta_dr2"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Results paths
JSON_RESULTS_DIR = RESULTS_DIR / "json"
NPZ_RESULTS_DIR = RESULTS_DIR / "npz"
LOGS_DIR = RESULTS_DIR / "logs"

# Visualization paths
PLOTS_DIR = VISUALIZATIONS_DIR / "plots"
FIGURES_DIR = VISUALIZATIONS_DIR / "figures"

# Analysis paths
CORRELATION_DIR = ANALYSIS_DIR / "correlation"
SPECTRAL_DIR = ANALYSIS_DIR / "spectral"
ML_DIR = ANALYSIS_DIR / "ml"

# Test paths
UNIT_TESTS_DIR = TESTS_DIR / "unit"
INTEGRATION_TESTS_DIR = TESTS_DIR / "integration"

# Archive paths
OLD_ENGINES_DIR = ARCHIVE_DIR / "old_engines"
BACKUP_DIR = ARCHIVE_DIR / "backup"

# Configuration paths
SETTINGS_DIR = CONFIG_DIR / "settings"
PARAMETERS_DIR = CONFIG_DIR / "parameters"

# Core engine settings
CORE_ENGINE_FILE = CORE_ENGINE_DIR / "Core_ForensicSky_V1.py"
REQUIREMENTS_FILE = CONFIG_DIR / "requirements.txt"
SETUP_FILE = CONFIG_DIR / "setup.py"

# Data file patterns
IPTA_DATA_PATTERN = "ipta_dr2_version*_processed.npz"
COSMIC_STRING_INPUTS_PATTERN = "cosmic_string_inputs_version*.npz"

# Analysis parameters
DEFAULT_G_MU_RANGE = (1e-12, 1e-6)
DEFAULT_N_G_MU_VALUES = 100
DEFAULT_CORRELATION_THRESHOLD = 0.1
DEFAULT_SPECTRAL_SLOPE_TOLERANCE = 0.5
DEFAULT_FAP_THRESHOLD = 0.05

# GPU settings
GPU_AVAILABLE = True  # Will be detected at runtime
GPU_MEMORY_LIMIT = 8.6  # GB
CUDA_DEVICE = 0

# ML settings
ML_AVAILABLE = True  # Will be detected at runtime
TORCH_AVAILABLE = True  # Will be detected at runtime
SKLEARN_AVAILABLE = True  # Will be detected at runtime

# Visualization settings
DEFAULT_DPI = 300
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_COLORMAP = "viridis"
DEFAULT_STYLE = "seaborn-v0_8"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "cosmic_strings.log"

# Analysis settings
MIN_OBSERVATIONS = 50
OUTLIER_THRESHOLD = 5.0
BOOTSTRAP_SAMPLES = 1000
MONTE_CARLO_TRIALS = 1000

# Physical constants
G = 6.67430e-11  # m³/kg/s²
C = 2.99792458e8  # m/s
H0 = 2.2e-18  # 1/s (H0 = 70 km/s/Mpc)
RHO_C = 3 * H0**2 / (8 * 3.14159 * G)  # Critical density

# Cosmic string parameters
DEFAULT_GAMMA = 50  # Gravitational wave emission efficiency
DEFAULT_ALPHA = 0.1  # Loop formation size parameter
DEFAULT_BETA = 0.6  # Loop decay parameter
DEFAULT_XI = 0.3  # String network correlation length
DEFAULT_V = 0.6  # String network velocity

# Detection thresholds
DISCOVERY_THRESHOLD = 5.0  # 5σ for discovery
STRONG_EVIDENCE_THRESHOLD = 4.0  # 4σ for strong evidence
WEAK_EVIDENCE_THRESHOLD = 3.0  # 3σ for weak evidence
SUGGESTIVE_THRESHOLD = 2.0  # 2σ for suggestive

# File extensions
JSON_EXT = ".json"
NPZ_EXT = ".npz"
PNG_EXT = ".png"
PDF_EXT = ".pdf"
CSV_EXT = ".csv"
TXT_EXT = ".txt"
LOG_EXT = ".log"

# Version information
VERSION = "1.0.0"
BUILD_DATE = "2025-09-05"
AUTHOR = "Cosmic Strings Team"
DESCRIPTION = "Core ForensicSky V1 - Ultimate Cosmic String Detection Engine"

# Export all settings
__all__ = [
    # Paths
    "PROJECT_ROOT", "CORE_ENGINE_DIR", "DATA_DIR", "ANALYSIS_DIR", "RESULTS_DIR",
    "VISUALIZATIONS_DIR", "DOCUMENTATION_DIR", "TESTS_DIR", "ARCHIVE_DIR",
    "CONFIG_DIR", "NOTEBOOKS_DIR", "IPTA_DATA_DIR", "PROCESSED_DATA_DIR",
    "RAW_DATA_DIR", "JSON_RESULTS_DIR", "NPZ_RESULTS_DIR", "LOGS_DIR",
    "PLOTS_DIR", "FIGURES_DIR", "CORRELATION_DIR", "SPECTRAL_DIR", "ML_DIR",
    "UNIT_TESTS_DIR", "INTEGRATION_TESTS_DIR", "OLD_ENGINES_DIR", "BACKUP_DIR",
    "SETTINGS_DIR", "PARAMETERS_DIR",
    
    # Files
    "CORE_ENGINE_FILE", "REQUIREMENTS_FILE", "SETUP_FILE",
    
    # Patterns
    "IPTA_DATA_PATTERN", "COSMIC_STRING_INPUTS_PATTERN",
    
    # Parameters
    "DEFAULT_G_MU_RANGE", "DEFAULT_N_G_MU_VALUES", "DEFAULT_CORRELATION_THRESHOLD",
    "DEFAULT_SPECTRAL_SLOPE_TOLERANCE", "DEFAULT_FAP_THRESHOLD",
    
    # Settings
    "GPU_AVAILABLE", "GPU_MEMORY_LIMIT", "CUDA_DEVICE", "ML_AVAILABLE",
    "TORCH_AVAILABLE", "SKLEARN_AVAILABLE", "DEFAULT_DPI", "DEFAULT_FIGURE_SIZE",
    "DEFAULT_COLORMAP", "DEFAULT_STYLE", "LOG_LEVEL", "LOG_FORMAT", "LOG_FILE",
    "MIN_OBSERVATIONS", "OUTLIER_THRESHOLD", "BOOTSTRAP_SAMPLES", "MONTE_CARLO_TRIALS",
    
    # Constants
    "G", "C", "H0", "RHO_C", "DEFAULT_GAMMA", "DEFAULT_ALPHA", "DEFAULT_BETA",
    "DEFAULT_XI", "DEFAULT_V",
    
    # Thresholds
    "DISCOVERY_THRESHOLD", "STRONG_EVIDENCE_THRESHOLD", "WEAK_EVIDENCE_THRESHOLD",
    "SUGGESTIVE_THRESHOLD",
    
    # Extensions
    "JSON_EXT", "NPZ_EXT", "PNG_EXT", "PDF_EXT", "CSV_EXT", "TXT_EXT", "LOG_EXT",
    
    # Version
    "VERSION", "BUILD_DATE", "AUTHOR", "DESCRIPTION"
]
