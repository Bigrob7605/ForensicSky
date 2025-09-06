#!/usr/bin/env python3
"""
Setup script for Core ForensicSky V1
Cosmic String Detection Engine
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent.parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read version from config
version = "1.0.0"
build_date = "2025-09-05"
author = "Cosmic Strings Team"
description = "Core ForensicSky V1 - Ultimate Cosmic String Detection Engine"

setup(
    name="cosmic-strings",
    version=version,
    author=author,
    author_email="cosmic-strings@example.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cosmic-strings",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cosmic-strings/issues",
        "Source": "https://github.com/yourusername/cosmic-strings",
        "Documentation": "https://cosmic-strings.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
        ],
        "gpu": [
            "cupy-cuda11x>=10.0.0",
        ],
        "ml": [
            "transformers>=4.20.0",
            "timm>=0.6.0",
            "optuna>=3.0.0",
        ],
        "viz": [
            "bokeh>=2.4.0",
            "plotly>=5.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cosmic-strings=01_Core_Engine.Core_ForensicSky_V1:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cosmic-strings": [
            "data/*.npz",
            "data/*.json",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "cosmic-strings",
        "gravitational-waves",
        "pulsar-timing-array",
        "astronomy",
        "physics",
        "machine-learning",
        "gpu-acceleration",
        "data-analysis",
    ],
    platforms=["any"],
    license="MIT",
    build_date=build_date,
    # Custom metadata
    _custom_metadata={
        "core_engine": "Core_ForensicSky_V1.py",
        "advanced_technologies": 32,
        "analysis_steps": 27,
        "data_source": "IPTA DR2",
        "gpu_support": True,
        "ml_integration": True,
        "forensic_protection": True,
    },
)