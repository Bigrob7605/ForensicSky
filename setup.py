#!/usr/bin/env python3
"""
Setup script for Cosmic Strings Simulation & Detection Toolkit

This package provides tools for simulating cosmic string networks,
computing their observable signatures, and prototyping detection methods.

Author: Cosmic Strings Research Team
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cosmic-strings-toolkit",
    version="0.1.0",
    author="Cosmic Strings Research Team",
    author_email="research@cosmicstrings.org",
    description="A comprehensive toolkit for cosmic string simulation and detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Bigrob7605/The-Universal-Open-Science-Toolbox",
    project_urls={
        "Bug Reports": "https://github.com/Bigrob7605/The-Universal-Open-Science-Toolbox/issues",
        "Source": "https://github.com/Bigrob7605/The-Universal-Open-Science-Toolbox",
        "Documentation": "https://cosmic-strings-toolkit.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "gpu": [
            "cupy-cuda11x>=10.0.0",
            "cupy-cuda12x>=12.0.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cosmic-strings-sim=simulations.network_evolution:main",
            "cosmic-strings-gw=analysis.gravitational_waves:main",
            "cosmic-strings-frb=detection.frb_lensing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords="cosmic-strings, cosmology, gravitational-waves, astrophysics, physics",
)
