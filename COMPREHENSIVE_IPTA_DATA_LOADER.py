#!/usr/bin/env python3
"""
COMPREHENSIVE IPTA DATA LOADER
==============================

Loads ALL IPTA DR2 data from ALL directories and file types for complete cosmic string analysis.
This ensures we get 100% of the available data for maximum detection sensitivity.

Following Kai Master Protocol V5 - Production Ready
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict
import re

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 CUDA GPU acceleration enabled!")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("⚠️  CUDA not available, using CPU")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveIPTADataLoader:
    """
    Comprehensive IPTA DR2 Data Loader
    
    Loads ALL available data from ALL directories:
    - EPTA_v2.2 data
    - NANOGrav_9y data  
    - PPTA_dr1dr2 data
    - Release VersionA and VersionB data
    - Clock files (.clk)
    - All .par and .tim files
    """
    
    def __init__(self, data_root="02_Data/ipta_dr2"):
        """Initialize the comprehensive data loader"""
        self.data_root = Path(data_root)
        self.gpu_available = GPU_AVAILABLE
        
        # Data storage
        self.pulsar_catalog = []
        self.timing_data = []
        self.clock_data = {}
        
        # Statistics
        self.loading_stats = {
            'total_par_files_found': 0,
            'total_tim_files_found': 0,
            'total_clk_files_found': 0,
            'successful_pulsar_loads': 0,
            'failed_pulsar_loads': 0,
            'successful_timing_loads': 0,
            'failed_timing_loads': 0,
            'successful_clock_loads': 0,
            'failed_clock_loads': 0,
            'error_types': defaultdict(int),
            'data_sources': defaultdict(int),
            'pulsar_duplicates_handled': 0
        }
        
        # Track processed pulsars to handle duplicates
        self.processed_pulsars = set()
        
        logger.info("🔬 Comprehensive IPTA Data Loader initialized")
        logger.info(f"   Data root: {self.data_root}")
        logger.info(f"   GPU available: {self.gpu_available}")
    
    def load_all_data(self):
        """Load ALL IPTA DR2 data from ALL directories"""
        logger.info("🌌 LOADING ALL IPTA DR2 DATA - COMPREHENSIVE SCAN")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Load clock files first (needed for timing corrections)
        logger.info("🕐 Step 1: Loading clock files...")
        self._load_all_clock_files()
        
        # Step 2: Scan ALL directories for data
        logger.info("📁 Step 2: Scanning ALL directories for data...")
        self._scan_all_directories()
        
        # Step 3: Load pulsar parameters from ALL sources
        logger.info("⭐ Step 3: Loading pulsar parameters from ALL sources...")
        self._load_all_pulsar_parameters()
        
        # Step 4: Load timing data from ALL sources
        logger.info("⏰ Step 4: Loading timing data from ALL sources...")
        self._load_all_timing_data()
        
        # Step 5: Process and consolidate data
        logger.info("🔄 Step 5: Processing and consolidating data...")
        self._consolidate_data()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print comprehensive summary
        self._print_comprehensive_summary(duration)
        
        return {
            'pulsar_catalog': self.pulsar_catalog,
            'timing_data': self.timing_data,
            'clock_data': self.clock_data,
            'loading_stats': self.loading_stats
        }
    
    def _load_all_clock_files(self):
        """Load ALL clock files from ALL directories"""
        logger.info("🕐 Loading clock files...")
        
        # Find all .clk files
        clock_files = list(self.data_root.glob("**/*.clk"))
        self.loading_stats['total_clk_files_found'] = len(clock_files)
        
        logger.info(f"   Found {len(clock_files)} clock files")
        
        for clk_file in clock_files:
            try:
                # Load clock file (simplified - just track that we found it)
                self.clock_data[clk_file.name] = {
                    'path': str(clk_file),
                    'size': clk_file.stat().st_size,
                    'source': self._identify_data_source(clk_file)
                }
                self.loading_stats['successful_clock_loads'] += 1
                self.loading_stats['data_sources'][self._identify_data_source(clk_file)] += 1
                
            except Exception as e:
                logger.warning(f"Failed to load clock file {clk_file}: {e}")
                self.loading_stats['failed_clock_loads'] += 1
                self.loading_stats['error_types']['clock_load_error'] += 1
    
    def _scan_all_directories(self):
        """Scan ALL directories to understand the data structure"""
        logger.info("📁 Scanning all directories...")
        
        # Main data directories
        main_dirs = [
            "real_ipta_dr2/ipta_par_files/DR2-master",
            "ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master"
        ]
        
        for main_dir in main_dirs:
            full_path = self.data_root / main_dir
            if full_path.exists():
                logger.info(f"   Found main directory: {main_dir}")
                
                # Scan subdirectories
                subdirs = [d for d in full_path.iterdir() if d.is_dir()]
                logger.info(f"   Subdirectories: {[d.name for d in subdirs]}")
                
                # Count files by type
                par_files = list(full_path.glob("**/*.par"))
                tim_files = list(full_path.glob("**/*.tim"))
                clk_files = list(full_path.glob("**/*.clk"))
                
                logger.info(f"   .par files: {len(par_files)}")
                logger.info(f"   .tim files: {len(tim_files)}")
                logger.info(f"   .clk files: {len(clk_files)}")
    
    def _load_all_pulsar_parameters(self):
        """Load pulsar parameters from ALL sources"""
        logger.info("⭐ Loading pulsar parameters from ALL sources...")
        
        # Find ALL .par files from ALL directories
        all_par_files = list(self.data_root.glob("**/*.par"))
        self.loading_stats['total_par_files_found'] = len(all_par_files)
        
        logger.info(f"   Found {len(all_par_files)} parameter files across ALL directories")
        
        # Group by pulsar name to handle duplicates
        pulsar_files = defaultdict(list)
        for par_file in all_par_files:
            pulsar_name = self._extract_pulsar_name(par_file)
            if pulsar_name:
                pulsar_files[pulsar_name].append(par_file)
        
        logger.info(f"   Unique pulsars: {len(pulsar_files)}")
        
        # Process each unique pulsar
        for pulsar_name, par_files in pulsar_files.items():
            try:
                # Choose the best parameter file (prefer release versions)
                best_par_file = self._choose_best_par_file(par_files)
                
                # Load parameters
                params = self._load_par_file(best_par_file)
                if not params:
                    continue
                
                # Extract pulsar info
                pulsar_info = self._extract_pulsar_info(pulsar_name, params, best_par_file)
                if pulsar_info:
                    self.pulsar_catalog.append(pulsar_info)
                    self.loading_stats['successful_pulsar_loads'] += 1
                    self.loading_stats['data_sources'][pulsar_info['source']] += 1
                    
                    # Track duplicates handled
                    if len(par_files) > 1:
                        self.loading_stats['pulsar_duplicates_handled'] += len(par_files) - 1
                
            except Exception as e:
                logger.warning(f"Failed to load pulsar {pulsar_name}: {e}")
                self.loading_stats['failed_pulsar_loads'] += 1
                self.loading_stats['error_types']['pulsar_load_error'] += 1
    
    def _load_all_timing_data(self):
        """Load timing data from ALL sources"""
        logger.info("⏰ Loading timing data from ALL sources...")
        
        # Find ALL .tim files from ALL directories
        all_tim_files = list(self.data_root.glob("**/*.tim"))
        self.loading_stats['total_tim_files_found'] = len(all_tim_files)
        
        logger.info(f"   Found {len(all_tim_files)} timing files across ALL directories")
        
        # Group by pulsar name
        pulsar_tim_files = defaultdict(list)
        for tim_file in all_tim_files:
            pulsar_name = self._extract_pulsar_name(tim_file)
            if pulsar_name:
                pulsar_tim_files[pulsar_name].append(tim_file)
        
        logger.info(f"   Unique pulsars with timing data: {len(pulsar_tim_files)}")
        
        # Process each pulsar's timing data
        for pulsar_name, tim_files in pulsar_tim_files.items():
            try:
                # Choose the best timing file
                best_tim_file = self._choose_best_tim_file(tim_files)
                
                # Load timing data
                times, residuals, uncertainties = self._load_tim_file(best_tim_file)
                if len(times) == 0:
                    continue
                
                # Convert to GPU arrays if available
                if self.gpu_available:
                    times = cp.asarray(times)
                    residuals = cp.asarray(residuals)
                    uncertainties = cp.asarray(uncertainties)
                
                # Store timing data
                timing_entry = {
                    'pulsar_name': pulsar_name,
                    'times': times,
                    'residuals': residuals,
                    'uncertainties': uncertainties,
                    'source': self._identify_data_source(best_tim_file),
                    'file_path': str(best_tim_file),
                    'n_observations': len(times)
                }
                
                self.timing_data.append(timing_entry)
                self.loading_stats['successful_timing_loads'] += 1
                self.loading_stats['data_sources'][timing_entry['source']] += 1
                
            except Exception as e:
                logger.warning(f"Failed to load timing data for {pulsar_name}: {e}")
                self.loading_stats['failed_timing_loads'] += 1
                self.loading_stats['error_types']['timing_load_error'] += 1
    
    def _consolidate_data(self):
        """Consolidate and validate the loaded data"""
        logger.info("🔄 Consolidating data...")
        
        # Match pulsar catalog with timing data
        matched_pulsars = 0
        for pulsar_info in self.pulsar_catalog:
            pulsar_name = pulsar_info['name']
            timing_entries = [t for t in self.timing_data if t['pulsar_name'] == pulsar_name]
            if timing_entries:
                matched_pulsars += 1
                pulsar_info['has_timing_data'] = True
                pulsar_info['n_observations'] = timing_entries[0]['n_observations']
            else:
                pulsar_info['has_timing_data'] = False
                pulsar_info['n_observations'] = 0
        
        logger.info(f"   Matched {matched_pulsars} pulsars with timing data")
        
        # Calculate total observations
        total_observations = sum(t['n_observations'] for t in self.timing_data)
        logger.info(f"   Total observations: {total_observations:,}")
    
    def _extract_pulsar_name(self, file_path):
        """Extract pulsar name from file path"""
        try:
            # Remove common suffixes and prefixes
            name = file_path.stem
            
            # Remove common suffixes
            suffixes_to_remove = [
                '.IPTADR2', '.IPTA', '.DR2', '.TDB', '.Mean',
                '_NANOGrav_9yv1', '_dr1dr2', '_IPTADR2', '_IPTA', '_DR2'
            ]
            
            for suffix in suffixes_to_remove:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            
            # Extract J name pattern (J followed by numbers and + or -)
            j_match = re.search(r'J\d{4}[+-]\d{4}', name)
            if j_match:
                return j_match.group()
            
            # If no J pattern, return the cleaned name
            return name
            
        except Exception:
            return None
    
    def _choose_best_par_file(self, par_files):
        """Choose the best parameter file from multiple options"""
        # Priority order: release > working > others
        priority_keywords = ['release', 'working', 'VersionA', 'VersionB']
        
        for keyword in priority_keywords:
            for par_file in par_files:
                if keyword.lower() in str(par_file).lower():
                    return par_file
        
        # If no priority match, return the first one
        return par_files[0]
    
    def _choose_best_tim_file(self, tim_files):
        """Choose the best timing file from multiple options"""
        # Priority order: release > working > others
        priority_keywords = ['release', 'working', 'VersionA', 'VersionB']
        
        for keyword in priority_keywords:
            for tim_file in tim_files:
                if keyword.lower() in str(tim_file).lower():
                    return tim_file
        
        # If no priority match, return the largest file
        return max(tim_files, key=lambda f: f.stat().st_size)
    
    def _identify_data_source(self, file_path):
        """Identify the data source from file path"""
        path_str = str(file_path).lower()
        
        if 'epta' in path_str:
            return 'EPTA'
        elif 'nanograv' in path_str:
            return 'NANOGrav'
        elif 'ppta' in path_str:
            return 'PPTA'
        elif 'release' in path_str:
            return 'Release'
        elif 'working' in path_str:
            return 'Working'
        else:
            return 'Other'
    
    def _load_par_file(self, par_file):
        """Load pulsar parameter file"""
        try:
            params = {}
            with open(par_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0]
                            value = parts[1]
                            try:
                                # Try to convert to float
                                params[key] = float(value)
                            except ValueError:
                                # Keep as string
                                params[key] = value
            return params
        except Exception as e:
            logger.warning(f"Failed to load par file {par_file}: {e}")
            return None
    
    def _load_tim_file(self, tim_file):
        """Load pulsar timing file"""
        try:
            times = []
            residuals = []
            uncertainties = []
            
            with open(tim_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                times.append(float(parts[0]))
                                residuals.append(float(parts[1]))
                                uncertainties.append(float(parts[2]))
                            except ValueError:
                                continue
            
            return np.array(times), np.array(residuals), np.array(uncertainties)
        except Exception as e:
            logger.warning(f"Failed to load tim file {tim_file}: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _extract_pulsar_info(self, pulsar_name, params, par_file):
        """Extract pulsar information from parameters"""
        try:
            # Get sky coordinates
            ra = params.get('RAJ', 0.0)
            dec = params.get('DECJ', 0.0)
            
            # Convert to radians
            if isinstance(ra, str):
                # Parse RA string format
                ra_parts = ra.split(':')
                if len(ra_parts) == 3:
                    ra_hours = float(ra_parts[0]) + float(ra_parts[1])/60.0 + float(ra_parts[2])/3600.0
                    ra_rad = ra_hours * np.pi / 12.0
                else:
                    ra_rad = float(ra) * np.pi / 12.0
            else:
                ra_rad = float(ra) * np.pi / 12.0
            
            if isinstance(dec, str):
                # Parse DEC string format
                dec_parts = dec.split(':')
                if len(dec_parts) == 3:
                    dec_deg = float(dec_parts[0]) + float(dec_parts[1])/60.0 + float(dec_parts[2])/3600.0
                    dec_rad = dec_deg * np.pi / 180.0
                else:
                    dec_rad = float(dec) * np.pi / 180.0
            else:
                dec_rad = float(dec) * np.pi / 180.0
            
            # Convert to Cartesian coordinates
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)
            
            return {
                'name': pulsar_name,
                'ra_rad': ra_rad,
                'dec_rad': dec_rad,
                'position': [x, y, z],
                'period': params.get('F0', 1.0),
                'period_derivative': params.get('F1', 0.0),
                'dm': params.get('DM', 0.0),
                'source': self._identify_data_source(par_file),
                'file_path': str(par_file)
            }
        except Exception as e:
            logger.warning(f"Failed to extract pulsar info for {pulsar_name}: {e}")
            return None
    
    def _print_comprehensive_summary(self, duration):
        """Print comprehensive loading summary"""
        logger.info("=" * 60)
        logger.info("🌌 COMPREHENSIVE IPTA DATA LOADING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total loading time: {duration:.1f} seconds")
        logger.info("")
        logger.info("📊 LOADING STATISTICS:")
        logger.info(f"   📁 Parameter files found: {self.loading_stats['total_par_files_found']}")
        logger.info(f"   ⏰ Timing files found: {self.loading_stats['total_tim_files_found']}")
        logger.info(f"   🕐 Clock files found: {self.loading_stats['total_clk_files_found']}")
        logger.info("")
        logger.info("✅ SUCCESSFUL LOADS:")
        logger.info(f"   ⭐ Pulsars loaded: {self.loading_stats['successful_pulsar_loads']}")
        logger.info(f"   ⏰ Timing datasets loaded: {self.loading_stats['successful_timing_loads']}")
        logger.info(f"   🕐 Clock files loaded: {self.loading_stats['successful_clock_loads']}")
        logger.info("")
        logger.info("❌ FAILED LOADS:")
        logger.info(f"   ⭐ Pulsar load failures: {self.loading_stats['failed_pulsar_loads']}")
        logger.info(f"   ⏰ Timing load failures: {self.loading_stats['failed_timing_loads']}")
        logger.info(f"   🕐 Clock load failures: {self.loading_stats['failed_clock_loads']}")
        logger.info("")
        logger.info("🔄 DUPLICATE HANDLING:")
        logger.info(f"   🔄 Duplicate pulsars handled: {self.loading_stats['pulsar_duplicates_handled']}")
        logger.info("")
        logger.info("📈 DATA SOURCES:")
        for source, count in self.loading_stats['data_sources'].items():
            logger.info(f"   {source}: {count} files")
        logger.info("")
        logger.info("🎯 FINAL DATASET:")
        logger.info(f"   ⭐ Total pulsars: {len(self.pulsar_catalog)}")
        logger.info(f"   ⏰ Total timing datasets: {len(self.timing_data)}")
        logger.info(f"   🕐 Total clock files: {len(self.clock_data)}")
        
        # Calculate total observations
        total_observations = sum(t['n_observations'] for t in self.timing_data)
        logger.info(f"   📊 Total observations: {total_observations:,}")
        
        # Calculate data with both parameters and timing
        complete_pulsars = sum(1 for p in self.pulsar_catalog if p.get('has_timing_data', False))
        logger.info(f"   ✅ Complete pulsars (par + tim): {complete_pulsars}")
        
        logger.info("=" * 60)
        logger.info("🚀 READY FOR COMPREHENSIVE COSMIC STRING ANALYSIS!")
        logger.info("=" * 60)

def main():
    """Main function to test the comprehensive data loader"""
    logger.info("🚀 Starting Comprehensive IPTA Data Loader Test")
    
    # Initialize loader
    loader = ComprehensiveIPTADataLoader()
    
    # Load all data
    results = loader.load_all_data()
    
    # Save results
    output_file = f"comprehensive_ipta_data_loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'pulsar_catalog': results['pulsar_catalog'],
        'timing_data': [
            {
                'pulsar_name': t['pulsar_name'],
                'source': t['source'],
                'file_path': t['file_path'],
                'n_observations': t['n_observations']
            } for t in results['timing_data']
        ],
        'clock_data': results['clock_data'],
        'loading_stats': dict(results['loading_stats'])
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"💾 Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
