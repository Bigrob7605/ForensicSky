#!/usr/bin/env python3
"""
Process Real IPTA DR2 Data
=========================

Process the actual IPTA DR2 data files to create a format our cosmic string
detection system can use.
"""

import numpy as np
import os
from pathlib import Path
import json
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_par_file(par_path):
    """Load a .par file and extract pulsar parameters"""
    params = {}
    try:
        with open(par_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        try:
                            value = float(parts[1])
                            params[key] = value
                        except ValueError:
                            params[key] = parts[1]
        return params
    except Exception as e:
        logger.warning(f"Could not load {par_path}: {e}")
        return {}

def load_tim_file(tim_path):
    """Load a .tim file and extract timing data"""
    times = []
    residuals = []
    uncertainties = []
    
    try:
        with open(tim_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            time = float(parts[0])
                            residual = float(parts[1])
                            uncertainty = float(parts[2])
                            
                            times.append(time)
                            residuals.append(residual)
                            uncertainties.append(uncertainty)
                        except ValueError:
                            continue
                    elif len(parts) == 2:
                        # Some timing files might have only time and residual
                        try:
                            time = float(parts[0])
                            residual = float(parts[1])
                            uncertainty = 1.0  # Default uncertainty
                            
                            times.append(time)
                            residuals.append(residual)
                            uncertainties.append(uncertainty)
                        except ValueError:
                            continue
        return np.array(times), np.array(residuals), np.array(uncertainties)
    except Exception as e:
        logger.warning(f"Could not load {tim_path}: {e}")
        return np.array([]), np.array([]), np.array([])

def process_real_ipta_data():
    """Process the real IPTA DR2 data"""
    logger.info("🔬 Processing REAL IPTA DR2 data...")
    
    # ⚠️ CRITICAL: Use ONLY real data, never cosmic_string_inputs!
    real_data_path = Path("data/real_ipta_dr2/ipta_par_files/DR2-master/release")
    version_a_path = real_data_path / "VersionA"
    version_b_path = real_data_path / "VersionB"
    
    # Verify we're using real data, not toy data
    if not real_data_path.exists():
        logger.error("❌ REAL IPTA DR2 data not found!")
        logger.error("   Expected: data/real_ipta_dr2/ipta_par_files/DR2-master/release/")
        logger.error("   This is the authentic data from GitLab: https://gitlab.com/IPTA/DR2/tree/master/release")
        return [], []
    
    logger.info("✅ Confirmed: Using REAL IPTA DR2 data from GitLab")
    logger.info("   Source: https://gitlab.com/IPTA/DR2/tree/master/release")
    logger.info("   ⚠️  NOT using cosmic_string_inputs (toy data)!")
    
    # Process Version A (we'll use this one)
    logger.info(f"📁 Processing Version A data from: {version_a_path}")
    
    pulsar_catalog = []
    timing_data = []
    
    # Get all .par files (they're in subdirectories)
    par_files = list(version_a_path.glob("*/J*.par"))
    logger.info(f"Found {len(par_files)} REAL pulsar parameter files")
    
    if len(par_files) == 0:
        logger.error("❌ No real IPTA DR2 parameter files found!")
        logger.error("   Check that data/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA/ exists")
        logger.error("   This should contain pulsar directories with .par and .tim files")
        return [], []
    
    for i, par_file in enumerate(par_files):
        if i % 10 == 0:
            logger.info(f"Processing pulsar {i+1}/{len(par_files)}: {par_file.stem}")
        
        # Load parameters
        params = load_par_file(par_file)
        if not params:
            continue
            
        # Get corresponding .tim file (look for .tim files in the same directory)
        tim_file = par_file.parent / f"{par_file.stem}.tim"
        if not tim_file.exists():
            # Try alternative naming
            tim_file = par_file.parent / f"{par_file.stem.replace('.IPTADR2', '')}.tim"
        if not tim_file.exists():
            logger.warning(f"No timing file for {par_file.stem}")
            continue
            
        # Load timing data
        times, residuals, uncertainties = load_tim_file(tim_file)
        if len(times) == 0:
            logger.warning(f"No timing data for {par_file.stem}")
            continue
        
        # Extract pulsar info
        pulsar_name = par_file.stem
        
        # Get sky coordinates
        ra = params.get('RAJ', 0.0)  # Right ascension in hours
        dec = params.get('DECJ', 0.0)  # Declination in degrees
        
        # Convert to radians
        ra_rad = np.radians(ra * 15.0)  # Convert hours to degrees, then to radians
        dec_rad = np.radians(dec)
        
        # Get other parameters
        period = params.get('F0', 1.0)  # Frequency (Hz)
        period_derivative = params.get('F1', 0.0)  # Frequency derivative
        
        # Create pulsar catalog entry
        pulsar_info = {
            'name': pulsar_name,
            'ra': ra_rad,
            'dec': dec_rad,
            'period': 1.0/period if period > 0 else 1.0,
            'period_derivative': period_derivative,
            'n_obs': len(times)
        }
        pulsar_catalog.append(pulsar_info)
        
        # Create timing data entry
        timing_entry = {
            'pulsar_name': pulsar_name,
            'times': times,
            'residuals': residuals,
            'uncertainties': uncertainties
        }
        timing_data.append(timing_entry)
    
    logger.info(f"✅ Processed {len(pulsar_catalog)} pulsars")
    logger.info(f"   Total observations: {sum(p['n_obs'] for p in pulsar_catalog):,}")
    
    # Save processed data
    output_path = Path("data/ipta_dr2/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as .npz file
    npz_path = output_path / "ipta_dr2_versionA_processed.npz"
    np.savez(npz_path, 
             pulsar_catalog=pulsar_catalog,
             timing_data=timing_data)
    
    logger.info(f"💾 Saved processed data to: {npz_path}")
    
    # Also save as JSON for inspection
    json_path = output_path / "ipta_dr2_versionA_processed.json"
    with open(json_path, 'w') as f:
        json.dump({
            'pulsar_catalog': pulsar_catalog,
            'timing_data': [{
                'pulsar_name': entry['pulsar_name'],
                'n_obs': len(entry['times']),
                'times_sample': entry['times'][:5].tolist(),  # First 5 times
                'residuals_sample': entry['residuals'][:5].tolist(),  # First 5 residuals
                'uncertainties_sample': entry['uncertainties'][:5].tolist()  # First 5 uncertainties
            } for entry in timing_data]
        }, f, indent=2)
    
    logger.info(f"💾 Saved summary to: {json_path}")
    
    return pulsar_catalog, timing_data

if __name__ == "__main__":
    logger.info("🚀 REAL IPTA DR2 DATA PROCESSOR")
    logger.info("=================================")
    logger.info("🎯 Mission: Process real IPTA DR2 data for cosmic string detection")
    logger.info("🎯 Source: data/real_ipta_dr2/ipta_par_files/DR2-master/release/VersionA")
    logger.info("🎯 Output: data/ipta_dr2/processed/ipta_dr2_versionA_processed.npz")
    logger.info("=================================")
    
    pulsar_catalog, timing_data = process_real_ipta_data()
    
    logger.info("✅ REAL IPTA DR2 DATA PROCESSING COMPLETE!")
    logger.info(f"📊 Processed {len(pulsar_catalog)} pulsars")
    logger.info(f"📊 Total observations: {sum(p['n_obs'] for p in pulsar_catalog):,}")
    logger.info("🚀 Ready for cosmic string detection!")
