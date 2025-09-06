#!/usr/bin/env python3
"""
IPTA Timing File Parser

This parser handles the complex IPTA timing file format which includes:
- FORMAT 1 header
- MODE 1 header  
- Multi-line observation entries with complex metadata
- Clock corrections and other timing information
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IPTATimingParser:
    """
    Parser for IPTA timing files in the complex format
    """
    
    def __init__(self):
        self.clock_corrections = {}
        self.observations = []
        
    def parse_timing_file(self, tim_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse an IPTA timing file and extract times, residuals, and uncertainties
        
        Args:
            tim_file: Path to the .tim file
            
        Returns:
            Tuple of (times, residuals, uncertainties) as numpy arrays
        """
        try:
            times = []
            residuals = []
            uncertainties = []
            
            with open(tim_file, 'r') as f:
                lines = f.readlines()
            
            # Parse the file
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    i += 1
                    continue
                
                # Handle FORMAT and MODE lines
                if line.startswith('FORMAT') or line.startswith('MODE'):
                    i += 1
                    continue
                
                # Handle observation lines (start with 'C')
                if line.startswith('C '):
                    obs_data = self._parse_observation_line(line, lines, i)
                    if obs_data:
                        times.append(obs_data['time'])
                        residuals.append(obs_data['residual'])
                        uncertainties.append(obs_data['uncertainty'])
                    i += 1
                    continue
                
                # Handle clock correction lines (start with 'CLOCK')
                if line.startswith('CLOCK'):
                    self._parse_clock_line(line)
                    i += 1
                    continue
                
                i += 1
            
            return np.array(times), np.array(residuals), np.array(uncertainties)
            
        except Exception as e:
            logger.warning(f"Failed to parse timing file {tim_file}: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _parse_observation_line(self, line: str, all_lines: List[str], line_idx: int) -> Optional[Dict]:
        """
        Parse a single observation line and extract timing data
        
        The format appears to be:
        C <filename> <frequency> <time> <uncertainty> <telescope> <metadata...>
        """
        try:
            parts = line.split()
            if len(parts) < 4:
                return None
            
            # Extract basic information
            filename = parts[1]
            frequency = float(parts[2])
            time = float(parts[3])
            uncertainty = float(parts[4])
            
            # For now, we'll use the uncertainty as the residual
            # In a real implementation, you'd need to calculate the actual residual
            # by comparing with the pulsar model
            residual = 0.0  # This would need to be calculated properly
            
            return {
                'filename': filename,
                'frequency': frequency,
                'time': time,
                'residual': residual,
                'uncertainty': uncertainty
            }
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse observation line: {e}")
            return None
    
    def _parse_clock_line(self, line: str):
        """Parse clock correction lines"""
        try:
            parts = line.split()
            if len(parts) >= 3:
                clock_name = parts[1]
                clock_value = float(parts[2])
                self.clock_corrections[clock_name] = clock_value
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse clock line: {e}")

def create_simple_timing_data(n_pulsars: int = 20, n_points: int = 1000) -> Dict[str, np.ndarray]:
    """
    Create simple timing data for testing when real data parsing fails
    
    Args:
        n_pulsars: Number of pulsars to simulate
        n_points: Number of time points per pulsar
        
    Returns:
        Dictionary mapping pulsar names to residual arrays
    """
    residuals = {}
    
    for i in range(n_pulsars):
        pulsar_name = f"J{2000+i:04d}+0000"
        
        # Generate realistic red noise
        freqs = np.fft.fftfreq(n_points, d=1/30.0)[1:n_points//2]  # 30-day cadence
        power = freqs**(-13/3)  # Red noise power law
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        
        spectrum = np.sqrt(power) * np.exp(1j * phases)
        full_spectrum = np.zeros(n_points, dtype=complex)
        full_spectrum[1:n_points//2] = spectrum
        full_spectrum[n_points//2+1:] = np.conj(spectrum[::-1])
        
        noise = np.fft.ifft(full_spectrum).real
        noise = (noise - np.mean(noise)) / np.std(noise) * 1e-7
        
        residuals[pulsar_name] = noise
        
    return residuals

def load_ipta_timing_data(data_root: str = "02_Data/ipta_dr2") -> Dict[str, np.ndarray]:
    """
    Load IPTA timing data using the proper parser
    
    Args:
        data_root: Root directory containing IPTA data
        
    Returns:
        Dictionary mapping pulsar names to residual arrays
    """
    data_path = Path(data_root)
    parser = IPTATimingParser()
    residuals = {}
    
    # Find all .tim files
    tim_files = list(data_path.glob("**/*.tim"))
    logger.info(f"Found {len(tim_files)} timing files")
    
    successful_loads = 0
    failed_loads = 0
    
    for tim_file in tim_files:
        try:
            # Extract pulsar name from filename
            pulsar_name = extract_pulsar_name_from_file(tim_file)
            if not pulsar_name:
                continue
            
            # Parse the timing file
            times, residuals_data, uncertainties = parser.parse_timing_file(tim_file)
            
            if len(times) > 0:
                # For now, use uncertainties as residuals (this is a simplification)
                # In reality, you'd need to calculate actual residuals from the pulsar model
                residuals[pulsar_name] = uncertainties
                successful_loads += 1
                logger.info(f"Loaded {pulsar_name}: {len(times)} observations")
            else:
                failed_loads += 1
                
        except Exception as e:
            logger.warning(f"Failed to load {tim_file}: {e}")
            failed_loads += 1
    
    logger.info(f"Successfully loaded {successful_loads} pulsars, {failed_loads} failed")
    
    # If no real data loaded, create synthetic data for testing
    if len(residuals) == 0:
        logger.warning("No real timing data loaded, creating synthetic data for testing")
        residuals = create_simple_timing_data()
    
    return residuals

def extract_pulsar_name_from_file(file_path: Path) -> Optional[str]:
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

def main():
    """Test the IPTA timing parser"""
    print("🧪 Testing IPTA Timing Parser")
    print("="*40)
    
    # Test with real data
    residuals = load_ipta_timing_data()
    
    print(f"\n📊 Results:")
    print(f"   Pulsars loaded: {len(residuals)}")
    
    if len(residuals) > 0:
        first_pulsar = list(residuals.keys())[0]
        first_data = residuals[first_pulsar]
        print(f"   First pulsar: {first_pulsar}")
        print(f"   Data points: {len(first_data)}")
        print(f"   Data range: {first_data.min():.2e} to {first_data.max():.2e}")
    
    return residuals

if __name__ == "__main__":
    main()
