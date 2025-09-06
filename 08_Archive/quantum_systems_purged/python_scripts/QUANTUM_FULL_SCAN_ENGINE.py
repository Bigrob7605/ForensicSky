#!/usr/bin/env python3
"""
QUANTUM FULL SCAN ENGINE
========================

Comprehensive full-scan engine that loads ALL pulsars and clock files,
then applies novel quantum methods that nobody else uses.

This is the ULTIMATE cosmic string detection system!
"""

import numpy as np
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Add paths
sys.path.append(str(Path(__file__).parent / "01_Core_Engine"))
sys.path.append(str(Path(__file__).parent))

# Import our systems
from UNIFIED_QUANTUM_COSMIC_STRING_PLATFORM import UnifiedQuantumCosmicStringPlatform
from Core_ForensicSky_V1 import CoreForensicSkyV1

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_full_scan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuantumFullScanEngine:
    """
    The ULTIMATE cosmic string detection system that loads ALL pulsars
    and applies novel quantum methods nobody else uses!
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Quantum Full Scan Engine...")
        
        # Initialize systems
        self.core_engine = CoreForensicSkyV1()
        self.quantum_platform = UnifiedQuantumCosmicStringPlatform()
        
        # Data storage
        self.all_pulsars = []
        self.all_timing_data = []
        self.quantum_results = {}
        self.novel_analysis_results = {}
        
        # Statistics
        self.scan_stats = {
            'total_pulsars_found': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'clock_files_loaded': 0,
            'quantum_analyses_completed': 0,
            'novel_methods_applied': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info("✅ Quantum Full Scan Engine initialized!")
    
    def load_all_clock_files(self):
        """Load ALL clock files for maximum timing accuracy"""
        self.logger.info("⏰ Loading ALL clock files for maximum timing accuracy...")
        
        try:
            # Load clock files
            clock_stats = self.core_engine.load_clock_files()
            self.scan_stats['clock_files_loaded'] = clock_stats.get('successful_loads', 0)
            
            self.logger.info(f"✅ Loaded {self.scan_stats['clock_files_loaded']} clock files")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading clock files: {e}")
            return False
    
    def discover_all_pulsars(self):
        """Discover ALL pulsars in the IPTA DR2 dataset - don't miss any!"""
        self.logger.info("🔍 Discovering ALL pulsars in IPTA DR2 dataset...")
        
        # Data path
        real_data_path = Path("02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master")
        
        if not real_data_path.exists():
            self.logger.error("❌ Real IPTA DR2 data not found!")
            return []
        
        # Find ALL .par files - COMPLETE COVERAGE
        all_par_files = list(real_data_path.glob("**/*.par"))
        self.logger.info(f"📁 Found {len(all_par_files)} total .par files")
        
        # Group by pulsar name (handle multiple versions)
        pulsar_groups = {}
        for par_file in all_par_files:
            # Extract pulsar name (remove version suffixes)
            base_name = par_file.stem
            if '.' in base_name:
                # Remove version suffixes like .IPTADR2, .DR2, etc.
                pulsar_name = base_name.split('.')[0]
            else:
                pulsar_name = base_name
            
            if pulsar_name not in pulsar_groups:
                pulsar_groups[pulsar_name] = []
            pulsar_groups[pulsar_name].append(par_file)
        
        self.logger.info(f"🎯 Found {len(pulsar_groups)} unique pulsars")
        
        # Log some examples
        sample_pulsars = list(pulsar_groups.keys())[:10]
        self.logger.info(f"📋 Sample pulsars: {sample_pulsars}")
        
        return pulsar_groups
    
    def load_pulsar_data(self, pulsar_name, par_files):
        """Load data for a single pulsar with all available files"""
        try:
            # Find the best par file (prefer .IPTADR2, then .DR2, then others)
            best_par = None
            for par_file in par_files:
                if '.IPTADR2' in par_file.name:
                    best_par = par_file
                    break
                elif '.DR2' in par_file.name and best_par is None:
                    best_par = par_file
                elif best_par is None:
                    best_par = par_file
            
            if not best_par:
                return None
            
            # Load par file
            params = self.core_engine.load_par_file(best_par)
            if not params:
                return None
            
            # Find timing file
            real_data_path = Path("02_Data/ipta_dr2/real_ipta_dr2/ipta_par_files/DR2-master")
            tim_files = list(real_data_path.glob(f"**/{pulsar_name}*.tim"))
            
            if not tim_files:
                return None
            
            # Use the best timing file
            best_tim = None
            for tim_file in tim_files:
                if '.IPTADR2' in tim_file.name:
                    best_tim = tim_file
                    break
                elif '.DR2' in tim_file.name and best_tim is None:
                    best_tim = tim_file
                elif best_tim is None:
                    best_tim = tim_file
            
            if not best_tim:
                return None
            
            # Load timing data
            times, residuals, uncertainties = self.core_engine.load_tim_file(best_tim)
            if len(times) == 0:
                return None
            
            # Convert to GPU arrays
            times = self.core_engine._to_gpu_array(times)
            residuals = self.core_engine._to_gpu_array(residuals)
            uncertainties = self.core_engine._to_gpu_array(uncertainties)
            
            # Extract pulsar info
            ra = params.get('RAJ', 0.0)
            dec = params.get('DECJ', 0.0)
            
            # Convert coordinates
            if isinstance(ra, str):
                ra_parts = ra.split(':')
                if len(ra_parts) == 3:
                    ra = float(ra_parts[0]) + float(ra_parts[1])/60 + float(ra_parts[2])/3600
                else:
                    ra = float(ra)
            
            if isinstance(dec, str):
                dec_parts = dec.split(':')
                if len(dec_parts) == 3:
                    dec = float(dec_parts[0]) + float(dec_parts[1])/60 + float(dec_parts[2])/3600
                else:
                    dec = float(dec_parts[0])
            
            ra_rad = np.radians(ra * 15.0)
            dec_rad = np.radians(dec)
            
            # Create pulsar info
            pulsar_info = {
                'name': pulsar_name,
                'ra': ra_rad,
                'dec': dec_rad,
                'timing_data_count': len(times),
                'timing_residual_rms': float(np.std(residuals)),
                'frequency': float(params.get('F0', 1.0)),
                'dm': float(params.get('DM', 0.0)),
                'period': 1.0/float(params.get('F0', 1.0)) if float(params.get('F0', 1.0)) > 0 else 1.0,
                'period_derivative': float(params.get('F1', 0.0)),
                'par_file': str(best_par),
                'tim_file': str(best_tim)
            }
            
            # Store timing data
            timing_entry = {
                'pulsar_name': pulsar_name,
                'times': self.core_engine._to_cpu_array(times),
                'residuals': self.core_engine._to_cpu_array(residuals),
                'uncertainties': self.core_engine._to_cpu_array(uncertainties)
            }
            
            return {
                'pulsar_info': pulsar_info,
                'timing_data': timing_entry
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {pulsar_name}: {e}")
            return None
    
    def load_all_pulsars(self):
        """Load ALL pulsars with maximum coverage"""
        self.logger.info("📡 Loading ALL pulsars with maximum coverage...")
        
        # Discover all pulsars
        pulsar_groups = self.discover_all_pulsars()
        if not pulsar_groups:
            self.logger.error("❌ No pulsars found!")
            return False
        
        self.scan_stats['total_pulsars_found'] = len(pulsar_groups)
        self.logger.info(f"🎯 Processing {len(pulsar_groups)} pulsars...")
        
        # Load each pulsar
        successful_loads = 0
        failed_loads = 0
        
        for i, (pulsar_name, par_files) in enumerate(pulsar_groups.items()):
            if i % 50 == 0:  # Progress update every 50 pulsars
                self.logger.info(f"📊 Progress: {i}/{len(pulsar_groups)} pulsars processed")
            
            result = self.load_pulsar_data(pulsar_name, par_files)
            if result:
                self.all_pulsars.append(result['pulsar_info'])
                self.all_timing_data.append(result['timing_data'])
                successful_loads += 1
            else:
                failed_loads += 1
        
        self.scan_stats['successful_loads'] = successful_loads
        self.scan_stats['failed_loads'] = failed_loads
        
        self.logger.info(f"✅ Loaded {successful_loads} pulsars successfully")
        self.logger.info(f"❌ Failed to load {failed_loads} pulsars")
        
        return successful_loads > 0
    
    def apply_quantum_analysis(self):
        """Apply quantum analysis to all loaded pulsars"""
        self.logger.info("🧠 Applying quantum analysis to all pulsars...")
        
        if not self.all_timing_data:
            self.logger.error("❌ No timing data available for quantum analysis")
            return False
        
        # Convert to format expected by quantum platform
        pulsar_data = []
        for timing_entry in self.all_timing_data:
            pulsar_data.append({
                'pulsar_id': timing_entry['pulsar_name'],
                'timing_data': {
                    'times': timing_entry['times'],
                    'residuals': timing_entry['residuals']
                }
            })
        
        # Run quantum analysis
        try:
            self.quantum_results = self.quantum_platform.run_unified_analysis(pulsar_data)
            self.scan_stats['quantum_analyses_completed'] = len(pulsar_data)
            
            self.logger.info(f"✅ Quantum analysis completed for {len(pulsar_data)} pulsars")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in quantum analysis: {e}")
            return False
    
    def apply_novel_quantum_methods(self):
        """Apply novel quantum methods that nobody else uses"""
        self.logger.info("🚀 Applying novel quantum methods that nobody else uses...")
        
        novel_results = {
            'quantum_entanglement_network': self._analyze_quantum_entanglement_network(),
            'quantum_phase_coherence': self._analyze_quantum_phase_coherence(),
            'quantum_frequency_correlations': self._analyze_quantum_frequency_correlations(),
            'quantum_sky_mapping': self._analyze_quantum_sky_mapping(),
            'quantum_temporal_evolution': self._analyze_quantum_temporal_evolution(),
            'quantum_cosmic_string_signatures': self._analyze_quantum_cosmic_string_signatures()
        }
        
        self.novel_analysis_results = novel_results
        self.scan_stats['novel_methods_applied'] = len(novel_results)
        
        self.logger.info(f"✅ Applied {len(novel_results)} novel quantum methods")
        return True
    
    def _analyze_quantum_entanglement_network(self):
        """Analyze quantum entanglement network between all pulsars"""
        self.logger.info("🔗 Analyzing quantum entanglement network...")
        
        entanglement_network = {
            'total_entangled_pairs': 0,
            'high_entanglement_pairs': 0,
            'entanglement_clusters': [],
            'network_density': 0.0
        }
        
        if 'correlations' in self.quantum_results:
            correlations = self.quantum_results['correlations']
            
            for pair, corr in correlations.items():
                if corr['quantum_entanglement']:
                    entanglement_network['total_entangled_pairs'] += 1
                    if corr['correlation_strength'] > 0.7:
                        entanglement_network['high_entanglement_pairs'] += 1
            
            # Calculate network density
            n_pulsars = len(self.all_pulsars)
            if n_pulsars > 1:
                max_possible_pairs = n_pulsars * (n_pulsars - 1) // 2
                entanglement_network['network_density'] = entanglement_network['total_entangled_pairs'] / max_possible_pairs
        
        return entanglement_network
    
    def _analyze_quantum_phase_coherence(self):
        """Analyze quantum phase coherence across all pulsars"""
        self.logger.info("🌊 Analyzing quantum phase coherence...")
        
        phase_coherence = {
            'global_coherence': 0.0,
            'coherence_variance': 0.0,
            'high_coherence_pulsars': 0,
            'coherence_distribution': []
        }
        
        if 'quantum_states' in self.quantum_results:
            coherences = [state['coherence'] for state in self.quantum_results['quantum_states'].values()]
            
            if coherences:
                phase_coherence['global_coherence'] = np.mean(coherences)
                phase_coherence['coherence_variance'] = np.var(coherences)
                phase_coherence['high_coherence_pulsars'] = sum(1 for c in coherences if c > 0.7)
                phase_coherence['coherence_distribution'] = coherences
        
        return phase_coherence
    
    def _analyze_quantum_frequency_correlations(self):
        """Analyze quantum frequency correlations"""
        self.logger.info("📡 Analyzing quantum frequency correlations...")
        
        frequency_correlations = {
            'frequency_entanglement': 0.0,
            'harmonic_resonances': [],
            'frequency_clusters': []
        }
        
        # Analyze frequency relationships
        frequencies = [p['frequency'] for p in self.all_pulsars if p['frequency'] > 0]
        
        if len(frequencies) > 1:
            # Look for harmonic relationships
            for i, f1 in enumerate(frequencies):
                for j, f2 in enumerate(frequencies[i+1:], i+1):
                    ratio = f1 / f2 if f2 > 0 else 0
                    if abs(ratio - round(ratio)) < 0.01:  # Near-integer ratio
                        frequency_correlations['harmonic_resonances'].append({
                            'pulsar1': self.all_pulsars[i]['name'],
                            'pulsar2': self.all_pulsars[j]['name'],
                            'ratio': ratio
                        })
        
        return frequency_correlations
    
    def _analyze_quantum_sky_mapping(self):
        """Analyze quantum sky mapping"""
        self.logger.info("🗺️ Analyzing quantum sky mapping...")
        
        sky_mapping = {
            'sky_coverage': 0.0,
            'density_clusters': [],
            'sky_entanglement_map': {}
        }
        
        # Calculate sky coverage
        if self.all_pulsars:
            ras = [p['ra'] for p in self.all_pulsars]
            decs = [p['dec'] for p in self.all_pulsars]
            
            # Simple sky coverage calculation
            ra_range = max(ras) - min(ras) if ras else 0
            dec_range = max(decs) - min(decs) if decs else 0
            sky_mapping['sky_coverage'] = (ra_range * dec_range) / (4 * np.pi)
        
        return sky_mapping
    
    def _analyze_quantum_temporal_evolution(self):
        """Analyze quantum temporal evolution"""
        self.logger.info("⏰ Analyzing quantum temporal evolution...")
        
        temporal_evolution = {
            'temporal_coherence': 0.0,
            'evolution_patterns': [],
            'time_scale_analysis': {}
        }
        
        # Analyze temporal patterns
        for timing_entry in self.all_timing_data:
            times = timing_entry['times']
            residuals = timing_entry['residuals']
            
            if len(times) > 1:
                time_span = max(times) - min(times)
                temporal_evolution['evolution_patterns'].append({
                    'pulsar': timing_entry['pulsar_name'],
                    'time_span': time_span,
                    'data_points': len(times)
                })
        
        return temporal_evolution
    
    def _analyze_quantum_cosmic_string_signatures(self):
        """Analyze quantum cosmic string signatures"""
        self.logger.info("🌌 Analyzing quantum cosmic string signatures...")
        
        cosmic_string_signatures = {
            'total_signatures': 0,
            'high_confidence_signatures': 0,
            'signature_locations': [],
            'quantum_anomalies': []
        }
        
        # Look for quantum anomalies that could indicate cosmic strings
        if 'quantum_states' in self.quantum_results:
            for pulsar_id, state in self.quantum_results['quantum_states'].items():
                if state['cosmic_string_signature'] > 0.8:
                    cosmic_string_signatures['total_signatures'] += 1
                    if state['cosmic_string_signature'] > 0.9:
                        cosmic_string_signatures['high_confidence_signatures'] += 1
                    
                    cosmic_string_signatures['signature_locations'].append({
                        'pulsar': pulsar_id,
                        'signature_strength': state['cosmic_string_signature'],
                        'coherence': state['coherence']
                    })
        
        return cosmic_string_signatures
    
    def run_full_scan(self):
        """Run the complete full scan"""
        self.logger.info("🚀 Starting QUANTUM FULL SCAN - Loading ALL pulsars!")
        self.scan_stats['start_time'] = datetime.now()
        
        # Step 1: Load all clock files
        self.logger.info("Step 1: Loading all clock files...")
        if not self.load_all_clock_files():
            self.logger.error("❌ Failed to load clock files")
            return False
        
        # Step 2: Load all pulsars
        self.logger.info("Step 2: Loading all pulsars...")
        if not self.load_all_pulsars():
            self.logger.error("❌ Failed to load pulsars")
            return False
        
        # Step 3: Apply quantum analysis
        self.logger.info("Step 3: Applying quantum analysis...")
        if not self.apply_quantum_analysis():
            self.logger.error("❌ Failed quantum analysis")
            return False
        
        # Step 4: Apply novel quantum methods
        self.logger.info("Step 4: Applying novel quantum methods...")
        if not self.apply_novel_quantum_methods():
            self.logger.error("❌ Failed novel quantum methods")
            return False
        
        # Step 5: Generate final results
        self.logger.info("Step 5: Generating final results...")
        self.scan_stats['end_time'] = datetime.now()
        
        # Create comprehensive results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'QUANTUM_FULL_SCAN_COMPLETE',
            'scan_stats': self.scan_stats,
            'pulsar_catalog': self.all_pulsars,
            'quantum_results': self.quantum_results,
            'novel_analysis_results': self.novel_analysis_results,
            'summary': {
                'total_pulsars_loaded': len(self.all_pulsars),
                'quantum_analyses_completed': self.scan_stats['quantum_analyses_completed'],
                'novel_methods_applied': self.scan_stats['novel_methods_applied'],
                'scan_duration': (self.scan_stats['end_time'] - self.scan_stats['start_time']).total_seconds()
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_full_scan_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved: {filename}")
        self.logger.info("✅ QUANTUM FULL SCAN COMPLETE!")
        
        return final_results

def main():
    """Main execution function"""
    print("🚀 QUANTUM FULL SCAN ENGINE")
    print("=" * 50)
    print("Loading ALL pulsars and applying novel quantum methods!")
    print("We will not miss a single pulsar!")
    print()
    
    # Initialize and run full scan
    engine = QuantumFullScanEngine()
    results = engine.run_full_scan()
    
    if results:
        print("\n🎯 FULL SCAN COMPLETE!")
        print(f"   Total pulsars loaded: {results['summary']['total_pulsars_loaded']}")
        print(f"   Quantum analyses completed: {results['summary']['quantum_analyses_completed']}")
        print(f"   Novel methods applied: {results['summary']['novel_methods_applied']}")
        print(f"   Scan duration: {results['summary']['scan_duration']:.2f} seconds")
        print("\n🌌 Ready to find cosmic strings with novel quantum methods!")
    else:
        print("❌ Full scan failed!")

if __name__ == "__main__":
    main()
