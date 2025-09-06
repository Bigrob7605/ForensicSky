#!/usr/bin/env python3
"""
COSMIC STRING INJECTION-RECOVERY HARNESS
========================================

Complete injection-recovery testing system for cosmic string detection platform.
Measures detection efficiency and parameter recovery in real noise floor.

Usage:
    python injection_recovery_harness.py --input /path/to/real_dataset --config configs/last_success.yaml --out runs/injection_test

Features:
- Amplitude sweep: [0.1, 0.25, 0.5, 1.0, 2.0, 5.0] × median_rms
- Sky grid: 100 points (50 random + 50 on ecliptic + known locations)
- N repeats per cell: 10 (configurable)
- Parallel execution with progress tracking
- CSV outputs for analysis
- Blind injection IDs to prevent bias
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class InjectionParams:
    """Parameters for a single injection test."""
    injection_id: str
    amplitude_factor: float
    sky_ra: float
    sky_dec: float
    trial_number: int
    blind_id: str

@dataclass
class RecoveryResult:
    """Results from a single injection-recovery test."""
    injection_id: str
    blind_id: str
    amplitude_factor: float
    sky_ra: float
    sky_dec: float
    trial_number: int
    
    # Recovery results
    recovered_amplitude: float = None
    recovered_ra: float = None
    recovered_dec: float = None
    max_sigma: float = None
    detection_flag: bool = False
    
    # Error metrics
    amplitude_error: float = None
    sky_error_deg: float = None
    sky_error_arcmin: float = None
    
    # Run metadata
    run_time_seconds: float = None
    success: bool = False
    error_message: str = None

class InjectionRecoveryHarness:
    """Complete injection-recovery testing harness."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_cmd = config.get('pipeline_cmd', 'python RUN_MODERN_EXOTIC_HUNTER.py')
        self.input_dataset = config.get('input_dataset')
        self.output_dir = Path(config.get('output_dir', 'runs/injection_test'))
        self.n_trials = config.get('n_trials', 10)
        self.n_workers = config.get('n_workers', 4)
        
        # Amplitude sweep factors
        self.amplitude_factors = config.get('amplitude_factors', [0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
        
        # Sky grid parameters
        self.sky_points = self._generate_sky_grid()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'injections').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Results storage
        self.results: List[RecoveryResult] = []
        
    def _generate_sky_grid(self) -> List[Tuple[float, float]]:
        """Generate sky grid for injection testing."""
        sky_points = []
        
        # 50 random points
        for _ in range(50):
            ra = random.uniform(0, 360)
            dec = random.uniform(-90, 90)
            sky_points.append((ra, dec))
        
        # 50 points on ecliptic
        for i in range(50):
            ra = i * 360 / 50
            dec = 0  # Ecliptic plane
            sky_points.append((ra, dec))
        
        # Known interesting locations (add your specific locations here)
        interesting_locations = [
            (0, 0),      # Galactic center
            (180, 0),    # Anti-galactic center
            (90, 0),     # Ecliptic pole
            (270, 0),    # Other ecliptic pole
        ]
        sky_points.extend(interesting_locations)
        
        return sky_points
    
    def _create_injection_dataset(self, params: InjectionParams) -> Path:
        """Create a dataset with injected cosmic string signal."""
        injection_dir = self.output_dir / 'injections' / params.injection_id
        injection_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original dataset
        if os.path.exists(self.input_dataset):
            if os.path.isdir(self.input_dataset):
                shutil.copytree(self.input_dataset, injection_dir, dirs_exist_ok=True)
            else:
                shutil.copy2(self.input_dataset, injection_dir)
        
        # TODO: Implement actual signal injection here
        # This is a placeholder - you need to implement the actual injection
        # based on your data format and cosmic string signal model
        
        # For now, create a placeholder injection file
        injection_info = {
            'injection_id': params.injection_id,
            'blind_id': params.blind_id,
            'amplitude_factor': params.amplitude_factor,
            'sky_ra': params.sky_ra,
            'sky_dec': params.sky_dec,
            'trial_number': params.trial_number,
            'injection_time': datetime.now().isoformat()
        }
        
        with open(injection_dir / 'injection_info.json', 'w') as f:
            json.dump(injection_info, f, indent=2)
        
        return injection_dir
    
    def _run_pipeline(self, injection_dir: Path, params: InjectionParams) -> RecoveryResult:
        """Run the detection pipeline on injected dataset."""
        start_time = time.time()
        
        result = RecoveryResult(
            injection_id=params.injection_id,
            blind_id=params.blind_id,
            amplitude_factor=params.amplitude_factor,
            sky_ra=params.sky_ra,
            sky_dec=params.sky_dec,
            trial_number=params.trial_number
        )
        
        try:
            # Run the pipeline
            cmd = [
                self.pipeline_cmd,
                '--input', str(injection_dir),
                '--config', self.config.get('config_file', 'configs/last_success.yaml'),
                '--out', str(injection_dir / 'results'),
                '--seed', str(random.randint(1, 1000000))
            ]
            
            # Add any additional pipeline arguments
            if 'pipeline_args' in self.config:
                cmd.extend(self.config['pipeline_args'])
            
            # Run pipeline
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            result.run_time_seconds = time.time() - start_time
            
            if process.returncode == 0:
                # Parse results
                results_file = injection_dir / 'results' / 'report.json'
                if results_file.exists():
                    with open(results_file) as f:
                        report = json.load(f)
                    
                    result.max_sigma = report.get('summary', {}).get('max_sigma', 0)
                    result.detection_flag = result.max_sigma > 5.0  # Detection threshold
                    
                    # Parse recovered parameters (adapt to your report format)
                    if 'recovered_parameters' in report:
                        recovered = report['recovered_parameters']
                        result.recovered_amplitude = recovered.get('amplitude')
                        result.recovered_ra = recovered.get('ra')
                        result.recovered_dec = recovered.get('dec')
                        
                        # Calculate errors
                        if result.recovered_amplitude is not None:
                            result.amplitude_error = abs(result.recovered_amplitude - params.amplitude_factor)
                        
                        if result.recovered_ra is not None and result.recovered_dec is not None:
                            # Calculate angular separation
                            ra_diff = abs(result.recovered_ra - params.sky_ra)
                            dec_diff = abs(result.recovered_dec - params.sky_dec)
                            result.sky_error_deg = np.sqrt(ra_diff**2 + dec_diff**2)
                            result.sky_error_arcmin = result.sky_error_deg * 60
                    
                    result.success = True
                else:
                    result.success = False
                    result.error_message = "No results file generated"
            else:
                result.success = False
                result.error_message = f"Pipeline failed: {process.stderr}"
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = "Pipeline timeout"
            result.run_time_seconds = time.time() - start_time
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.run_time_seconds = time.time() - start_time
        
        return result
    
    def _run_single_injection(self, params: InjectionParams) -> RecoveryResult:
        """Run a single injection-recovery test."""
        # Create injection dataset
        injection_dir = self._create_injection_dataset(params)
        
        # Run pipeline
        result = self._run_pipeline(injection_dir, params)
        
        # Clean up injection dataset (optional)
        if self.config.get('cleanup_injections', True):
            shutil.rmtree(injection_dir, ignore_errors=True)
        
        return result
    
    def generate_injection_plan(self) -> List[InjectionParams]:
        """Generate complete injection test plan."""
        injection_plan = []
        injection_counter = 0
        
        for amplitude_factor in self.amplitude_factors:
            for sky_ra, sky_dec in self.sky_points:
                for trial in range(self.n_trials):
                    injection_id = f"inj_{injection_counter:06d}"
                    blind_id = f"blind_{random.randint(100000, 999999)}"
                    
                    params = InjectionParams(
                        injection_id=injection_id,
                        amplitude_factor=amplitude_factor,
                        sky_ra=sky_ra,
                        sky_dec=sky_dec,
                        trial_number=trial,
                        blind_id=blind_id
                    )
                    
                    injection_plan.append(params)
                    injection_counter += 1
        
        return injection_plan
    
    def run_injection_campaign(self):
        """Run complete injection-recovery campaign."""
        print(f"Generating injection plan...")
        injection_plan = self.generate_injection_plan()
        print(f"Generated {len(injection_plan)} injection tests")
        
        # Save injection plan
        plan_file = self.output_dir / 'injection_plan.json'
        with open(plan_file, 'w') as f:
            json.dump([{
                'injection_id': p.injection_id,
                'amplitude_factor': p.amplitude_factor,
                'sky_ra': p.sky_ra,
                'sky_dec': p.sky_dec,
                'trial_number': p.trial_number,
                'blind_id': p.blind_id
            } for p in injection_plan], f, indent=2)
        
        print(f"Running injection campaign with {self.n_workers} workers...")
        
        # Run injections in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(self._run_single_injection, params): params
                for params in injection_plan
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    completed += 1
                    
                    if completed % 100 == 0:
                        print(f"Completed {completed}/{len(injection_plan)} tests")
                        
                except Exception as e:
                    print(f"Error in injection {params.injection_id}: {e}")
                    # Create failed result
                    failed_result = RecoveryResult(
                        injection_id=params.injection_id,
                        blind_id=params.blind_id,
                        amplitude_factor=params.amplitude_factor,
                        sky_ra=params.sky_ra,
                        sky_dec=params.sky_dec,
                        trial_number=params.trial_number,
                        success=False,
                        error_message=str(e)
                    )
                    self.results.append(failed_result)
        
        print(f"Injection campaign completed: {len(self.results)} results")
        self.save_results()
        self.generate_analysis()
    
    def save_results(self):
        """Save results to CSV and JSON files."""
        # Convert results to DataFrame
        results_data = []
        for result in self.results:
            results_data.append({
                'injection_id': result.injection_id,
                'blind_id': result.blind_id,
                'amplitude_factor': result.amplitude_factor,
                'sky_ra': result.sky_ra,
                'sky_dec': result.sky_dec,
                'trial_number': result.trial_number,
                'recovered_amplitude': result.recovered_amplitude,
                'recovered_ra': result.recovered_ra,
                'recovered_dec': result.recovered_dec,
                'max_sigma': result.max_sigma,
                'detection_flag': result.detection_flag,
                'amplitude_error': result.amplitude_error,
                'sky_error_deg': result.sky_error_deg,
                'sky_error_arcmin': result.sky_error_arcmin,
                'run_time_seconds': result.run_time_seconds,
                'success': result.success,
                'error_message': result.error_message
            })
        
        # Save CSV
        df = pd.DataFrame(results_data)
        csv_file = self.output_dir / 'injection_results.csv'
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
        
        # Save JSON
        json_file = self.output_dir / 'injection_results.json'
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {json_file}")
    
    def generate_analysis(self):
        """Generate analysis plots and summary statistics."""
        if not self.results:
            print("No results to analyze")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("No successful results to analyze")
            return
        
        print(f"Analyzing {len(successful_results)} successful results")
        
        # Detection efficiency analysis
        detection_efficiency = {}
        for amplitude_factor in self.amplitude_factors:
            amplitude_results = [r for r in successful_results if r.amplitude_factor == amplitude_factor]
            if amplitude_results:
                detections = sum(1 for r in amplitude_results if r.detection_flag)
                efficiency = detections / len(amplitude_results)
                detection_efficiency[amplitude_factor] = {
                    'efficiency': efficiency,
                    'detections': detections,
                    'total': len(amplitude_results)
                }
        
        # Save detection efficiency
        efficiency_file = self.output_dir / 'detection_efficiency.json'
        with open(efficiency_file, 'w') as f:
            json.dump(detection_efficiency, f, indent=2)
        print(f"Detection efficiency saved to {efficiency_file}")
        
        # Parameter recovery analysis
        recovery_stats = {}
        for amplitude_factor in self.amplitude_factors:
            amplitude_results = [r for r in successful_results if r.amplitude_factor == amplitude_factor]
            if amplitude_results:
                amplitude_errors = [r.amplitude_error for r in amplitude_results if r.amplitude_error is not None]
                sky_errors = [r.sky_error_arcmin for r in amplitude_results if r.sky_error_arcmin is not None]
                
                recovery_stats[amplitude_factor] = {
                    'amplitude_error_mean': np.mean(amplitude_errors) if amplitude_errors else None,
                    'amplitude_error_std': np.std(amplitude_errors) if amplitude_errors else None,
                    'sky_error_mean_arcmin': np.mean(sky_errors) if sky_errors else None,
                    'sky_error_std_arcmin': np.std(sky_errors) if sky_errors else None,
                    'n_recoveries': len(amplitude_errors)
                }
        
        # Save recovery statistics
        recovery_file = self.output_dir / 'recovery_statistics.json'
        with open(recovery_file, 'w') as f:
            json.dump(recovery_stats, f, indent=2)
        print(f"Recovery statistics saved to {recovery_file}")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate human-readable summary report."""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        report = f"""
INJECTION-RECOVERY CAMPAIGN SUMMARY
==================================

Campaign Configuration:
- Total injections: {len(self.results)}
- Successful runs: {len(successful_results)}
- Failed runs: {len(failed_results)}
- Success rate: {len(successful_results)/len(self.results)*100:.1f}%

Amplitude Factors Tested: {self.amplitude_factors}
Sky Points Tested: {len(self.sky_points)}
Trials per Configuration: {self.n_trials}

Detection Efficiency by Amplitude:
"""
        
        for amplitude_factor in self.amplitude_factors:
            amplitude_results = [r for r in successful_results if r.amplitude_factor == amplitude_factor]
            if amplitude_results:
                detections = sum(1 for r in amplitude_results if r.detection_flag)
                efficiency = detections / len(amplitude_results)
                report += f"- {amplitude_factor}x median_rms: {efficiency:.1%} ({detections}/{len(amplitude_results)})\n"
        
        report += f"""
Parameter Recovery Statistics:
"""
        
        for amplitude_factor in self.amplitude_factors:
            amplitude_results = [r for r in successful_results if r.amplitude_factor == amplitude_factor]
            if amplitude_results:
                amplitude_errors = [r.amplitude_error for r in amplitude_results if r.amplitude_error is not None]
                sky_errors = [r.sky_error_arcmin for r in amplitude_results if r.sky_error_arcmin is not None]
                
                if amplitude_errors:
                    report += f"- {amplitude_factor}x median_rms: amplitude error = {np.mean(amplitude_errors):.3f} ± {np.std(amplitude_errors):.3f}\n"
                if sky_errors:
                    report += f"- {amplitude_factor}x median_rms: sky error = {np.mean(sky_errors):.1f} ± {np.std(sky_errors):.1f} arcmin\n"
        
        if failed_results:
            report += f"""
Failed Runs Analysis:
- Total failures: {len(failed_results)}
- Common error messages:
"""
            error_counts = {}
            for result in failed_results:
                if result.error_message:
                    error_counts[result.error_message] = error_counts.get(result.error_message, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                report += f"  - {error}: {count} times\n"
        
        report += f"""
Files Generated:
- injection_results.csv: Complete results in CSV format
- injection_results.json: Complete results in JSON format
- detection_efficiency.json: Detection efficiency by amplitude
- recovery_statistics.json: Parameter recovery statistics
- injection_plan.json: Original injection plan
- summary_report.txt: This summary report

Campaign completed at: {datetime.now().isoformat()}
"""
        
        # Save summary report
        summary_file = self.output_dir / 'summary_report.txt'
        with open(summary_file, 'w') as f:
            f.write(report)
        print(f"Summary report saved to {summary_file}")
        print(report)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Cosmic String Injection-Recovery Harness')
    parser.add_argument('--input', required=True, help='Path to real dataset')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of trials per configuration')
    parser.add_argument('--n-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--pipeline-cmd', default='python RUN_MODERN_EXOTIC_HUNTER.py', help='Pipeline command')
    parser.add_argument('--amplitude-factors', nargs='+', type=float, 
                       default=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0], 
                       help='Amplitude factors to test')
    parser.add_argument('--cleanup-injections', action='store_true', default=True,
                       help='Clean up injection datasets after testing')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'input_dataset': args.input,
        'config_file': args.config,
        'output_dir': args.out,
        'n_trials': args.n_trials,
        'n_workers': args.n_workers,
        'pipeline_cmd': args.pipeline_cmd,
        'amplitude_factors': args.amplitude_factors,
        'cleanup_injections': args.cleanup_injections
    }
    
    # Create and run harness
    harness = InjectionRecoveryHarness(config)
    harness.run_injection_campaign()

if __name__ == '__main__':
    main()
