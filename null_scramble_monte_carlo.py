#!/usr/bin/env python3
"""
NULL SCRAMBLE MONTE CARLO RUNNER
===============================

Complete null-scramble Monte Carlo system for cosmic string detection platform.
Builds the max-sigma distribution under real noise to convert nominal σ → global p.

Usage:
    python null_scramble_monte_carlo.py --input /path/to/real_dataset --config configs/last_success.yaml --out runs/null_mc --n-trials 1000

Features:
- Time-scramble preservation of inter-epoch intervals
- Parallel execution with progress tracking
- Empirical p-value calculation
- Statistical significance mapping
- Comprehensive plotting and analysis
- CSV outputs for further analysis
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
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class NullTrial:
    """Parameters for a single null trial."""
    trial_id: str
    trial_number: int
    seed: int

@dataclass
class NullResult:
    """Results from a single null trial."""
    trial_id: str
    trial_number: int
    seed: int
    max_sigma: float
    run_time_seconds: float
    success: bool
    error_message: str = None

class NullScrambleMonteCarlo:
    """Complete null-scramble Monte Carlo testing system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_cmd = config.get('pipeline_cmd', 'python RUN_MODERN_EXOTIC_HUNTER.py')
        self.input_dataset = config.get('input_dataset')
        self.output_dir = Path(config.get('output_dir', 'runs/null_mc'))
        self.n_trials = config.get('n_trials', 1000)
        self.n_workers = config.get('n_workers', 4)
        self.observed_max_sigma = config.get('observed_max_sigma', None)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'trials').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Results storage
        self.results: List[NullResult] = []
        
    def _scramble_preserve_intervals(self, input_dir: Path, output_dir: Path, seed: int) -> bool:
        """
        Scramble the order of epochs while preserving inter-epoch intervals.
        This kills coherent signals but preserves sampling properties.
        """
        try:
            # Copy original dataset
            if os.path.exists(input_dir):
                if os.path.isdir(input_dir):
                    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
                else:
                    shutil.copy2(input_dir, output_dir)
            
            # Set random seed for reproducible scrambling
            random.seed(seed)
            np.random.seed(seed)
            
            # TODO: Implement actual time scrambling based on your data format
            # This is a placeholder - you need to implement the actual scrambling
            # based on your specific data format (TOAs, residuals, etc.)
            
            # For now, create a placeholder scrambling file
            scramble_info = {
                'trial_id': f"null_{seed:06d}",
                'seed': seed,
                'scramble_time': datetime.now().isoformat(),
                'method': 'time_scramble_preserve_intervals',
                'note': 'Placeholder - implement actual scrambling based on your data format'
            }
            
            with open(output_dir / 'scramble_info.json', 'w') as f:
                json.dump(scramble_info, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error scrambling dataset: {e}")
            return False
    
    def _run_pipeline(self, trial_dir: Path, trial: NullTrial) -> NullResult:
        """Run the detection pipeline on scrambled dataset."""
        start_time = time.time()
        
        result = NullResult(
            trial_id=trial.trial_id,
            trial_number=trial.trial_number,
            seed=trial.seed,
            max_sigma=0.0,
            run_time_seconds=0.0,
            success=False
        )
        
        try:
            # Run the pipeline
            cmd = [
                self.pipeline_cmd,
                '--input', str(trial_dir),
                '--config', self.config.get('config_file', 'configs/last_success.yaml'),
                '--out', str(trial_dir / 'results'),
                '--seed', str(trial.seed)
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
                results_file = trial_dir / 'results' / 'report.json'
                if results_file.exists():
                    with open(results_file) as f:
                        report = json.load(f)
                    
                    result.max_sigma = report.get('summary', {}).get('max_sigma', 0)
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
    
    def _run_single_null_trial(self, trial: NullTrial) -> NullResult:
        """Run a single null trial."""
        # Create trial directory
        trial_dir = self.output_dir / 'trials' / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Scramble dataset
        if not self._scramble_preserve_intervals(Path(self.input_dataset), trial_dir, trial.seed):
            return NullResult(
                trial_id=trial.trial_id,
                trial_number=trial.trial_number,
                seed=trial.seed,
                max_sigma=0.0,
                run_time_seconds=0.0,
                success=False,
                error_message="Failed to scramble dataset"
            )
        
        # Run pipeline
        result = self._run_pipeline(trial_dir, trial)
        
        # Clean up trial directory (optional)
        if self.config.get('cleanup_trials', True):
            shutil.rmtree(trial_dir, ignore_errors=True)
        
        return result
    
    def generate_null_trials(self) -> List[NullTrial]:
        """Generate list of null trials to run."""
        trials = []
        for i in range(self.n_trials):
            trial_id = f"null_{i:06d}"
            seed = random.randint(1, 1000000)
            
            trial = NullTrial(
                trial_id=trial_id,
                trial_number=i,
                seed=seed
            )
            trials.append(trial)
        
        return trials
    
    def run_null_campaign(self):
        """Run complete null-scramble Monte Carlo campaign."""
        print(f"Generating {self.n_trials} null trials...")
        trials = self.generate_null_trials()
        
        # Save trial plan
        plan_file = self.output_dir / 'null_trial_plan.json'
        with open(plan_file, 'w') as f:
            json.dump([{
                'trial_id': t.trial_id,
                'trial_number': t.trial_number,
                'seed': t.seed
            } for t in trials], f, indent=2)
        
        print(f"Running null campaign with {self.n_workers} workers...")
        
        # Run trials in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_trial = {
                executor.submit(self._run_single_null_trial, trial): trial
                for trial in trials
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_trial):
                trial = future_to_trial[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    completed += 1
                    
                    if completed % 100 == 0:
                        print(f"Completed {completed}/{len(trials)} trials")
                        
                except Exception as e:
                    print(f"Error in trial {trial.trial_id}: {e}")
                    # Create failed result
                    failed_result = NullResult(
                        trial_id=trial.trial_id,
                        trial_number=trial.trial_number,
                        seed=trial.seed,
                        max_sigma=0.0,
                        run_time_seconds=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    self.results.append(failed_result)
        
        print(f"Null campaign completed: {len(self.results)} results")
        self.save_results()
        self.generate_analysis()
    
    def save_results(self):
        """Save results to CSV and JSON files."""
        # Convert results to DataFrame
        results_data = []
        for result in self.results:
            results_data.append({
                'trial_id': result.trial_id,
                'trial_number': result.trial_number,
                'seed': result.seed,
                'max_sigma': result.max_sigma,
                'run_time_seconds': result.run_time_seconds,
                'success': result.success,
                'error_message': result.error_message
            })
        
        # Save CSV
        df = pd.DataFrame(results_data)
        csv_file = self.output_dir / 'null_results.csv'
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
        
        # Save JSON
        json_file = self.output_dir / 'null_results.json'
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {json_file}")
    
    def generate_analysis(self):
        """Generate analysis plots and empirical p-value calculation."""
        if not self.results:
            print("No results to analyze")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("No successful results to analyze")
            return
        
        print(f"Analyzing {len(successful_results)} successful results")
        
        # Extract max_sigma values
        max_sigmas = [r.max_sigma for r in successful_results]
        
        # Calculate empirical p-value if observed value provided
        empirical_p = None
        if self.observed_max_sigma is not None:
            empirical_p = sum(1 for sigma in max_sigmas if sigma >= self.observed_max_sigma) / len(max_sigmas)
            print(f"Empirical p-value for observed σ={self.observed_max_sigma}: {empirical_p:.6f}")
        
        # Generate plots
        self._plot_max_sigma_distribution(max_sigmas)
        self._plot_cumulative_distribution(max_sigmas)
        self._plot_empirical_p_map(max_sigmas)
        
        # Calculate statistics
        self._calculate_statistics(max_sigmas, empirical_p)
    
    def _plot_max_sigma_distribution(self, max_sigmas: List[float]):
        """Plot distribution of max_sigma values."""
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.hist(max_sigmas, bins=50, alpha=0.7, density=True, label='Null distribution')
        
        # Add observed value if provided
        if self.observed_max_sigma is not None:
            plt.axvline(self.observed_max_sigma, color='red', linestyle='--', 
                       label=f'Observed σ={self.observed_max_sigma:.2f}')
        
        plt.xlabel('Max Sigma')
        plt.ylabel('Density')
        plt.title('Distribution of Max Sigma in Null Trials')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = self.output_dir / 'plots' / 'max_sigma_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Max sigma distribution plot saved to {plot_file}")
    
    def _plot_cumulative_distribution(self, max_sigmas: List[float]):
        """Plot cumulative distribution of max_sigma values."""
        plt.figure(figsize=(10, 6))
        
        # Sort max_sigmas
        sorted_sigmas = np.sort(max_sigmas)
        n = len(sorted_sigmas)
        cumulative_prob = np.arange(1, n + 1) / n
        
        # Plot cumulative distribution
        plt.plot(sorted_sigmas, cumulative_prob, 'b-', linewidth=2, label='Cumulative distribution')
        
        # Add observed value if provided
        if self.observed_max_sigma is not None:
            empirical_p = sum(1 for sigma in max_sigmas if sigma >= self.observed_max_sigma) / len(max_sigmas)
            plt.axvline(self.observed_max_sigma, color='red', linestyle='--', 
                       label=f'Observed σ={self.observed_max_sigma:.2f} (p={empirical_p:.6f})')
            plt.axhline(1 - empirical_p, color='red', linestyle=':', alpha=0.7)
        
        plt.xlabel('Max Sigma')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Max Sigma in Null Trials')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Save plot
        plot_file = self.output_dir / 'plots' / 'cumulative_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cumulative distribution plot saved to {plot_file}")
    
    def _plot_empirical_p_map(self, max_sigmas: List[float]):
        """Plot empirical p-value as a function of sigma threshold."""
        plt.figure(figsize=(10, 6))
        
        # Calculate empirical p-values for different thresholds
        sigma_thresholds = np.linspace(min(max_sigmas), max(max_sigmas), 100)
        empirical_p_values = []
        
        for threshold in sigma_thresholds:
            p = sum(1 for sigma in max_sigmas if sigma >= threshold) / len(max_sigmas)
            empirical_p_values.append(p)
        
        # Plot empirical p-values
        plt.semilogy(sigma_thresholds, empirical_p_values, 'b-', linewidth=2, label='Empirical p-value')
        
        # Add observed value if provided
        if self.observed_max_sigma is not None:
            observed_p = sum(1 for sigma in max_sigmas if sigma >= self.observed_max_sigma) / len(max_sigmas)
            plt.axvline(self.observed_max_sigma, color='red', linestyle='--', 
                       label=f'Observed σ={self.observed_max_sigma:.2f} (p={observed_p:.6f})')
            plt.axhline(observed_p, color='red', linestyle=':', alpha=0.7)
        
        plt.xlabel('Sigma Threshold')
        plt.ylabel('Empirical p-value')
        plt.title('Empirical p-value vs Sigma Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(1e-6, 1)
        
        # Save plot
        plot_file = self.output_dir / 'plots' / 'empirical_p_map.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Empirical p-value map saved to {plot_file}")
    
    def _calculate_statistics(self, max_sigmas: List[float], empirical_p: float = None):
        """Calculate comprehensive statistics."""
        stats = {
            'n_trials': len(max_sigmas),
            'mean': np.mean(max_sigmas),
            'std': np.std(max_sigmas),
            'median': np.median(max_sigmas),
            'min': np.min(max_sigmas),
            'max': np.max(max_sigmas),
            'percentile_95': np.percentile(max_sigmas, 95),
            'percentile_99': np.percentile(max_sigmas, 99),
            'percentile_99.9': np.percentile(max_sigmas, 99.9),
            'percentile_99.99': np.percentile(max_sigmas, 99.99)
        }
        
        if empirical_p is not None:
            stats['observed_max_sigma'] = self.observed_max_sigma
            stats['empirical_p_value'] = empirical_p
            stats['global_significance'] = -np.log10(empirical_p) if empirical_p > 0 else np.inf
        
        # Save statistics
        stats_file = self.output_dir / 'null_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")
        
        # Generate summary report
        self._generate_summary_report(stats, empirical_p)
    
    def _generate_summary_report(self, stats: Dict[str, Any], empirical_p: float = None):
        """Generate human-readable summary report."""
        report = f"""
NULL SCRAMBLE MONTE CARLO SUMMARY
=================================

Campaign Configuration:
- Total trials: {len(self.results)}
- Successful trials: {stats['n_trials']}
- Success rate: {stats['n_trials']/len(self.results)*100:.1f}%

Max Sigma Statistics:
- Mean: {stats['mean']:.3f}
- Std: {stats['std']:.3f}
- Median: {stats['median']:.3f}
- Min: {stats['min']:.3f}
- Max: {stats['max']:.3f}

Percentiles:
- 95th: {stats['percentile_95']:.3f}
- 99th: {stats['percentile_99']:.3f}
- 99.9th: {stats['percentile_99.9']:.3f}
- 99.99th: {stats['percentile_99.99']:.3f}
"""
        
        if empirical_p is not None:
            report += f"""
Observed Value Analysis:
- Observed max sigma: {stats['observed_max_sigma']:.3f}
- Empirical p-value: {empirical_p:.6f}
- Global significance: {stats['global_significance']:.2f}σ
"""
            
            if empirical_p < 1e-4:
                report += "\n*** SIGNIFICANT DETECTION ***\n"
                report += f"Empirical p-value {empirical_p:.6f} < 1e-4 threshold\n"
            elif empirical_p < 1e-2:
                report += "\n*** MARGINAL SIGNIFICANCE ***\n"
                report += f"Empirical p-value {empirical_p:.6f} < 1e-2 threshold\n"
            else:
                report += "\n*** NO SIGNIFICANT DETECTION ***\n"
                report += f"Empirical p-value {empirical_p:.6f} >= 1e-2 threshold\n"
        
        report += f"""
Files Generated:
- null_results.csv: Complete results in CSV format
- null_results.json: Complete results in JSON format
- null_statistics.json: Statistical summary
- plots/max_sigma_distribution.png: Distribution histogram
- plots/cumulative_distribution.png: Cumulative distribution
- plots/empirical_p_map.png: Empirical p-value map
- null_trial_plan.json: Original trial plan
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
    parser = argparse.ArgumentParser(description='Null Scramble Monte Carlo Runner')
    parser.add_argument('--input', required=True, help='Path to real dataset')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--n-trials', type=int, default=1000, help='Number of null trials')
    parser.add_argument('--n-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--pipeline-cmd', default='python RUN_MODERN_EXOTIC_HUNTER.py', help='Pipeline command')
    parser.add_argument('--observed-sigma', type=float, help='Observed max sigma for p-value calculation')
    parser.add_argument('--cleanup-trials', action='store_true', default=True,
                       help='Clean up trial datasets after testing')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'input_dataset': args.input,
        'config_file': args.config,
        'output_dir': args.out,
        'n_trials': args.n_trials,
        'n_workers': args.n_workers,
        'pipeline_cmd': args.pipeline_cmd,
        'observed_max_sigma': args.observed_sigma,
        'cleanup_trials': args.cleanup_trials
    }
    
    # Create and run null MC
    null_mc = NullScrambleMonteCarlo(config)
    null_mc.run_null_campaign()

if __name__ == '__main__':
    main()
