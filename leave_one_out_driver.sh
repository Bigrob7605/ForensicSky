#!/bin/bash
# Leave-One-Out Driver for Cosmic String Detection Platform
# =========================================================
#
# Tests for single-pulsar domination by removing each top pulsar and re-running analysis.
# If significance collapses after removing any single pulsar, investigate that pulsar.
#
# Usage:
#   ./leave_one_out_driver.sh --input /path/to/real_dataset --config configs/last_success.yaml --out runs/loo_test
#
# Features:
# - Tests top 10 contributing pulsars individually
# - Parallel execution for speed
# - Comprehensive logging and reporting
# - CSV outputs for analysis
# - Automatic significance comparison

set -euo pipefail

# Default configuration
PIPELINE_CMD="python RUN_MODERN_EXOTIC_HUNTER.py"
INPUT_DATASET=""
CONFIG_FILE=""
OUTPUT_DIR=""
N_WORKERS=4
TOP_PULSARS=("J1909-3744" "J1713+0747" "J1744-1134" "J1600-3053" "J0437-4715" "J0437-4715" "J1741+1351" "J1857+0943" "J1939+2134" "J2145-0750")
PIPELINE_ARGS=""
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Leave-One-Out Driver for Cosmic String Detection Platform
========================================================

Tests for single-pulsar domination by removing each top pulsar and re-running analysis.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --input PATH              Path to real dataset (required)
    --config PATH             Path to config file (required)
    --out PATH                Output directory (required)
    --pipeline-cmd CMD        Pipeline command (default: python RUN_MODERN_EXOTIC_HUNTER.py)
    --n-workers N             Number of parallel workers (default: 4)
    --top-pulsars LIST        Comma-separated list of pulsars to test (default: top 10)
    --pipeline-args ARGS      Additional pipeline arguments
    --verbose                 Enable verbose output
    --help                    Show this help message

EXAMPLES:
    # Basic usage
    $0 --input /path/to/real_dataset --config configs/last_success.yaml --out runs/loo_test
    
    # With custom pulsars
    $0 --input /path/to/real_dataset --config configs/last_success.yaml --out runs/loo_test \\
       --top-pulsars "J1909-3744,J1713+0747,J1744-1134"
    
    # With custom pipeline
    $0 --input /path/to/real_dataset --config configs/last_success.yaml --out runs/loo_test \\
       --pipeline-cmd "python 01_Core_Engine/Core_ForensicSky_V1.py"

OUTPUTS:
    - loo_results.csv: Complete results in CSV format
    - loo_summary.json: Summary statistics
    - loo_plots/: Analysis plots
    - logs/: Individual run logs
    - summary_report.txt: Human-readable summary

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input)
                INPUT_DATASET="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --out)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --pipeline-cmd)
                PIPELINE_CMD="$2"
                shift 2
                ;;
            --n-workers)
                N_WORKERS="$2"
                shift 2
                ;;
            --top-pulsars)
                IFS=',' read -ra TOP_PULSARS <<< "$2"
                shift 2
                ;;
            --pipeline-args)
                PIPELINE_ARGS="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_args() {
    if [[ -z "$INPUT_DATASET" ]]; then
        log_error "Input dataset path is required"
        show_help
        exit 1
    fi
    
    if [[ -z "$CONFIG_FILE" ]]; then
        log_error "Config file path is required"
        show_help
        exit 1
    fi
    
    if [[ -z "$OUTPUT_DIR" ]]; then
        log_error "Output directory is required"
        show_help
        exit 1
    fi
    
    if [[ ! -d "$INPUT_DATASET" ]] && [[ ! -f "$INPUT_DATASET" ]]; then
        log_error "Input dataset does not exist: $INPUT_DATASET"
        exit 1
    fi
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file does not exist: $CONFIG_FILE"
        exit 1
    fi
}

# Create output directories
setup_output_dirs() {
    log_info "Setting up output directories..."
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/loo_runs"
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/plots"
    mkdir -p "$OUTPUT_DIR/results"
    
    log_success "Output directories created"
}

# Run baseline analysis (no pulsars excluded)
run_baseline() {
    log_info "Running baseline analysis (no pulsars excluded)..."
    
    local baseline_dir="$OUTPUT_DIR/baseline"
    mkdir -p "$baseline_dir"
    
    local cmd=(
        "$PIPELINE_CMD"
        --input "$INPUT_DATASET"
        --config "$CONFIG_FILE"
        --out "$baseline_dir"
        --seed 123456
    )
    
    if [[ -n "$PIPELINE_ARGS" ]]; then
        cmd+=($PIPELINE_ARGS)
    fi
    
    log_info "Running: ${cmd[*]}"
    
    if "$VERBOSE"; then
        "${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/baseline.log"
    else
        "${cmd[@]}" > "$OUTPUT_DIR/logs/baseline.log" 2>&1
    fi
    
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Baseline analysis completed"
        
        # Extract baseline max_sigma
        if [[ -f "$baseline_dir/report.json" ]]; then
            BASELINE_MAX_SIGMA=$(jq -r '.summary.max_sigma // 0' "$baseline_dir/report.json" 2>/dev/null || echo "0")
            log_info "Baseline max_sigma: $BASELINE_MAX_SIGMA"
        else
            log_warning "No report.json found in baseline run"
            BASELINE_MAX_SIGMA="0"
        fi
    else
        log_error "Baseline analysis failed with exit code $exit_code"
        BASELINE_MAX_SIGMA="0"
    fi
}

# Run single leave-one-out test
run_loo_test() {
    local pulsar="$1"
    local run_dir="$OUTPUT_DIR/loo_runs/loo_$pulsar"
    
    log_info "Running leave-one-out test for pulsar: $pulsar"
    
    mkdir -p "$run_dir"
    
    local cmd=(
        "$PIPELINE_CMD"
        --input "$INPUT_DATASET"
        --config "$CONFIG_FILE"
        --out "$run_dir"
        --exclude-pulsar "$pulsar"
        --seed 123456
    )
    
    if [[ -n "$PIPELINE_ARGS" ]]; then
        cmd+=($PIPELINE_ARGS)
    fi
    
    log_info "Running: ${cmd[*]}"
    
    local start_time=$(date +%s)
    
    if "$VERBOSE"; then
        "${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/loo_$pulsar.log"
    else
        "${cmd[@]}" > "$OUTPUT_DIR/logs/loo_$pulsar.log" 2>&1
    fi
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local run_time=$((end_time - start_time))
    
    local result=""
    local max_sigma="0"
    local success="false"
    
    if [[ $exit_code -eq 0 ]]; then
        if [[ -f "$run_dir/report.json" ]]; then
            max_sigma=$(jq -r '.summary.max_sigma // 0' "$run_dir/report.json" 2>/dev/null || echo "0")
            success="true"
            log_success "LOO test for $pulsar completed (max_sigma: $max_sigma, time: ${run_time}s)"
        else
            log_warning "No report.json found for $pulsar"
            success="false"
        fi
    else
        log_error "LOO test for $pulsar failed with exit code $exit_code"
        success="false"
    fi
    
    # Calculate significance change
    local significance_change="0"
    if [[ "$BASELINE_MAX_SIGMA" != "0" ]] && [[ "$max_sigma" != "0" ]]; then
        significance_change=$(echo "$max_sigma - $BASELINE_MAX_SIGMA" | bc -l 2>/dev/null || echo "0")
    fi
    
    # Determine if this pulsar dominates
    local dominates="false"
    if [[ $(echo "$significance_change < -2.0" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        dominates="true"
        log_warning "Pulsar $pulsar appears to dominate (significance drop: $significance_change)"
    fi
    
    # Save individual result
    cat > "$OUTPUT_DIR/results/loo_$pulsar.json" << EOF
{
    "pulsar": "$pulsar",
    "max_sigma": $max_sigma,
    "baseline_max_sigma": $BASELINE_MAX_SIGMA,
    "significance_change": $significance_change,
    "dominates": $dominates,
    "success": $success,
    "run_time_seconds": $run_time,
    "exit_code": $exit_code
}
EOF
    
    echo "$pulsar,$max_sigma,$BASELINE_MAX_SIGMA,$significance_change,$dominates,$success,$run_time,$exit_code"
}

# Run all leave-one-out tests
run_all_loo_tests() {
    log_info "Running leave-one-out tests for ${#TOP_PULSARS[@]} pulsars..."
    
    # Create results CSV header
    echo "pulsar,max_sigma,baseline_max_sigma,significance_change,dominates,success,run_time_seconds,exit_code" > "$OUTPUT_DIR/loo_results.csv"
    
    # Run tests in parallel
    local pids=()
    local results=()
    
    for pulsar in "${TOP_PULSARS[@]}"; do
        # Run in background
        (
            result=$(run_loo_test "$pulsar")
            echo "$result" >> "$OUTPUT_DIR/loo_results.csv"
        ) &
        
        pids+=($!)
        
        # Limit number of parallel jobs
        if [[ ${#pids[@]} -ge $N_WORKERS ]]; then
            wait_for_jobs "${pids[@]}"
            pids=()
        fi
    done
    
    # Wait for remaining jobs
    if [[ ${#pids[@]} -gt 0 ]]; then
        wait_for_jobs "${pids[@]}"
    fi
    
    log_success "All leave-one-out tests completed"
}

# Wait for background jobs
wait_for_jobs() {
    local pids=("$@")
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# Generate analysis and plots
generate_analysis() {
    log_info "Generating analysis and plots..."
    
    # Create Python analysis script
    cat > "$OUTPUT_DIR/analyze_loo.py" << 'EOF'
#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def analyze_loo_results(results_file, output_dir):
    """Analyze leave-one-out results and generate plots."""
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Filter successful results
    successful = df[df['success'] == True]
    
    if len(successful) == 0:
        print("No successful results to analyze")
        return
    
    print(f"Analyzing {len(successful)} successful results")
    
    # Create plots
    plt.style.use('default')
    
    # Plot 1: Max sigma comparison
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(successful))
    plt.bar(x_pos, successful['max_sigma'], alpha=0.7, label='LOO Max Sigma')
    plt.bar(x_pos, successful['baseline_max_sigma'], alpha=0.5, label='Baseline Max Sigma')
    plt.xticks(x_pos, successful['pulsar'], rotation=45, ha='right')
    plt.ylabel('Max Sigma')
    plt.title('Leave-One-Out Max Sigma Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/max_sigma_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Significance change
    plt.figure(figsize=(12, 6))
    colors = ['red' if x < -2.0 else 'orange' if x < -1.0 else 'green' for x in successful['significance_change']]
    plt.bar(x_pos, successful['significance_change'], color=colors, alpha=0.7)
    plt.xticks(x_pos, successful['pulsar'], rotation=45, ha='right')
    plt.ylabel('Significance Change (σ)')
    plt.title('Significance Change After Removing Each Pulsar')
    plt.axhline(y=-2.0, color='red', linestyle='--', alpha=0.7, label='Dominance Threshold')
    plt.axhline(y=-1.0, color='orange', linestyle='--', alpha=0.7, label='Marginal Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/significance_change.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Run time comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x_pos, successful['run_time_seconds'], alpha=0.7)
    plt.xticks(x_pos, successful['pulsar'], rotation=45, ha='right')
    plt.ylabel('Run Time (seconds)')
    plt.title('Leave-One-Out Run Time Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/run_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary statistics
    summary = {
        'total_pulsars_tested': len(df),
        'successful_tests': len(successful),
        'failed_tests': len(df[df['success'] == False]),
        'success_rate': len(successful) / len(df) * 100,
        'dominating_pulsars': len(successful[successful['dominates'] == True]),
        'marginal_pulsars': len(successful[(successful['significance_change'] < -1.0) & (successful['significance_change'] >= -2.0)]),
        'baseline_max_sigma': successful['baseline_max_sigma'].iloc[0] if len(successful) > 0 else 0,
        'mean_loo_max_sigma': successful['max_sigma'].mean(),
        'std_loo_max_sigma': successful['max_sigma'].std(),
        'min_significance_change': successful['significance_change'].min(),
        'max_significance_change': successful['significance_change'].max(),
        'mean_run_time': successful['run_time_seconds'].mean(),
        'total_run_time': successful['run_time_seconds'].sum()
    }
    
    # Save summary
    with open(f'{output_dir}/loo_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Analysis completed")
    print(f"Summary saved to {output_dir}/loo_summary.json")
    print(f"Plots saved to {output_dir}/plots/")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python analyze_loo.py <results_file> <output_dir>")
        sys.exit(1)
    
    analyze_loo_results(sys.argv[1], sys.argv[2])
EOF
    
    # Run analysis
    python3 "$OUTPUT_DIR/analyze_loo.py" "$OUTPUT_DIR/loo_results.csv" "$OUTPUT_DIR"
    
    log_success "Analysis completed"
}

# Generate summary report
generate_summary_report() {
    log_info "Generating summary report..."
    
    local summary_file="$OUTPUT_DIR/loo_summary.json"
    local report_file="$OUTPUT_DIR/summary_report.txt"
    
    if [[ -f "$summary_file" ]]; then
        local total_pulsars=$(jq -r '.total_pulsars_tested' "$summary_file")
        local successful_tests=$(jq -r '.successful_tests' "$summary_file")
        local failed_tests=$(jq -r '.failed_tests' "$summary_file")
        local success_rate=$(jq -r '.success_rate' "$summary_file")
        local dominating_pulsars=$(jq -r '.dominating_pulsars' "$summary_file")
        local marginal_pulsars=$(jq -r '.marginal_pulsars' "$summary_file")
        local baseline_max_sigma=$(jq -r '.baseline_max_sigma' "$summary_file")
        local mean_loo_max_sigma=$(jq -r '.mean_loo_max_sigma' "$summary_file")
        local min_significance_change=$(jq -r '.min_significance_change' "$summary_file")
        local max_significance_change=$(jq -r '.max_significance_change' "$summary_file")
        
        cat > "$report_file" << EOF
LEAVE-ONE-OUT ANALYSIS SUMMARY
==============================

Test Configuration:
- Total pulsars tested: $total_pulsars
- Successful tests: $successful_tests
- Failed tests: $failed_tests
- Success rate: $success_rate%

Baseline Analysis:
- Baseline max_sigma: $baseline_max_sigma
- Mean LOO max_sigma: $mean_loo_max_sigma

Significance Analysis:
- Dominating pulsars (Δσ < -2.0): $dominating_pulsars
- Marginal pulsars (-2.0 ≤ Δσ < -1.0): $marginal_pulsars
- Min significance change: $min_significance_change
- Max significance change: $max_significance_change

Interpretation:
EOF
        
        if [[ "$dominating_pulsars" -gt 0 ]]; then
            echo "*** WARNING: $dominating_pulsars pulsar(s) appear to dominate the signal ***" >> "$report_file"
            echo "This suggests the signal may be due to systematic effects in specific pulsars." >> "$report_file"
        elif [[ "$marginal_pulsars" -gt 0 ]]; then
            echo "*** CAUTION: $marginal_pulsars pulsar(s) show marginal dominance ***" >> "$report_file"
            echo "Monitor these pulsars for potential systematic effects." >> "$report_file"
        else
            echo "*** GOOD: No pulsars show significant dominance ***" >> "$report_file"
            echo "The signal appears to be robust against single-pulsar removal." >> "$report_file"
        fi
        
        echo "" >> "$report_file"
        echo "Files Generated:" >> "$report_file"
        echo "- loo_results.csv: Complete results in CSV format" >> "$report_file"
        echo "- loo_summary.json: Summary statistics" >> "$report_file"
        echo "- plots/: Analysis plots" >> "$report_file"
        echo "- logs/: Individual run logs" >> "$report_file"
        echo "- summary_report.txt: This summary report" >> "$report_file"
        echo "" >> "$report_file"
        echo "Analysis completed at: $(date)" >> "$report_file"
        
        log_success "Summary report generated: $report_file"
        
        # Display key results
        echo ""
        log_info "KEY RESULTS:"
        echo "  - Baseline max_sigma: $baseline_max_sigma"
        echo "  - Dominating pulsars: $dominating_pulsars"
        echo "  - Marginal pulsars: $marginal_pulsars"
        echo "  - Success rate: $success_rate%"
        
        if [[ "$dominating_pulsars" -gt 0 ]]; then
            log_warning "WARNING: $dominating_pulsars pulsar(s) dominate the signal!"
        fi
    else
        log_error "Summary file not found: $summary_file"
    fi
}

# Main execution
main() {
    log_info "Starting Leave-One-Out Analysis"
    log_info "================================"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate arguments
    validate_args
    
    # Setup
    setup_output_dirs
    
    # Run baseline analysis
    run_baseline
    
    # Run leave-one-out tests
    run_all_loo_tests
    
    # Generate analysis
    generate_analysis
    
    # Generate summary report
    generate_summary_report
    
    log_success "Leave-One-Out Analysis completed successfully!"
    log_info "Results saved to: $OUTPUT_DIR"
}

# Run main function
main "$@"
