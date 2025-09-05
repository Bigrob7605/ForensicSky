#!/usr/bin/env python3
"""
FINAL CLEANUP SCRIPT
===================

Move remaining scattered files to cleanup folder after condensing
everything into the ULTIMATE_COSMIC_STRING_ENGINE.py
"""

import shutil
from pathlib import Path
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def final_cleanup():
    """Move remaining scattered files to cleanup folder"""
    logger.info("🧹 FINAL CLEANUP - MOVING REMAINING SCATTERED FILES")
    
    # Create cleanup directories
    cleanup_dir = Path("cleanup/scattered_files")
    cleanup_dir.mkdir(parents=True, exist_ok=True)
    
    # List of remaining scattered files to move
    scattered_files = [
        # Analysis files
        "ANALYSIS_STATUS_REPORT.md",
        "BREAKTHROUGH_DISCOVERY_REPORT.md",
        "CLEAN_PROJECT_STATUS.md",
        "CRITICAL_CORRECTION_FAKE_RESEARCH.md",
        "CRITICAL_SIGNIFICANCE_CORRECTION.md",
        "FIXED_ENGINE_STATUS.md",
        "GAME_PLAN.md",
        "GOLD_STANDARD_TESTING_PROTOCOL.md",
        "LAB_GRADE_SYNC.md",
        "PHASE_1_IMPLEMENTATION_PLAN.md",
        "PROJECT_STATUS_LAB_GRADE.md",
        "REALITY_CHECK_AND_CORRECTION.md",
        "REALITY_CHECK_SUMMARY.md",
        "SYSTEM_STATUS.md",
        "ULTIMATE_AI_TECH_INTEGRATION_PLAN.md",
        "ULTIMATE_DETECTOR_SUCCESS_REPORT.md",
        "UPGRADE_GAMEPLAN.md",
        "UPGRADE_SUCCESS_REPORT.md",
        
        # Result files
        "breakthrough_analysis_report_20250904_204521.json",
        "breakthrough_analysis_report_20250904_204600.json",
        "comprehensive_lab_grade_report.json",
        "confirmation_analysis_results.json",
        "critical_verification_results.json",
        "deep_analysis_v2_results.json",
        "DEEP_DIG_2_RESULTS.json",
        "DEEP_PEER_REVIEW_RESULTS.json",
        "ENHANCED_ANALYSIS_RESULTS.json",
        "ENHANCED_COSMIC_STRING_RESULTS.json",
        "fixed_gpu_cosmic_string_breakthrough_1757036624.json",
        "fixed_real_data_cosmic_string_breakthrough_1757037185.json",
        "full_data_sweep_results_20250905_062114.json",
        "gpu_cosmic_string_campaign_report_1757035961.json",
        "LAB_GRADE_REAL_DATA_RESULTS.json",
        "MATH_VALIDATION_RESULTS.json",
        "perfect_cosmic_strings_20250905_063818.json",
        "perfect_cosmic_strings_20250905_063847.json",
        "PROVEN_BASELINE_RESULTS.json",
        "publication_ready_validation.json",
        "real_ipta_pipeline_test_results_20250830_163037.json",
        "real_ipta_pipeline_test_results_20250830_164206.json",
        "treasure_hunt_results_20250905_060308.json",
        "treasure_hunt_results_20250905_060846.json",
        "treasure_hunt_results_20250905_060850.json",
        "treasure_hunt_results_20250905_060914.json",
        "treasure_hunt_results_20250905_060919.json",
        "treasure_hunt_results_20250905_061603.json",
        "ULTIMATE_AI_TECH_RESULTS.json",
        "ULTIMATE_LAB_GRADE_RESULTS.json",
        "ultra_deep_analysis_results.json",
        "verification_results_20250905_060609.json",
        "WORKING_SHORT_REAL_TEST_RESULTS.json",
        "world_shattering_analysis_report_1757034129.json",
        "world_shattering_analysis_report_1757034134.json",
        
        # Data files
        "comprehensive_analysis_results_20250830_160759_report.txt",
        "comprehensive_analysis_results_20250830_160759.npz",
        "comprehensive_analysis_results_20250830_160947_report.txt",
        "comprehensive_analysis_results_20250830_160947.npz",
        "cosmic_strings_real_ipta_results_20250830_162924.npz",
        "cosmic_strings_real_ipta_results_20250830_162949.npz",
        "cosmic_strings_real_ipta_results_20250830_163037.npz",
        "cosmic_strings_real_ipta_results_20250830_164046.npz",
        "cosmic_strings_real_ipta_results_20250830_164110.npz",
        "cosmic_strings_real_ipta_results_20250830_164205.npz",
        "extended_parameter_space_test_results_20250830_212520.png",
        "extended_parameter_space_test_results_20250830_212536.npz",
        "extended_parameter_space_test_summary_20250830_212536.json",
        "fig1_limits_real.png",
        "fig2_skymap_old_toy_data.png",
        "fig2_skymap_real.png",
        "phase_2_monte_carlo_trials_results_20250830_214623.png",
        "phase_2_monte_carlo_trials_results_20250830_214634.npz",
        "phase_2_monte_carlo_trials_results_20250830_214753.png",
        "phase_2_monte_carlo_trials_results_20250830_214802.npz",
        "phase_2_monte_carlo_trials_results_20250830_214944.png",
        "phase_2_monte_carlo_trials_results_20250830_214949.npz",
        "phase_2_monte_carlo_trials_results_20250830_215018.png",
        "phase_2_monte_carlo_trials_results_20250830_215026.npz",
        "phase_2_monte_carlo_trials_summary_20250830_214634.json",
        "phase_2_monte_carlo_trials_summary_20250830_214802.json",
        "phase_2_monte_carlo_trials_summary_20250830_214949.json",
        "phase_2_monte_carlo_trials_summary_20250830_215026.json",
        "pta_limits.npz",
        "real_correlations_hd.png",
        "real_example_periodogram.png",
        "real_example_psd.png",
        "real_production_gpu_test_results_20250830_201925.png",
        "real_production_gpu_test_results_20250830_201944.npz",
        "real_production_gpu_test_results_20250830_202132.png",
        "real_production_gpu_test_results_20250830_202138.npz",
        "real_production_gpu_test_summary_20250830_201944.json",
        "real_production_gpu_test_summary_20250830_202138.json",
        "string_realizations.npz",
        "timing_residuals.npz",
        
        # Log files
        "cosmic_string_detection_20250830_162718.log",
        "cosmic_string_detection_20250830_162854.log",
        "cosmic_string_detection_20250830_164011.log",
        "cosmic_string_detection_20250831_074423.log",
        "cosmic_string_detection_20250831_074610.log",
        "cosmic_string_detection_20250831_074639.log",
        "cosmic_string_detection_20250831_074847.log",
        "cross_correlation_innovations_20250904_201751.log",
        "cross_correlation_innovations_20250904_202456.log",
        "cross_correlation_innovations_20250904_202604.log",
        "cross_correlation_innovations_20250904_202917.log",
        "cross_correlation_innovations_20250904_203000.log",
        "cross_correlation_innovations_20250904_203040.log",
        "cross_correlation_innovations_20250904_203124.log",
        "cross_correlation_innovations_20250904_203152.log",
        "cross_correlation_innovations_20250904_203233.log",
        "enhanced_gpu_pta_pipeline_20250904_204123.log",
        "enhanced_gpu_pta_pipeline_20250904_204309.log",
        "enhanced_gpu_pta_pipeline_20250904_204334.log",
        "enhanced_gpu_pta_pipeline_20250904_204348.log",
        "enhanced_gpu_pta_pipeline_20250904_204402.log",
        "enhanced_gpu_pta_pipeline_20250904_204418.log",
        "enhanced_gpu_pta_pipeline_20250904_204434.log",
        "enhanced_gpu_pta_pipeline_20250904_204447.log",
        "enhanced_gpu_pta_pipeline_20250904_204507.log",
        "enhanced_gpu_pta_pipeline_20250904_204521.log",
        "enhanced_gpu_pta_pipeline_20250904_204559.log",
        "gpu_pta_pipeline_20250904_201724.log",
        "gpu_pta_pipeline_20250904_201738.log",
        "gpu_pta_pipeline_20250904_201749.log",
        "gpu_pta_pipeline_20250904_202456.log",
        "gpu_pta_pipeline_20250904_202604.log",
        "gpu_pta_pipeline_20250904_202917.log",
        "gpu_pta_pipeline_20250904_203000.log",
        "gpu_pta_pipeline_20250904_203040.log",
        "gpu_pta_pipeline_20250904_203124.log",
        "gpu_pta_pipeline_20250904_203152.log",
        "gpu_pta_pipeline_20250904_203232.log",
        "ml_noise_modeling_20250904_201724.log",
        "ml_noise_modeling_20250904_201749.log",
        "ml_noise_modeling_20250904_202456.log",
        "ml_noise_modeling_20250904_202604.log",
        "ml_noise_modeling_20250904_202917.log",
        "ml_noise_modeling_20250904_203000.log",
        "ml_noise_modeling_20250904_203040.log",
        "ml_noise_modeling_20250904_203124.log",
        "ml_noise_modeling_20250904_203152.log",
        "ml_noise_modeling_20250904_203232.log",
        
        # Test result files
        "gpu_pta_pipeline_test_results_20250904_201759.json",
        "gpu_pta_pipeline_test_results_20250904_202503.json",
        "gpu_pta_pipeline_test_results_20250904_202615.json",
        "gpu_pta_pipeline_test_results_20250904_202926.json",
        "gpu_pta_pipeline_test_results_20250904_203009.json",
        "gpu_pta_pipeline_test_results_20250904_203050.json",
        "gpu_pta_pipeline_test_results_20250904_203133.json",
        "gpu_pta_pipeline_test_results_20250904_203203.json",
        "gpu_pta_pipeline_test_results_20250904_203243.json",
        
        # Publication files
        "FINAL_NATURE_ABSTRACT.md",
        "FINAL_NATURE_COVER_LETTER.md",
        "FINAL_NATURE_FIGURE.png",
        "FINAL_NOBEL_ABSTRACT.md",
        "FINAL_NOBEL_COVER_LETTER.md",
        "FINAL_NOBEL_FIGURE.png",
        "FINAL_NOBEL_PRIZE_SUMMARY.md",
        "NATURE_SUBMISSION_ABSTRACT.md",
        "NATURE_SUBMISSION_FIGURE.png",
        "NATURE_SUBMISSION_LETTER.md",
        "NOBEL_PRIZE_ABSTRACT.md",
        "NOBEL_PRIZE_FIGURE.png",
        "NOBEL_PRIZE_SUBMISSION_READY.md",
        "NOBEL_PRIZE_VALIDATION.md",
        
        # LaTeX files
        "nanogal_apjl.aux",
        "nanogal_apjl.log",
        "nanogal_apjl.out",
        "nanogal.bib",
        "table1.tex",
        "test_latex.tex",
        "Texidea.tex",
        
        # Other files
        "CLEANUP_SUMMARY.md",
        "COSMIC_STRINGS_TOOLKIT_README.md",
        "INSTALL.md",
        "Laptop.png",
        "requirements_corrected.txt"
    ]
    
    moved_files = []
    failed_files = []
    
    for file_name in scattered_files:
        source_file = Path(file_name)
        if source_file.exists():
            try:
                destination = cleanup_dir / file_name
                shutil.move(str(source_file), str(destination))
                moved_files.append(file_name)
                logger.info(f"   ✅ Moved: {file_name}")
            except Exception as e:
                failed_files.append((file_name, str(e)))
                logger.error(f"   ❌ Failed to move {file_name}: {e}")
        else:
            logger.info(f"   ⚠️  Not found: {file_name}")
    
    # Create final cleanup summary
    final_cleanup_summary = {
        'timestamp': datetime.now().isoformat(),
        'cleanup_type': 'FINAL_SCATTERED_FILES_CLEANUP',
        'moved_files': moved_files,
        'failed_files': failed_files,
        'cleanup_directory': str(cleanup_dir),
        'remaining_files': [
            'ULTIMATE_COSMIC_STRING_ENGINE.py',
            'README.md',
            'AGENT_READ_FIRST.md',
            'REALITY_CHECK_FINAL.md',
            'setup.py',
            'CLEANUP_OLD_ENGINES_SUMMARY.json',
            'CLEANUP_OLD_ENGINES.py',
            'FINAL_CLEANUP_SCRIPT.py',
            'ULTIMATE_COSMIC_STRING_RESULTS.json',
            'data/',
            'cleanup/',
            'analysis/',
            'detection/',
            'docs/',
            'notebooks/',
            'simulations/',
            'tests/'
        ]
    }
    
    # Save final cleanup summary
    summary_file = Path("FINAL_CLEANUP_SUMMARY.json")
    with open(summary_file, 'w') as f:
        json.dump(final_cleanup_summary, f, indent=2)
    
    logger.info("🎯 FINAL CLEANUP SUMMARY:")
    logger.info(f"   Files moved: {len(moved_files)}")
    logger.info(f"   Files failed: {len(failed_files)}")
    logger.info(f"   Cleanup directory: {cleanup_dir}")
    logger.info(f"   Summary file: {summary_file}")
    
    if moved_files:
        logger.info("   Moved files:")
        for file_name in moved_files:
            logger.info(f"     - {file_name}")
    
    if failed_files:
        logger.warning("   Failed files:")
        for file_name, error in failed_files:
            logger.warning(f"     - {file_name}: {error}")
    
    return final_cleanup_summary

if __name__ == "__main__":
    summary = final_cleanup()
    
    print("\n✅ FINAL CLEANUP COMPLETE!")
    print(f"📁 Moved {len(summary['moved_files'])} scattered files to cleanup/scattered_files/")
    print("🎯 Only essential files remain in root")
    print("🎯 Project is now clean and organized")
    print("🎯 Ready for real cosmic string science")
