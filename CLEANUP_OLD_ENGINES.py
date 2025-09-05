#!/usr/bin/env python3
"""
CLEANUP OLD ENGINES
==================

Move all the old engine files to cleanup folder after condensing
everything into the ULTIMATE_COSMIC_STRING_ENGINE.py
"""

import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_old_engines():
    """Move old engine files to cleanup folder"""
    logger.info("🧹 CLEANING UP OLD ENGINES")
    
    # Create cleanup directory
    cleanup_dir = Path("cleanup/old_engines")
    cleanup_dir.mkdir(parents=True, exist_ok=True)
    
    # List of old engine files to move
    old_engine_files = [
        "ULTIMATE_LAB_GRADE_ENGINE.py",
        "BUILD_ON_ESTABLISHED_TOOLS.py",
        "INTEGRATE_ESTABLISHED_TOOLS.py",
        "PROVEN_BASELINE_TESTS.py",
        "PROPER_COSMIC_STRING_HUNT.py",
        "SCRAPE_REAL_IPTA_DATA.py",
        "DOWNLOAD_REAL_IPTA_FROM_SOURCES.py",
        "DOWNLOAD_REAL_IPTA_DATA.py",
        "CHECK_REAL_DATA_VERSIONS.py",
        "REAL_DATA_VALIDATION.py",
        "DEBUG_DATA_ANALYSIS.py",
        "ESTABLISHED_TOOLS_INTEGRATION_SUMMARY.md",
        "PROJECT_CLEANUP_SUMMARY.md",
        "MASSIVE_CLEANUP_SCRIPT.py"
    ]
    
    moved_files = []
    failed_files = []
    
    for file_name in old_engine_files:
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
    
    # Create cleanup summary
    cleanup_summary = {
        'timestamp': datetime.now().isoformat(),
        'cleanup_type': 'OLD_ENGINES_CLEANUP',
        'moved_files': moved_files,
        'failed_files': failed_files,
        'cleanup_directory': str(cleanup_dir),
        'remaining_files': [
            'ULTIMATE_COSMIC_STRING_ENGINE.py',
            'README.md',
            'AGENT_READ_FIRST.md',
            'setup.py'
        ]
    }
    
    # Save cleanup summary
    summary_file = Path("CLEANUP_OLD_ENGINES_SUMMARY.json")
    with open(summary_file, 'w') as f:
        json.dump(cleanup_summary, f, indent=2)
    
    logger.info("🎯 CLEANUP SUMMARY:")
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
    
    return cleanup_summary

if __name__ == "__main__":
    from datetime import datetime
    import json
    
    summary = cleanup_old_engines()
    
    print("\n✅ OLD ENGINES CLEANUP COMPLETE!")
    print(f"📁 Moved {len(summary['moved_files'])} files to cleanup/old_engines/")
    print("🎯 Only ULTIMATE_COSMIC_STRING_ENGINE.py remains in root")
    print("🎯 All working systems condensed into ultimate engine")
