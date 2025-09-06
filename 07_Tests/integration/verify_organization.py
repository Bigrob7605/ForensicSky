#!/usr/bin/env python3
"""
Verify Project Organization
Check that all files are in their correct locations
"""

import os
from pathlib import Path
import sys

def verify_project_organization():
    """Verify that the project is properly organized"""
    print("🔍 VERIFYING PROJECT ORGANIZATION")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent.parent
    errors = []
    warnings = []
    
    # Check main directories
    required_dirs = [
        "01_Core_Engine",
        "02_Data",
        "03_Analysis", 
        "04_Results",
        "05_Visualizations",
        "06_Documentation",
        "07_Tests",
        "08_Archive",
        "09_Config",
        "10_Notebooks"
    ]
    
    print("\n📁 Checking main directories...")
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name}")
        else:
            error_msg = f"❌ {dir_name} - Missing or not a directory"
            errors.append(error_msg)
            print(error_msg)
    
    # Check core engine
    print("\n🚀 Checking core engine...")
    core_engine_path = project_root / "01_Core_Engine" / "Core_ForensicSky_V1.py"
    if core_engine_path.exists() and core_engine_path.is_file():
        print("✅ Core_ForensicSky_V1.py")
    else:
        error_msg = "❌ Core_ForensicSky_V1.py - Missing or not a file"
        errors.append(error_msg)
        print(error_msg)
    
    # Check configuration files
    print("\n⚙️ Checking configuration files...")
    config_files = [
        "09_Config/requirements.txt",
        "09_Config/setup.py",
        "09_Config/settings/project_config.py"
    ]
    
    for config_file in config_files:
        file_path = project_root / config_file
        if file_path.exists() and file_path.is_file():
            print(f"✅ {config_file}")
        else:
            error_msg = f"❌ {config_file} - Missing or not a file"
            errors.append(error_msg)
            print(error_msg)
    
    # Check documentation
    print("\n📚 Checking documentation...")
    doc_files = [
        "README.md",
        "PROJECT_STRUCTURE.md"
    ]
    
    for doc_file in doc_files:
        file_path = project_root / doc_file
        if file_path.exists() and file_path.is_file():
            print(f"✅ {doc_file}")
        else:
            error_msg = f"❌ {doc_file} - Missing or not a file"
            errors.append(error_msg)
            print(error_msg)
    
    # Check subdirectories
    print("\n📂 Checking subdirectories...")
    subdirs = [
        "02_Data/ipta_dr2",
        "02_Data/processed",
        "02_Data/raw",
        "03_Analysis/correlation",
        "03_Analysis/spectral",
        "03_Analysis/ml",
        "04_Results/json",
        "04_Results/npz",
        "04_Results/logs",
        "05_Visualizations/plots",
        "05_Visualizations/figures",
        "06_Documentation/api",
        "06_Documentation/guides",
        "07_Tests/unit",
        "07_Tests/integration",
        "08_Archive/old_engines",
        "08_Archive/backup",
        "09_Config/settings",
        "09_Config/parameters"
    ]
    
    for subdir in subdirs:
        dir_path = project_root / subdir
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {subdir}")
        else:
            warning_msg = f"⚠️ {subdir} - Missing or not a directory"
            warnings.append(warning_msg)
            print(warning_msg)
    
    # Check for any remaining files in root
    print("\n🔍 Checking for remaining files in root...")
    root_files = [f for f in project_root.iterdir() if f.is_file() and f.name != "verify_organization.py"]
    if root_files:
        print("⚠️ Files remaining in root directory:")
        for file in root_files:
            warning_msg = f"⚠️ {file.name} - Consider moving to appropriate directory"
            warnings.append(warning_msg)
            print(warning_msg)
    else:
        print("✅ No files remaining in root directory")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 ORGANIZATION SUMMARY")
    print("=" * 50)
    
    if errors:
        print(f"❌ ERRORS: {len(errors)}")
        for error in errors:
            print(f"   {error}")
    else:
        print("✅ NO ERRORS")
    
    if warnings:
        print(f"⚠️ WARNINGS: {len(warnings)}")
        for warning in warnings:
            print(f"   {warning}")
    else:
        print("✅ NO WARNINGS")
    
    if not errors and not warnings:
        print("\n🎉 PROJECT ORGANIZATION IS PERFECT!")
        print("🚀 Core ForensicSky V1 is ready for production!")
    elif not errors:
        print("\n✅ PROJECT ORGANIZATION IS GOOD!")
        print("⚠️ Some minor issues to address")
    else:
        print("\n❌ PROJECT ORGANIZATION NEEDS ATTENTION!")
        print("🔧 Please fix the errors above")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = verify_project_organization()
    sys.exit(0 if success else 1)
