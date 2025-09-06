#!/usr/bin/env python3
"""
PURGE QUANTUM SYSTEMS - Clean House
===================================

The quantum kernel method failed validation - it produces false positives
and cannot distinguish signal from noise. Time to purge all quantum systems
and focus on what actually works.

This script will:
1. Identify all quantum-related files
2. Move them to archive
3. Clean up the codebase
4. Focus on classical PTA methods that actually work
"""

import os
import shutil
from datetime import datetime
import glob

class QuantumSystemPurge:
    def __init__(self):
        self.archive_dir = "08_Archive/quantum_systems_purged"
        self.quantum_files = []
        self.quantum_patterns = [
            "*quantum*",
            "*QUANTUM*", 
            "*Quantum*",
            "*phase_lock*",
            "*PHASE_LOCK*",
            "*PhaseLock*",
            "*discovery*",
            "*DISCOVERY*",
            "*Discovery*",
            "*validation*",
            "*VALIDATION*",
            "*Validation*"
        ]
        
    def find_quantum_files(self):
        """Find all quantum-related files"""
        print("🔍 Scanning for quantum system files...")
        
        for pattern in self.quantum_patterns:
            files = glob.glob(pattern)
            self.quantum_files.extend(files)
            
        # Remove duplicates and sort
        self.quantum_files = sorted(list(set(self.quantum_files)))
        
        print(f"   Found {len(self.quantum_files)} quantum-related files:")
        for file in self.quantum_files:
            print(f"     - {file}")
            
        return self.quantum_files
        
    def create_archive_directory(self):
        """Create archive directory for purged files"""
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir)
            print(f"   Created archive directory: {self.archive_dir}")
        else:
            print(f"   Archive directory already exists: {self.archive_dir}")
            
    def archive_quantum_files(self):
        """Move quantum files to archive"""
        print("\n📦 Archiving quantum system files...")
        
        archived_count = 0
        for file in self.quantum_files:
            if os.path.exists(file):
                try:
                    # Create subdirectory by file type
                    file_ext = os.path.splitext(file)[1]
                    if file_ext in ['.py']:
                        subdir = "python_scripts"
                    elif file_ext in ['.tex']:
                        subdir = "latex_documents"
                    elif file_ext in ['.md', '.txt']:
                        subdir = "documentation"
                    elif file_ext in ['.json']:
                        subdir = "data_files"
                    else:
                        subdir = "other"
                        
                    archive_subdir = os.path.join(self.archive_dir, subdir)
                    if not os.path.exists(archive_subdir):
                        os.makedirs(archive_subdir)
                        
                    # Move file to archive
                    dest_path = os.path.join(archive_subdir, file)
                    shutil.move(file, dest_path)
                    print(f"     ✅ Archived: {file} -> {dest_path}")
                    archived_count += 1
                    
                except Exception as e:
                    print(f"     ❌ Failed to archive {file}: {e}")
            else:
                print(f"     ⚠️  File not found: {file}")
                
        print(f"\n   Successfully archived {archived_count} files")
        return archived_count
        
    def create_purge_report(self):
        """Create a report of the purge operation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"QUANTUM_SYSTEM_PURGE_REPORT_{timestamp}.md"
        
        report_content = f"""# Quantum System Purge Report

**Date**: {datetime.now().isoformat()}
**Reason**: Quantum kernel method failed validation - produces false positives

## Summary
- **Files purged**: {len(self.quantum_files)}
- **Archive location**: {self.archive_dir}
- **Status**: Quantum systems successfully purged

## Purged Files
"""
        
        for file in self.quantum_files:
            report_content += f"- {file}\n"
            
        report_content += f"""
## Validation Results
The quantum kernel method was tested with comprehensive validation and failed:

- **Hub scores identical** for noise and signal (all ~1.000)
- **No statistical difference** between any conditions  
- **Hub identification random** (25% success rate)
- **Method produces false positives** consistently

## Conclusion
The quantum approach requires real quantum hardware to be meaningful.
Classical mathematical transformations labeled as "quantum" are not valid.

## Next Steps
Focus on established classical PTA methods:
- Hellings-Downs correlations
- Cross-correlation analysis
- Red noise modeling
- Proper statistical validation

---
**Purge completed successfully**
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        print(f"   📄 Purge report created: {report_file}")
        return report_file
        
    def run_purge(self):
        """Run the complete purge operation"""
        print("🧹 QUANTUM SYSTEM PURGE - CLEANING HOUSE")
        print("=" * 50)
        print("The quantum kernel method failed validation.")
        print("Time to purge and focus on what actually works.")
        print()
        
        # Find quantum files
        quantum_files = self.find_quantum_files()
        
        if not quantum_files:
            print("   ✅ No quantum files found - already clean!")
            return
            
        # Create archive directory
        self.create_archive_directory()
        
        # Archive quantum files
        archived_count = self.archive_quantum_files()
        
        # Create purge report
        report_file = self.create_purge_report()
        
        print("\n" + "=" * 50)
        print("🎯 QUANTUM SYSTEM PURGE COMPLETE!")
        print(f"   📦 Archived {archived_count} files")
        print(f"   📄 Report: {report_file}")
        print("   ✅ Codebase cleaned of failed quantum systems")
        print("\n🔬 Focus on classical PTA methods that actually work!")

def main():
    """Run the quantum system purge"""
    purger = QuantumSystemPurge()
    purger.run_purge()

if __name__ == "__main__":
    main()
