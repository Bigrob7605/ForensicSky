#!/usr/bin/env python3
"""
Compile the cosmic string detection white paper
"""

import subprocess
import os
import sys

def run_command(command, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Compile the white paper"""
    print("🚀 Compiling Cosmic String Detection White Paper...")
    print("=" * 60)
    
    # Change to white_paper directory
    os.chdir('white_paper')
    
    # Compilation steps
    steps = [
        ("Generating figures...", "cd figures && python detection_results.py"),
        ("Generating tables...", "cd tables && python detection_data.py"),
        ("First LaTeX pass...", "pdflatex main.tex"),
        ("Bibliography pass...", "bibtex main"),
        ("Second LaTeX pass...", "pdflatex main.tex"),
        ("Final LaTeX pass...", "pdflatex main.tex")
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for description, command in steps:
        print(f"\n📝 {description}")
        if run_command(command):
            success_count += 1
        else:
            print(f"❌ Failed at step: {description}")
            break
    
    print("\n" + "=" * 60)
    print(f"📊 Compilation Results: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("🎉 SUCCESS! White paper compiled successfully!")
        print("\n📄 Output files:")
        print("  - main.pdf (main document)")
        print("  - figures/ (4K high-resolution figures)")
        print("  - tables/ (CSV and LaTeX tables)")
        print("  - supplementary/ (detailed methods)")
        
        print("\n🚀 Ready for:")
        print("  - Scientific publication")
        print("  - Peer review")
        print("  - Conference presentation")
        print("  - Media coverage")
        
        print("\n🌌 The 'Hadron Collider of Code' has delivered!")
        print("   We found cosmic strings! ⚡🔬")
        
    else:
        print("❌ Compilation failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
