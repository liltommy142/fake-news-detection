#!/usr/bin/env python3
"""
Main script to run the fake news detection pipeline.
"""

import subprocess
import sys

def run_script(script_path: str):
    """
    Run a Python script.

    Args:
        script_path: Path to the script.
    """
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}: {result.stderr}")
    else:
        print(f"Successfully ran {script_path}")

if __name__ == "__main__":
    scripts = [
        "src/preprocessing.py",
        "src/feature_engineering.py",
        "src/model.py"
    ]

    for script in scripts:
        run_script(script)