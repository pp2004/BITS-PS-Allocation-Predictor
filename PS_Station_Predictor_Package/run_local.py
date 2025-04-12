#!/usr/bin/env python3
"""
PS Station Predictor - Helper script to run the application locally
This script verifies dependencies and starts the Streamlit application.
"""

import importlib
import subprocess
import sys
import os

def check_dependency(module_name):
    """Check if a Python module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Main function to run the application."""
    # List of required dependencies
    dependencies = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "sklearn",
        "openpyxl"
    ]
    
    # Check if all dependencies are installed
    missing_deps = [dep for dep in dependencies if not check_dependency(dep)]
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        install = input("Would you like to install them now? (y/n): ")
        
        if install.lower() == 'y':
            print("Installing missing dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_deps)
            print("Dependencies installed successfully.")
        else:
            print("Please install the required dependencies and try again.")
            print("You can install them manually with:")
            print(f"pip install {' '.join(missing_deps)}")
            return
    
    # Create .streamlit directory and config.toml if they don't exist
    if not os.path.exists(".streamlit"):
        os.makedirs(".streamlit")
    
    config_path = ".streamlit/config.toml"
    if not os.path.exists(config_path):
        with open(config_path, "w") as config_file:
            config_file.write("""[server]
headless = false
port = 8501
""")
    
    # Run the Streamlit application
    print("Starting PS Station Predictor application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()