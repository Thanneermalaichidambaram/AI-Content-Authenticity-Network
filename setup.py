#!/usr/bin/env python3
"""
Setup script for AI Content Authenticity Network
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e.stderr}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║            AI Content Authenticity Network Setup            ║
║                      Installation Script                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Install required packages
    print("📦 Installing Python dependencies...")
    
    # Essential packages first
    essential_packages = [
        "google-cloud-bigquery",
        "pandas",
        "numpy", 
        "streamlit",
        "plotly",
        "requests",
        "python-dotenv",
        "pandas-gbq",
        "db-dtypes"
    ]
    
    for package in essential_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    # Optional packages
    optional_packages = [
        "scikit-learn",
        "matplotlib", 
        "seaborn",
        "nltk",
        "textstat",
        "langdetect",
        "networkx",
        "pillow",
        "opencv-python"
    ]
    
    print("\n📦 Installing optional packages...")
    for package in optional_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Try to install spacy model
    print("\n🧠 Installing language model...")
    run_command("python -m spacy download en_core_web_sm", "Installing spaCy English model")
    
    # Install transformers and torch (these might take longer)
    print("\n🤖 Installing ML libraries...")
    run_command("pip install transformers torch", "Installing transformers and torch")
    
    print("\n✅ Setup completed!")
    print("\n🚀 Next steps:")
    print("1. Configure your .env file with Google Cloud credentials")
    print("2. Run: python main.py setup")
    print("3. Run: python main.py collect --limit 100")
    print("4. Run: python main.py dashboard")

if __name__ == "__main__":
    main()