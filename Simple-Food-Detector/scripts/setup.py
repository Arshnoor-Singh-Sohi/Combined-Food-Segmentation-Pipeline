#!/usr/bin/env python3
"""
Setup script for the Simple Food Detection project
This script helps you get everything configured properly
"""

import os
import sys
import urllib.request
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/uploads',
        'data/crops', 
        'data/results',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env and add your OpenAI API key")
        else:
            print("❌ .env.example not found")
    else:
        print("✅ .env file already exists")

def check_sam_model():
    """Check if SAM model is downloaded"""
    model_path = 'models/sam_vit_h_4b8939.pth'
    if not os.path.exists(model_path):
        print(f"❌ SAM model not found at {model_path}")
        print("📥 You can download it from:")
        print("   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("   Place it in the models/ directory")
        return False
    else:
        print("✅ SAM model found")
        return True

def main():
    print("🚀 Setting up Simple Food Detection project...")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Check SAM model
    model_exists = check_sam_model()
    
    print("\n" + "="*50)
    print("Setup Summary:")
    print("✅ Directories created")
    print("✅ Environment file prepared" if os.path.exists('.env') else "❌ Environment file missing")
    print("✅ SAM model ready" if model_exists else "❌ SAM model missing")
    
    if not model_exists or not os.path.exists('.env'):
        print("\n⚠️  Action required:")
        if not model_exists:
            print("   1. Download SAM model to models/ directory")
        if not os.path.exists('.env'):
            print("   2. Configure .env file with your OpenAI API key")
    else:
        print("\n🎉 Setup complete! You can now run: streamlit run main.py")

if __name__ == "__main__":
    main()
