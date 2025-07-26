#!/usr/bin/env python3
"""
Download SAM model weights automatically
"""

import os
import urllib.request
from pathlib import Path

def download_sam_model():
    """Download the SAM model if it doesn't exist"""
    model_path = 'models/sam_vit_h_4b8939.pth'
    model_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    
    if os.path.exists(model_path):
        print(f"✅ SAM model already exists at {model_path}")
        return True
    
    print(f"📥 Downloading SAM model to {model_path}...")
    print(f"🌐 URL: {model_url}")
    print("⏳ This may take several minutes (file is ~2.4GB)")
    
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100
            print(f"\r📊 Progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='')
        
        urllib.request.urlretrieve(model_url, model_path, show_progress)
        print(f"\n✅ Successfully downloaded SAM model to {model_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading SAM model: {e}")
        return False

if __name__ == "__main__":
    success = download_sam_model()
    if success:
        print("🎉 Ready to run the food detection system!")
    else:
        print("❌ Please download the model manually")