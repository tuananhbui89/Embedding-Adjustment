#!/usr/bin/env python3
"""
Script to download FLUX.1-dev model locally for offline inference
Run this once when you have internet connection.
"""

import os
import argparse
from huggingface_hub import snapshot_download
import torch

def download_flux_model(local_dir="./models/flux-1-dev", use_auth_token=None):
    """
    Download FLUX.1-dev model to local directory
    
    Args:
        local_dir: Local directory to save the model
        use_auth_token: HF token if needed (FLUX is gated model)
    """
    
    print("=" * 60)
    print("🔄 Downloading FLUX.1-dev model locally...")
    print(f"📁 Target directory: {local_dir}")
    print("=" * 60)
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download the model
        print("⬇️  Downloading model files...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=local_dir,
            token=use_auth_token,
            ignore_patterns=["*.git*", "README.md", ".gitattributes"]
        )
        
        print("✅ Model downloaded successfully!")
        print(f"📁 Model saved to: {os.path.abspath(local_dir)}")
        
        # Verify key files exist
        key_files = [
            "model_index.json",
            "scheduler/scheduler_config.json", 
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "transformer/config.json",
            "vae/config.json"
        ]
        
        print("\n🔍 Verifying downloaded files...")
        missing_files = []
        for file in key_files:
            file_path = os.path.join(local_dir, file)
            if os.path.exists(file_path):
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - MISSING")
                missing_files.append(file)
        
        if missing_files:
            print(f"\n⚠️  Warning: {len(missing_files)} files are missing!")
            print("This might cause issues during loading.")
        else:
            print("\n🎉 All key files verified successfully!")
            
        print(f"\n💡 To use this model, update your code:")
        print(f'   base_path = "{os.path.abspath(local_dir)}"')
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\n💡 This might be a gated model requiring authentication.")
            print("Try:")
            print("1. Login to HuggingFace: huggingface-cli login")
            print("2. Or provide token: --token YOUR_HF_TOKEN")
            print("3. Make sure you have access to FLUX.1-dev on HuggingFace")
            
        return False

def main():
    parser = argparse.ArgumentParser(description="Download FLUX.1-dev model locally")
    parser.add_argument("--local_dir", type=str, default="./models/flux-1-dev",
                       help="Local directory to save the model")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace token (if model is gated)")
    
    args = parser.parse_args()
    
    success = download_flux_model(args.local_dir, args.token)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("💡 Next steps:")
        print("1. Update your inference scripts to use the local path")
        print("2. You can now run inference without internet connection")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ DOWNLOAD FAILED!")
        print("Please check the error messages above and try again.")
        print("=" * 60)

if __name__ == "__main__":
    main() 