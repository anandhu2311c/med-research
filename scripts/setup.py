#!/usr/bin/env python3
"""Setup script for Medical Literature Assistant."""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {cmd}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return None

def main():
    """Main setup function."""
    print("🚀 Setting up Medical Literature Assistant...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("📝 Creating .env file from template...")
        subprocess.run("copy .env.example .env", shell=True)
        print("⚠️  Please edit .env file with your API keys before continuing")
        return
    
    # Install Python dependencies
    print("📦 Installing Python dependencies...")
    run_command("pip install -r requirements.txt")
    
    # Setup frontend
    print("🎨 Setting up frontend...")
    if Path("frontend").exists():
        run_command("npm install", cwd="frontend")
    
    # Create necessary directories
    print("📁 Creating directories...")
    Path("reports").mkdir(exist_ok=True)
    
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: uvicorn app.main:app --reload")
    print("3. In another terminal: cd frontend && npm run dev")

if __name__ == "__main__":
    main()