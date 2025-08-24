#!/usr/bin/env python3
"""Run script for Medical Literature Assistant."""

import subprocess
import sys
import time
from pathlib import Path

def run_backend():
    """Start the FastAPI backend."""
    print("🚀 Starting backend server...")
    subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "app.main:app", "--reload", "--port", "8000"
    ])

def run_frontend():
    """Start the React frontend."""
    if Path("frontend").exists():
        print("🎨 Starting frontend server...")
        subprocess.Popen([
            "npm", "run", "dev"
        ], cwd="frontend")

def main():
    """Main run function."""
    print("🏃 Starting Medical Literature Assistant...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ .env file not found. Run setup.py first.")
        return
    
    try:
        run_backend()
        time.sleep(2)  # Give backend time to start
        run_frontend()
        
        print("\n✅ Services started!")
        print("Backend: http://localhost:8000")
        print("Frontend: http://localhost:5173")
        print("API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all services")
        
        # Keep script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")

if __name__ == "__main__":
    main()