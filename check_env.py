#!/usr/bin/env python3
"""Quick environment check script."""

import os
from pathlib import Path
from dotenv import load_dotenv

def check_env():
    """Check if .env file exists and has required keys."""
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Run: copy .env.example .env")
        print("Then edit .env with your API keys")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check required keys
    required_keys = [
        "GROQ_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENV",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST"
    ]
    
    missing_keys = []
    for key in required_keys:
        value = os.getenv(key)
        if not value or "your_" in value:
            missing_keys.append(key)
    
    if missing_keys:
        print("❌ Missing or invalid API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease update your .env file with valid API keys")
        print("See docs/api-setup.md for instructions")
        return False
    
    print("✅ Environment variables look good!")
    return True

if __name__ == "__main__":
    check_env()