#!/usr/bin/env python3
"""Verify API setup for Medical Literature Assistant."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_env_file():
    """Check if .env file exists and load it."""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("Run: copy .env.example .env")
        print("Then edit .env with your API keys")
        return False
    
    load_dotenv()
    print("✅ .env file found and loaded")
    return True

def verify_groq():
    """Verify Groq API connection."""
    try:
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found in .env")
            return False
        
        if not api_key.startswith("gsk_"):
            print("❌ GROQ_API_KEY format invalid (should start with 'gsk_')")
            return False
        
        # Test connection
        llm = ChatGroq(model_name="llama-3.1-70b-versatile", api_key=api_key)
        response = llm.invoke("Hello")
        
        print("✅ Groq API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return False

def verify_pinecone():
    """Verify Pinecone connection."""
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV")
        
        if not api_key:
            print("❌ PINECONE_API_KEY not found in .env")
            return False
        
        if not env:
            print("❌ PINECONE_ENV not found in .env")
            return False
        
        # Test connection
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        
        # Check if our index exists
        index_name = "medlit-embeddings"
        if index_name not in [idx.name for idx in indexes]:
            print(f"⚠️  Index '{index_name}' not found")
            print("Create it in Pinecone console with:")
            print("- Name: medlit-embeddings")
            print("- Dimensions: 384")
            print("- Metric: cosine")
            return False
        
        print("✅ Pinecone connection and index verified")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        return False

def verify_langfuse():
    """Verify Langfuse connection."""
    try:
        from langfuse import Langfuse
        
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST")
        
        if not public_key:
            print("❌ LANGFUSE_PUBLIC_KEY not found in .env")
            return False
        
        if not secret_key:
            print("❌ LANGFUSE_SECRET_KEY not found in .env")
            return False
        
        if not host:
            print("❌ LANGFUSE_HOST not found in .env")
            return False
        
        # Test connection
        lf = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        
        # Create a test trace
        trace = lf.trace(name="setup_verification")
        trace.update(output={"status": "success"})
        
        print("✅ Langfuse connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Langfuse error: {e}")
        return False

def verify_dependencies():
    """Verify required Python packages are installed."""
    required_packages = [
        "fastapi",
        "langchain",
        "langgraph", 
        "langchain_groq",
        "pinecone-client",
        "sentence-transformers",
        "langfuse"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def main():
    """Main verification function."""
    print("🔍 Verifying Medical Literature Assistant setup...\n")
    
    checks = [
        ("Environment file", check_env_file),
        ("Python dependencies", verify_dependencies),
        ("Groq API", verify_groq),
        ("Pinecone", verify_pinecone),
        ("Langfuse", verify_langfuse)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"Checking {name}...")
        result = check_func()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run backend: uvicorn app.main:app --reload")
        print("2. Run frontend: cd frontend && npm run dev")
        print("3. Open: http://localhost:5173")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("See docs/api-setup.md for detailed setup instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()