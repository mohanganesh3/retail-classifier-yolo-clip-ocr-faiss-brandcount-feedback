#!/usr/bin/env python3
"""
Run Streamlit app with OpenMP fixes and enhanced product counting
"""

import os
import sys
import subprocess

def setup_environment():
    """Setup environment to prevent OpenMP conflicts"""
    # Fix OpenMP conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Optimize for better performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    print("✅ Environment configured for OpenMP safety")

def main():
    print("🚀 STARTING RETAIL SHELF MONITOR (OPENMP SAFE)")
    print("=" * 50)
    
    setup_environment()
    
    try:
        # Run streamlit with fixed environment
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.headless', 'true',
            '--server.enableCORS', 'false'
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
