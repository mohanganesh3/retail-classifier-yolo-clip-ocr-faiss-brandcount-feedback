#!/usr/bin/env python3
"""
Run Streamlit with OpenMP fix
"""

import os
import sys
import subprocess

def run_streamlit_safe():
    """Run Streamlit with OpenMP environment fixes"""
    print("🚀 STARTING STREAMLIT WITH OPENMP FIX")
    print("=" * 45)
    
    # Set environment variables
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    env['OMP_NUM_THREADS'] = '1'
    
    print("✅ Set OpenMP environment variables")
    print("✅ Starting Streamlit...")
    
    try:
        # Run streamlit with fixed environment
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py'
        ], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit stopped by user")
    except Exception as e:
        print(f"\n❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    run_streamlit_safe()
