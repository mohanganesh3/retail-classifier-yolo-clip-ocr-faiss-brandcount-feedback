#!/usr/bin/env python3
"""
Fix OpenMP library conflicts on macOS
"""

import os
import sys

def fix_openmp_conflict():
    """Fix OpenMP library conflicts"""
    print("🔧 FIXING OPENMP LIBRARY CONFLICTS")
    print("=" * 40)
    
    # Set environment variable to allow duplicate OpenMP libraries
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'  # Limit to single thread to avoid conflicts
    
    print("✅ Set KMP_DUPLICATE_LIB_OK=TRUE")
    print("✅ Set OMP_NUM_THREADS=1")
    
    # Also set for current session
    import subprocess
    
    # For bash/zsh
    bashrc_content = """
# Fix OpenMP conflicts for retail shelf monitor
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
"""
    
    # Try to append to shell config
    shell_configs = ['~/.bashrc', '~/.zshrc', '~/.bash_profile']
    
    for config_file in shell_configs:
        config_path = os.path.expanduser(config_file)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    content = f.read()
                
                if 'KMP_DUPLICATE_LIB_OK' not in content:
                    with open(config_path, 'a') as f:
                        f.write(bashrc_content)
                    print(f"✅ Added OpenMP fix to {config_file}")
                else:
                    print(f"✅ OpenMP fix already in {config_file}")
            except Exception as e:
                print(f"⚠️ Could not modify {config_file}: {e}")
    
    print("\n🎯 OpenMP conflict fix applied!")
    print("\nTo make this permanent, run:")
    print("export KMP_DUPLICATE_LIB_OK=TRUE")
    print("export OMP_NUM_THREADS=1")
    
    return True

if __name__ == "__main__":
    fix_openmp_conflict()
