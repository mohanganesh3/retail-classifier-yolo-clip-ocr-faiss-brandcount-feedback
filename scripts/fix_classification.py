#!/usr/bin/env python3
"""
Fix classification issues by rebuilding FAISS index with correct dimensions
"""

import os
import sys
import torch
sys.path.append('.')

from src.classification.prototype_builder import PrototypeBuilder
from src.utils.config import ensure_directories

def fix_classification():
    """Fix classification by rebuilding FAISS index"""
    print("🔧 Fixing Classification System")
    print("=" * 40)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if prototype data exists
    prototype_dir = 'data/prototypes'
    meta_csv = 'data/meta.csv'
    
    if not os.path.exists(prototype_dir):
        print(f"❌ Prototype directory not found: {prototype_dir}")
        return False
    
    if not os.path.exists(meta_csv):
        print(f"❌ Metadata file not found: {meta_csv}")
        return False
    
    # Count available SKUs
    sku_dirs = [d for d in os.listdir(prototype_dir) if os.path.isdir(os.path.join(prototype_dir, d))]
    print(f"📊 Found {len(sku_dirs)} SKU directories: {sku_dirs}")
    
    # Remove old FAISS index
    old_index = 'models/faiss_index/index.bin'
    old_labels = 'models/faiss_index/labels.pt'
    
    if os.path.exists(old_index):
        os.remove(old_index)
        print(f"🗑️ Removed old index: {old_index}")
    
    if os.path.exists(old_labels):
        os.remove(old_labels)
        print(f"🗑️ Removed old labels: {old_labels}")
    
    # Rebuild prototypes
    print("\n🏗️ Rebuilding FAISS index with correct dimensions...")
    try:
        builder = PrototypeBuilder()
        index, labels = builder.build_prototypes(prototype_dir, meta_csv)
        
        print(f"✅ Successfully rebuilt FAISS index!")
        print(f"📊 Index dimensions: {index.d}")
        print(f"📦 SKUs processed: {len(labels)}")
        print(f"🏷️ SKU labels: {labels}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error rebuilding prototypes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = fix_classification()
    
    if success:
        print("\n🎉 Classification system fixed!")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Upload your shelf image")
        print("3. Classification should now work properly")
    else:
        print("\n❌ Failed to fix classification system")
        print("Please check your prototype data and try again")

if __name__ == "__main__":
    main()
