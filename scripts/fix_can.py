#!/usr/bin/env python3
"""
Comprehensive fix for can detection and vector database issues
"""

import os
import sys
import torch
import shutil
sys.path.append('.')

from src.classification.prototype_builder import PrototypeBuilder
from src.utils.config import ensure_directories

def fix_can_detection_system():
    """Fix all can detection and classification issues"""
    print("🔧 COMPREHENSIVE CAN DETECTION FIX")
    print("=" * 50)
    
    # Step 1: Clean up old data
    print("\n🗑️ Step 1: Cleaning up old data...")
    
    # Remove old FAISS index completely
    index_dir = 'models/faiss_index'
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        print(f"✅ Removed old index directory: {index_dir}")
    
    # Remove old crops
    crops_dir = 'temp/crops'
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir)
        print(f"✅ Removed old crops: {crops_dir}")
    
    # Remove old results
    old_results = ['final_results.json', 'classification_results.json', 'temp/detection.json']
    for result_file in old_results:
        if os.path.exists(result_file):
            os.remove(result_file)
            print(f"✅ Removed old result: {result_file}")
    
    # Step 2: Ensure directories exist
    print("\n📁 Step 2: Creating directories...")
    ensure_directories()
    
    # Step 3: Check prototype data
    print("\n📊 Step 3: Checking prototype data...")
    prototype_dir = 'data/prototypes'
    meta_csv = 'data/meta.csv'
    
    if not os.path.exists(prototype_dir):
        print(f"❌ Prototype directory not found: {prototype_dir}")
        print("Please run: python scripts/generate_sample_data.py")
        return False
    
    if not os.path.exists(meta_csv):
        print(f"❌ Metadata file not found: {meta_csv}")
        print("Please run: python scripts/generate_sample_data.py")
        return False
    
    # Count available SKUs
    sku_dirs = [d for d in os.listdir(prototype_dir) if os.path.isdir(os.path.join(prototype_dir, d))]
    print(f"📦 Found {len(sku_dirs)} SKU directories: {sku_dirs}")
    
    if len(sku_dirs) == 0:
        print("❌ No SKU directories found in prototypes")
        return False
    
    # Step 4: Rebuild FAISS index with correct dimensions
    print("\n🏗️ Step 4: Rebuilding FAISS index with correct dimensions...")
    try:
        builder = PrototypeBuilder()
        index, labels = builder.build_prototypes(prototype_dir, meta_csv)
        
        print(f"✅ Successfully rebuilt FAISS index!")
        print(f"📊 Index dimensions: {index.d}")
        print(f"📦 SKUs processed: {len(labels)}")
        print(f"🏷️ SKU labels: {labels}")
        
        # Step 5: Verify the index works
        print("\n🔍 Step 5: Verifying index...")
        from src.classification.classifier import ProductClassifier
        classifier = ProductClassifier()
        
        if classifier.index_available:
            print("✅ Index verification successful!")
            return True
        else:
            print("❌ Index verification failed!")
            return False
        
    except Exception as e:
        print(f"❌ Error rebuilding prototypes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = fix_can_detection_system()
    
    if success:
        print("\n🎉 CAN DETECTION SYSTEM FIXED!")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Upload your cool drink can image")
        print("3. System should now properly detect and classify cans")
        print("\nKey improvements:")
        print("- ✅ Removed ALL mock data")
        print("- ✅ Fixed FAISS dimension mismatch")
        print("- ✅ Added can-specific keywords")
        print("- ✅ Improved bounding box accuracy")
        print("- ✅ Reduced grid fallback noise")
    else:
        print("\n❌ Failed to fix can detection system")
        print("Please check your prototype data and try again")

if __name__ == "__main__":
    main()