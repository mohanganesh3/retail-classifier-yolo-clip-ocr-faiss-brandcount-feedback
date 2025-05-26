#!/usr/bin/env python3
"""
Fix classification dimension mismatch by rebuilding FAISS index with correct dimensions
"""

import os
import sys
import torch
import numpy as np
sys.path.append('.')

from src.classification.prototype_builder import PrototypeBuilder
from src.utils.config import ensure_directories
from src.utils.faiss_utils import load_faiss_index

def fix_classification_dimensions():
    """Fix classification by rebuilding FAISS index with correct dimensions"""
    print("🔧 FIXING CLASSIFICATION DIMENSION MISMATCH")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check current FAISS index
    index_path = 'models/faiss_index/index.bin'
    labels_path = 'models/faiss_index/labels.pt'
    
    if os.path.exists(index_path):
        try:
            old_index = load_faiss_index(index_path)
            old_labels = torch.load(labels_path)
            print(f"📊 Current index: {old_index.d} dimensions, {len(old_labels)} SKUs")
            
            # Remove old index
            os.remove(index_path)
            os.remove(labels_path)
            print("🗑️ Removed old incompatible index")
            
        except Exception as e:
            print(f"⚠️ Could not load old index: {e}")
    
    # Check if prototype data exists
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
        print("❌ No SKU directories found")
        print("Please run: python scripts/generate_sample_data.py")
        return False
    
    # Rebuild prototypes with dimension fixing
    print("\n🏗️ Rebuilding FAISS index with correct dimensions...")
    try:
        builder = PrototypeBuilder()
        index, labels = builder.build_prototypes(prototype_dir, meta_csv)
        
        print(f"✅ Successfully rebuilt FAISS index!")
        print(f"📊 New index dimensions: {index.d}")
        print(f"📦 SKUs processed: {len(labels)}")
        print(f"🏷️ SKU labels: {labels}")
        
        # Verify the new index works
        print("\n🔍 Verifying new index...")
        test_embedding = np.random.random((1, index.d)).astype(np.float32)
        distances, indices = index.search(test_embedding, 1)
        print(f"✅ Index verification successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error rebuilding prototypes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = fix_classification_dimensions()
    
    if success:
        print("\n🎉 CLASSIFICATION DIMENSIONS FIXED!")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Upload your shelf image")
        print("3. Classification should now work with correct dimensions")
        print("\nThe system will now:")
        print("✅ Detect cans with YOLOv7/YOLOv8")
        print("✅ Extract OCR text from can labels")
        print("✅ Classify products with correct embedding dimensions")
        print("✅ Count SKUs accurately")
    else:
        print("\n❌ Failed to fix classification dimensions")
        print("Please check your prototype data and try again")
        print("Run: python scripts/generate_sample_data.py to create sample data")

if __name__ == "__main__":
    main()
