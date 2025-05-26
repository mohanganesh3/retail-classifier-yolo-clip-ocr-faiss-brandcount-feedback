#!/usr/bin/env python3
"""
Build vector database from prototype images with combined image + OCR embeddings
"""

import os
import sys

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.append('.')

from src.classification.prototype_builder import PrototypeBuilder
from src.utils.config import ensure_directories
import pandas as pd

def check_prototype_structure():
    """Check if prototype images are properly organized"""
    prototype_dir = 'data/prototypes'
    meta_csv = 'data/meta.csv'
    
    print("🔍 Checking prototype structure...")
    
    # Check if directories exist
    if not os.path.exists(prototype_dir):
        print(f"❌ Prototype directory not found: {prototype_dir}")
        return False
    
    if not os.path.exists(meta_csv):
        print(f"❌ Metadata file not found: {meta_csv}")
        print("Creating sample metadata file...")
        create_sample_metadata()
    
    # Check SKU directories
    sku_dirs = [d for d in os.listdir(prototype_dir) if os.path.isdir(os.path.join(prototype_dir, d))]
    
    if not sku_dirs:
        print(f"❌ No SKU directories found in {prototype_dir}")
        return False
    
    print(f"✅ Found {len(sku_dirs)} SKU directories:")
    
    total_images = 0
    for sku_dir in sku_dirs:
        sku_path = os.path.join(prototype_dir, sku_dir)
        images = [f for f in os.listdir(sku_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        print(f"   📁 {sku_dir}: {len(images)} images")
        
        # Show first few image names
        if images:
            for img in images[:3]:
                print(f"      📸 {img}")
            if len(images) > 3:
                print(f"      ... and {len(images) - 3} more")
    
    print(f"📊 Total images: {total_images}")
    return total_images > 0

def create_sample_metadata():
    """Create metadata CSV from existing SKU directories"""
    prototype_dir = 'data/prototypes'
    
    if not os.path.exists(prototype_dir):
        return
    
    sku_dirs = [d for d in os.listdir(prototype_dir) if os.path.isdir(os.path.join(prototype_dir, d))]
    
    metadata = []
    for sku_dir in sku_dirs:
        # Try to infer product info from directory name
        if 'coca' in sku_dir.lower() or 'coke' in sku_dir.lower():
            brand, name = 'Coca Cola', 'Coca Cola Classic'
        elif 'pepsi' in sku_dir.lower():
            brand, name = 'Pepsi', 'Pepsi Cola'
        elif 'sprite' in sku_dir.lower():
            brand, name = 'Coca Cola', 'Sprite'
        elif 'fanta' in sku_dir.lower():
            brand, name = 'Coca Cola', 'Fanta Orange'
        elif 'tango' in sku_dir.lower():
            brand, name = 'Tango', 'Tango Orange'
        else:
            brand, name = 'Unknown', f'Product {sku_dir}'
        
        metadata.append({
            'SKU_ID': sku_dir,
            'name': name,
            'brand': brand,
            'flavor': 'Original',
            'category': 'Beverages'
        })
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv('data/meta.csv', index=False)
    print(f"✅ Created metadata for {len(metadata)} SKUs")

def build_vector_database():
    """Build the vector database with combined embeddings"""
    print("🚀 BUILDING VECTOR DATABASE WITH COMBINED EMBEDDINGS")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check prototype structure
    if not check_prototype_structure():
        print("\n❌ Prototype structure check failed!")
        print("\nPlease ensure:")
        print("1. Images are in: data/prototypes/SKU_001/, data/prototypes/SKU_002/, etc.")
        print("2. Each SKU directory contains 3-10 images")
        print("3. Images are in JPG/PNG format")
        return False
    
    print(f"\n🏗️ Building vector database...")
    
    try:
        # Initialize prototype builder
        builder = PrototypeBuilder()
        
        # Build prototypes with combined embeddings
        index, labels = builder.build_prototypes('data/prototypes', 'data/meta.csv')
        
        print(f"\n🎉 VECTOR DATABASE BUILT SUCCESSFULLY!")
        print(f"📊 Processed {len(labels)} SKUs")
        print(f"🏷️ SKU Labels: {labels}")
        print(f"📐 Index dimension: {index.d}")
        print(f"💾 Saved to: models/faiss_index/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error building vector database: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_vector_database():
    """Verify the built vector database"""
    print("\n🔍 Verifying vector database...")
    
    index_path = 'models/faiss_index/index.bin'
    labels_path = 'models/faiss_index/labels.pt'
    
    if os.path.exists(index_path) and os.path.exists(labels_path):
        try:
            import torch
            from src.utils.faiss_utils import load_faiss_index
            
            index = load_faiss_index(index_path)
            labels = torch.load(labels_path)
            
            print(f"✅ Vector database verified!")
            print(f"   📊 Index dimension: {index.d}")
            print(f"   🏷️ Number of SKUs: {len(labels)}")
            print(f"   📦 SKU IDs: {labels}")
            
            return True
            
        except Exception as e:
            print(f"❌ Vector database verification failed: {e}")
            return False
    else:
        print(f"❌ Vector database files not found")
        return False

def main():
    print("🎯 VECTOR DATABASE BUILDER")
    print("=" * 40)
    
    # Step 1: Build vector database
    success = build_vector_database()
    
    if success:
        # Step 2: Verify database
        verify_vector_database()
        
        print(f"\n🎉 READY TO USE!")
        print(f"\nNext steps:")
        print(f"1. Run: streamlit run app.py")
        print(f"2. Upload your shelf image")
        print(f"3. The system will use semantic search for classification")
        
    else:
        print(f"\n❌ Vector database build failed")
        print(f"\nPlease check your prototype images and try again")

if __name__ == "__main__":
    main()
