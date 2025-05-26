#!/usr/bin/env python3
"""
Check prototype image structure and show what's available
"""

import os
from PIL import Image

def check_prototype_images():
    """Check and display prototype image structure"""
    print("📁 PROTOTYPE IMAGE STRUCTURE CHECK")
    print("=" * 50)
    
    prototype_dir = 'data/prototypes'
    
    if not os.path.exists(prototype_dir):
        print(f"❌ Prototype directory not found: {prototype_dir}")
        print(f"\nPlease create the directory and add your images like:")
        print(f"data/prototypes/SKU_001/image1.jpg")
        print(f"data/prototypes/SKU_002/image1.jpg")
        return False
    
    # Get all SKU directories
    sku_dirs = []
    for item in os.listdir(prototype_dir):
        item_path = os.path.join(prototype_dir, item)
        if os.path.isdir(item_path):
            sku_dirs.append(item)
    
    if not sku_dirs:
        print(f"❌ No SKU directories found in {prototype_dir}")
        return False
    
    print(f"✅ Found {len(sku_dirs)} SKU directories:")
    print()
    
    total_images = 0
    valid_skus = 0
    
    for sku_dir in sorted(sku_dirs):
        sku_path = os.path.join(prototype_dir, sku_dir)
        
        # Get image files
        image_files = []
        for file in os.listdir(sku_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(file)
        
        print(f"📦 {sku_dir}:")
        print(f"   📸 Images: {len(image_files)}")
        
        if image_files:
            valid_skus += 1
            total_images += len(image_files)
            
            # Show image details
            for img_file in image_files[:5]:  # Show first 5
                img_path = os.path.join(sku_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        print(f"      ✅ {img_file} ({img.size[0]}x{img.size[1]})")
                except Exception as e:
                    print(f"      ❌ {img_file} (corrupted: {e})")
            
            if len(image_files) > 5:
                print(f"      ... and {len(image_files) - 5} more images")
        else:
            print(f"   ❌ No images found")
        
        print()
    
    print(f"📊 SUMMARY:")
    print(f"   Valid SKUs: {valid_skus}")
    print(f"   Total images: {total_images}")
    print(f"   Average images per SKU: {total_images/valid_skus:.1f}" if valid_skus > 0 else "   No valid SKUs")
    
    if valid_skus > 0:
        print(f"\n✅ Ready to build vector database!")
        print(f"Run: python build_vector_database.py")
    else:
        print(f"\n❌ No valid prototype images found")
        print(f"Please add images to SKU directories")
    
    return valid_skus > 0

if __name__ == "__main__":
    check_prototype_images()
