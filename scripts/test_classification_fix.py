#!/usr/bin/env python3
"""
Test the classification fix with a sample image
"""

import sys
import os
sys.path.append('.')

from src.pipeline.retail_pipeline import RetailShelfPipeline
from src.utils.config import ensure_directories

def test_classification_fix(image_path=None):
    """Test classification with dimension fixes"""
    print("🧪 TESTING CLASSIFICATION FIX")
    print("=" * 40)
    
    # Use sample image if none provided
    if image_path is None:
        test_images = [
            'temp/sample_shelf.jpg',
            'temp/testttttttttttt.jpg',
            'temp/tester.jpg'
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                image_path = img_path
                break
        
        if image_path is None:
            print("❌ No test images found")
            print("Please provide an image path or run: python scripts/generate_sample_data.py")
            return False
    
    print(f"📸 Testing with image: {image_path}")
    
    # Ensure directories
    ensure_directories()
    
    # Initialize pipeline
    print("🔧 Initializing pipeline...")
    try:
        pipeline = RetailShelfPipeline(use_ocr=True, use_yolov7=True)
        print(f"✅ Pipeline initialized")
        print(f"   Classifier available: {pipeline.classifier_available}")
        
        if pipeline.classifier_available:
            print(f"   FAISS index dimension: {pipeline.classifier.index.d}")
            print(f"   Number of SKUs: {len(pipeline.classifier.labels)}")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Process image
    print(f"\n🚀 Processing image...")
    try:
        results = pipeline.process_image(image_path)
        
        # Check results
        total_detections = len(results.get('detection_data', []))
        total_classifications = len(results.get('classifications', []))
        sku_counts = results.get('sku_counts', {})
        
        print(f"\n📊 Results Summary:")
        print(f"   Detections: {total_detections}")
        print(f"   Classifications: {total_classifications}")
        print(f"   SKU counts: {sku_counts}")
        print(f"   Classifier available: {results.get('classifier_available', False)}")
        
        # Check for dimension errors
        if total_classifications > 0:
            print("✅ Classification working - no dimension errors!")
            return True
        elif total_detections > 0:
            print("⚠️ Detection working but classification failed")
            print("Check if FAISS index needs rebuilding")
            return False
        else:
            print("⚠️ No detections found - check image quality")
            return False
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) > 1:
        success = test_classification_fix(sys.argv[1])
    else:
        success = test_classification_fix()
    
    if success:
        print("\n🎉 Classification fix successful!")
    else:
        print("\n❌ Classification still has issues")
        print("Try running: python scripts/fix_classification_dimensions.py")

if __name__ == "__main__":
    main()
