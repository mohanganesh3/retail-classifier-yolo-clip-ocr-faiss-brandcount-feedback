#!/usr/bin/env python3
"""
Debug the complete pipeline step by step
"""

import sys
import os
sys.path.append('.')

from src.pipeline.retail_pipeline import RetailShelfPipeline
from src.utils.config import ensure_directories

def debug_pipeline(image_path):
    """Debug pipeline with detailed logging"""
    print("🐛 Debugging Retail Pipeline")
    print("=" * 40)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Ensure directories
    ensure_directories()
    
    # Initialize pipeline with debug info
    print("🔧 Initializing pipeline...")
    pipeline = RetailShelfPipeline(use_ocr=True)
    
    print(f"   OCR enabled: {pipeline.use_ocr}")
    print(f"   OCR processor: {pipeline.ocr_processor is not None}")
    print(f"   Classifier available: {pipeline.classifier_available}")
    
    # Process image
    print(f"\n📸 Processing: {image_path}")
    results = pipeline.process_image(image_path)
    
    # Debug results
    print(f"\n📊 Pipeline Results:")
    print(f"   Total products: {results.get('total_products', 0)}")
    print(f"   Detections: {len(results.get('detection_data', []))}")
    print(f"   Classifications: {len(results.get('classifications', []))}")
    print(f"   SKU counts: {len(results.get('sku_counts', {}))}")
    print(f"   OCR enabled: {results.get('ocr_enabled', False)}")
    print(f"   Classifier available: {results.get('classifier_available', False)}")
    
    # Show detection details
    if results.get('detection_data'):
        print(f"\n🔍 Detection Details:")
        for i, det in enumerate(results['detection_data'][:3]):
            print(f"   {i+1}. Confidence: {det.get('confidence', 0):.3f}")
            print(f"      Class: {det.get('class_name', 'unknown')}")
            print(f"      Crop: {os.path.basename(det.get('crop_path', ''))}")
    
    return results

def main():
    if len(sys.argv) > 1:
        debug_pipeline(sys.argv[1])
    else:
        # Look for test images
        test_images = [
            'temp/sample_shelf.jpg',
            'sample_shelf.jpg'
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                debug_pipeline(img_path)
                break
        else:
            print("❌ No test images found")
            print("Usage: python scripts/debug_pipeline.py <image_path>")

if __name__ == "__main__":
    main()
