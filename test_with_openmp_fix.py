#!/usr/bin/env python3
"""
Test the system with OpenMP fixes
"""

import os
import sys

# Fix OpenMP conflicts BEFORE importing any ML libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.append('.')

from src.pipeline.retail_pipeline import RetailShelfPipeline

def test_complete_system():
    """Test the complete system with OpenMP fixes"""
    print("🧪 TESTING COMPLETE SYSTEM (OPENMP SAFE)")
    print("=" * 50)
    
    test_image = 'temp/tester.jpg'
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    try:
        # Initialize pipeline
        pipeline = RetailShelfPipeline(use_yolov7=True, use_ocr=True)
        
        # Process image
        results = pipeline.process_image(test_image)
        
        # Display results
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Total detections: {len(results.get('detection_data', []))}")
        print(f"   Total products: {results.get('total_products', 0)}")
        print(f"   Product counts: {results.get('sku_counts', {})}")
        
        # Show detailed product breakdown
        if results.get('sku_counts'):
            print(f"\n📊 PRODUCT BREAKDOWN:")
            for product, count in results['sku_counts'].items():
                print(f"   {product}: {count} units")
        
        # Show OCR results
        if results.get('ocr_texts'):
            ocr_found = [text for text in results['ocr_texts'] if text]
            print(f"\n📝 OCR RESULTS: Found text in {len(ocr_found)} products")
            for i, text in enumerate(ocr_found[:10]):  # Show first 10
                print(f"   {i+1}. '{text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    
    if success:
        print("\n🎉 SYSTEM TEST PASSED!")
        print("\nTo run the app:")
        print("python run_app_fixed.py")
    else:
        print("\n❌ System test failed")
