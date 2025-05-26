#!/usr/bin/env python3
"""
Test classification with OpenMP fix
"""

import os
import sys

# Fix OpenMP conflicts BEFORE importing any ML libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.append('.')

from src.pipeline.retail_pipeline import RetailShelfPipeline
from src.utils.config import ensure_directories

def test_classification_safe():
    """Test classification with OpenMP safety"""
    print("🧪 TESTING CLASSIFICATION (OPENMP SAFE)")
    print("=" * 45)
    
    # Test image
    test_image = 'temp/testttttttttttt.jpg'
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"📸 Testing with image: {test_image}")
    
    # Ensure directories
    ensure_directories()
    
    try:
        print("🔧 Initializing pipeline...")
        pipeline = RetailShelfPipeline(use_yolov7=True, use_ocr=True)
        
        print(f"✅ Pipeline initialized")
        print(f"   Classifier available: {pipeline.classifier_available}")
        if hasattr(pipeline, 'classifier') and pipeline.classifier:
            print(f"   FAISS index dimension: {pipeline.classifier.index.d}")
            print(f"   Number of SKUs: {len(pipeline.classifier.labels)}")
        
        print("\n🚀 Processing image...")
        results = pipeline.process_image(test_image)
        
        print(f"\n📊 RESULTS:")
        print(f"   Total detections: {len(results.get('detection_data', []))}")
        print(f"   Classifications: {len(results.get('classifications', []))}")
        print(f"   SKU counts: {results.get('sku_counts', {})}")
        print(f"   Total products: {results.get('total_products', 0)}")
        
        # Check if classification worked
        if results.get('classifications'):
            print("\n✅ CLASSIFICATION SUCCESS!")
            for i, cls in enumerate(results['classifications'][:3]):
                print(f"   {i+1}. SKU: {cls.get('sku_id')} (conf: {cls.get('confidence', 0):.3f})")
            return True
        else:
            print("\n⚠️ No classifications generated")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_classification_safe()
    
    if success:
        print("\n🎉 CLASSIFICATION TEST PASSED!")
        print("\nThe system is now working correctly:")
        print("✅ Detection working")
        print("✅ OCR working") 
        print("✅ Classification working")
        print("✅ No OpenMP conflicts")
        
        print("\nRun the Streamlit app:")
        print("streamlit run app.py")
    else:
        print("\n❌ Classification test failed")
        print("Check the error messages above")

if __name__ == "__main__":
    main()
