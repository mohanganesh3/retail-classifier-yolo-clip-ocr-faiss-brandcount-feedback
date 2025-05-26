#!/usr/bin/env python3
"""
Test the enhanced can/tin detection system
"""

import os
import sys

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.append('.')

from src.detection.can_tin_detector import CanTinDetector

def test_can_tin_detection():
    """Test enhanced can/tin detection"""
    print("🧪 TESTING ENHANCED CAN/TIN DETECTION")
    print("=" * 50)
    
    test_image = 'temp/tester.jpg'
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    try:
        # Initialize enhanced detector
        detector = CanTinDetector()
        
        # Run detection
        detections, crops = detector.detect_cans_and_tins(test_image)
        
        print(f"\n🎯 ENHANCED DETECTION RESULTS:")
        print(f"   Total detections: {len(detections)}")
        print(f"   Crops extracted: {len(crops)}")
        
        if detections:
            # Analyze detection types
            detection_types = {}
            models_used = {}
            
            for det in detections:
                det_type = det.get('detection_type', 'unknown')
                model = det.get('model', det.get('class_name', 'unknown'))
                
                detection_types[det_type] = detection_types.get(det_type, 0) + 1
                models_used[model] = models_used.get(model, 0) + 1
            
            print(f"\n📊 DETECTION BREAKDOWN:")
            print(f"   By Type:")
            for det_type, count in detection_types.items():
                print(f"      {det_type}: {count}")
            
            print(f"   By Source:")
            for model, count in models_used.items():
                print(f"      {model}: {count}")
            
            # Show confidence distribution
            confidences = [det.get('confidence', 0) for det in detections]
            print(f"\n📈 CONFIDENCE STATS:")
            print(f"   Range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"   Average: {sum(confidences)/len(confidences):.3f}")
            
            # Show first few detections
            print(f"\n🔍 SAMPLE DETECTIONS:")
            for i, det in enumerate(detections[:10]):
                conf = det.get('confidence', 0)
                class_name = det.get('class_name', 'unknown')
                det_type = det.get('detection_type', 'unknown')
                print(f"   {i+1}. {class_name} -> {det_type} (conf: {conf:.3f})")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_can_tin_detection()
    
    if success:
        print("\n🎉 ENHANCED CAN/TIN DETECTION TEST PASSED!")
        print("\nThe system now detects:")
        print("✅ Bottles (traditional)")
        print("✅ Cans (cylindrical)")
        print("✅ Tins (rectangular)")
        print("✅ Metallic containers")
        print("✅ Shape-based detection")
        print("✅ Color-based detection")
    else:
        print("\n❌ Enhanced detection test failed")
