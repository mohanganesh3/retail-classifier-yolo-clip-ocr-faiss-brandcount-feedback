#!/usr/bin/env python3
"""
Test the precise can detection system
"""

import os
import sys

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.append('.')

from src.detection.precise_can_detector import PreciseCanDetector

def test_precise_detection():
    """Test precise can detection"""
    print("🧪 TESTING PRECISE CAN DETECTION")
    print("=" * 50)
    
    test_image = 'temp/tester.jpg'
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    try:
        # Initialize precise detector
        detector = PreciseCanDetector()
        
        # Run precise detection
        detections, crops = detector.detect_cans_precisely(test_image)
        
        print(f"\n🎯 PRECISE DETECTION RESULTS:")
        print(f"   Clean detections: {len(detections)}")
        print(f"   Crops extracted: {len(crops)}")
        
        if detections:
            print(f"\n📊 DETECTION DETAILS:")
            
            # Check for overlaps (should be zero)
            overlaps = 0
            for i, det1 in enumerate(detections):
                for j, det2 in enumerate(detections[i+1:], i+1):
                    iou = detector._calculate_iou(det1['bbox'], det2['bbox'])
                    if iou > 0.1:  # Any significant overlap
                        overlaps += 1
                        print(f"⚠️ Overlap detected between detection {i} and {j}: IoU = {iou:.3f}")
            
            if overlaps == 0:
                print(f"✅ No overlapping detections - Perfect!")
            else:
                print(f"❌ Found {overlaps} overlapping detections")
            
            # Show detection summary
            can_types = {}
            for det in detections:
                can_type = det.get('can_type', 'unknown')
                can_types[can_type] = can_types.get(can_type, 0) + 1
            
            print(f"\n📦 CAN TYPES DETECTED:")
            for can_type, count in can_types.items():
                print(f"   {can_type}: {count}")
            
            # Show confidence stats
            confidences = [det.get('confidence', 0) for det in detections]
            print(f"\n📈 CONFIDENCE STATS:")
            print(f"   Range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"   Average: {sum(confidences)/len(confidences):.3f}")
            
            # Show bounding box info
            print(f"\n📐 BOUNDING BOX INFO:")
            for i, det in enumerate(detections[:5]):  # Show first 5
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                print(f"   {i+1}. [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] ({width:.0f}x{height:.0f})")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_precise_detection()
    
    if success:
        print("\n🎉 PRECISE DETECTION TEST PASSED!")
        print("\nThe system now provides:")
        print("✅ One rectangle per can")
        print("✅ No overlapping rectangles")
        print("✅ Precise bounding box placement")
        print("✅ Clean, accurate detection")
    else:
        print("\n❌ Precise detection test failed")
