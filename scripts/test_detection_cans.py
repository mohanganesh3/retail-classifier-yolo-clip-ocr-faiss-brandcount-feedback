#!/usr/bin/env python3
"""
Test detection specifically for cool drink cans with ultra-liberal settings
"""

import sys
import os
sys.path.append('.')

from src.detection.detector import ProductDetector
from src.utils.config import ensure_directories
import cv2

def test_can_detection(image_path):
    """Test detection with ultra-liberal settings for cans"""
    print("🥤 Testing Cool Drink Can Detection")
    print("=" * 40)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Ensure directories
    ensure_directories()
    
    # Initialize detector with ultra-liberal settings
    print("Initializing detector with ultra-liberal settings...")
    detector = ProductDetector(use_yolov7=True)
    
    # Override confidence threshold to ultra-low
    if hasattr(detector, 'detector'):
        detector.detector.conf_threshold = 0.001  # Ultra-low
    else:
        detector.conf_threshold = 0.001
    
    # Load and display image info
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"📸 Image: {image_path}")
    print(f"📐 Dimensions: {w}x{h}")
    
    # Test detection with ultra-liberal settings
    print("\n🔍 Running ultra-liberal detection...")
    detection_data, crops = detector.detect_with_fallback(image_path)
    
    print(f"\n📊 Results:")
    print(f"   Total detections: {len(detection_data)}")
    print(f"   Crops generated: {len(crops)}")
    
    if detection_data:
        print("\n📋 Detection Details:")
        
        # Group by class
        class_counts = {}
        for det in detection_data:
            class_name = det.get('class_name', 'unknown')
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        print("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   {class_name}: {count}")
        
        # Show first 10 detections
        print("\nFirst 10 detections:")
        for i, det in enumerate(detection_data[:10]):
            print(f"   {i+1}. {det.get('class_name', 'unknown')} (conf: {det.get('confidence', 0):.3f})")
    
    return len(crops) > 0

def main():
    # Test with available images
    test_images = [
        'temp/testttttttttttt.jpg',
        'data/raw_images/testttttttttttt.jpg',
        'test_image.jpg'
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            success = test_can_detection(img_path)
            if success:
                print("✅ Detection test completed!")
            else:
                print("⚠️ No detections found")
            break
    else:
        print("❌ No test images found")
        print("Please provide an image path:")
        print("python scripts/test_detection_cans.py <image_path>")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_can_detection(sys.argv[1])
    else:
        main()
