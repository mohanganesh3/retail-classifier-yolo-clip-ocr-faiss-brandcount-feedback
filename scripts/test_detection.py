#!/usr/bin/env python3
"""
Test detection system with debug information
"""

import sys
import os
sys.path.append('.')

from src.detection.detector import ProductDetector
from src.utils.config import ensure_directories
import cv2

def test_detection(image_path):
    """Test detection with detailed output"""
    print("🔍 Testing Detection System")
    print("=" * 40)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Ensure directories
    ensure_directories()
    
    # Initialize detector
    print("Initializing detector...")
    detector = ProductDetector()
    
    # Load and display image info
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"📸 Image: {image_path}")
    print(f"📐 Dimensions: {w}x{h}")
    
    # Test detection
    print("\n🔍 Running detection...")
    detection_data, crops = detector.detect_with_fallback(image_path)
    
    print(f"\n📊 Results:")
    print(f"   Detections: {len(detection_data)}")
    print(f"   Crops: {len(crops)}")
    
    if detection_data:
        print("\n📋 Detection Details:")
        for i, det in enumerate(detection_data[:5]):  # Show first 5
            print(f"   {i+1}. Class: {det.get('class_name', 'unknown')}")
            print(f"      Confidence: {det.get('confidence', 0):.3f}")
            print(f"      Crop: {det.get('crop_path', 'none')}")
    
    return len(crops) > 0

def main():
    # Test with sample image if available
    test_images = [
        'temp/sample_shelf.jpg',
        'sample_shelf.jpg',
        'test_image.jpg'
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            success = test_detection(img_path)
            if success:
                print("✅ Detection test passed!")
            else:
                print("⚠️ No detections found")
            break
    else:
        print("❌ No test images found")
        print("Please provide an image path:")
        print("python scripts/test_detection.py <image_path>")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_detection(sys.argv[1])
    else:
        main()
