import cv2
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image_detection(image_path, target_size=(640, 640)):
    """Preprocess image for YOLOv8 detection"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.transpose(2, 0, 1) / 255.0
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def preprocess_image_classification(image_path, target_size=(224, 224)):
    """Preprocess image for MobileNetV3 classification"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def extract_crops_from_detections(image_path, bboxes, output_dir='temp/crops'):
    """Extract crops from image based on bounding boxes with proper coordinate handling"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return []
        
        h, w = img.shape[:2]
        crops = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Image dimensions: {w}x{h}")
        print(f"Processing {len(bboxes)} bounding boxes")
        
        for i, bbox in enumerate(bboxes):
            try:
                # Handle different bbox formats
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    
                    # Convert to integers and ensure proper order
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure x1 < x2 and y1 < y2
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    # Clamp coordinates to image bounds
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))
                    
                    # Add padding but stay within bounds
                    padding = 10
                    x1_pad = max(0, x1 - padding)
                    y1_pad = max(0, y1 - padding)
                    x2_pad = min(w, x2 + padding)
                    y2_pad = min(h, y2 + padding)
                    
                    # Extract crop
                    crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    # Ensure crop is not empty
                    if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                        crop_path = f"{output_dir}/crop_{i:04d}.jpg"
                        success = cv2.imwrite(crop_path, crop)
                        
                        if success:
                            crops.append(crop_path)
                            print(f"Saved crop {i}: {crop.shape} -> {crop_path}")
                        else:
                            print(f"Failed to save crop {i}")
                    else:
                        print(f"Skipping invalid crop {i}: {crop.shape if crop.size > 0 else 'empty'}")
                        
            except Exception as e:
                print(f"Error processing bbox {i}: {e}")
                continue
        
        print(f"Successfully extracted {len(crops)} crops")
        return crops
        
    except Exception as e:
        print(f"Error in extract_crops_from_detections: {e}")
        return []
