import torch
import cv2
import json
import numpy as np
import os
import sys
import requests
from pathlib import Path
import zipfile
import subprocess
from ultralytics import YOLO

class YOLOv7W6Detector:
    def __init__(self, model_path='models/detector/yolov7-w6.pt', config_path='configs/detection.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = 0.15  # Lower threshold for retail
        self.iou_threshold = 0.45
        self.img_size = 1280  # YOLOv7-W6 optimal input size
        
        print(f"Initializing YOLOv7-W6 detector on {self.device}")
        
        # Try multiple approaches to load YOLOv7
        self.model = None
        self.model_loaded = False
        
        # Approach 1: Try ultralytics YOLOv7 (if available)
        try:
            print("Attempting to load YOLOv7 via ultralytics...")
            self.model = YOLO('yolov7.pt')  # This will download if not exists
            self.model_loaded = True
            print("✅ Successfully loaded YOLOv7 via ultralytics")
        except Exception as e:
            print(f"Ultralytics YOLOv7 failed: {e}")
        
        # Approach 2: Try torch.hub
        if not self.model_loaded:
            try:
                print("Attempting to load YOLOv7 via torch.hub...")
                self.model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                print("✅ Successfully loaded YOLOv7 via torch.hub")
            except Exception as e:
                print(f"Torch.hub YOLOv7 failed: {e}")
        
        # Approach 3: Fallback to YOLOv8 with larger model
        if not self.model_loaded:
            try:
                print("Falling back to YOLOv8x (largest YOLOv8 model)...")
                self.model = YOLO('yolov8x.pt')  # Largest YOLOv8 model
                self.model_loaded = True
                print("✅ Successfully loaded YOLOv8x as fallback")
            except Exception as e:
                print(f"YOLOv8x fallback failed: {e}")
        
        # Approach 4: Final fallback to YOLOv8n
        if not self.model_loaded:
            print("Final fallback to YOLOv8n...")
            self.model = YOLO('yolov8n.pt')
            self.model_loaded = True
            print("✅ Loaded YOLOv8n as final fallback")
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        
        # Cool drink can specific classes and keywords
        self.can_keywords = [
            'bottle', 'can', 'beverage', 'drink', 'soda', 'cola', 'pepsi', 'coke', 
            'sprite', 'fanta', 'juice', 'water', 'energy drink', 'beer', 'wine glass', 
            'cup', 'container', 'cylinder', 'aluminum', 'plastic bottle'
        ]

        # Expanded product classes specifically for cool drinks and cans
        self.product_classes = [
            'bottle', 'wine glass', 'cup', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'book', 'clock', 'vase', 'cell phone', 'remote', 'toothbrush', 'backpack', 'handbag',
            'suitcase', 'umbrella', 'tie', 'scissors', 'hair drier', 'teddy bear'
        ]
    
    def detect(self, image_path, output_dir='temp/crops'):
        """Optimized can detection with precise bounding boxes"""
        print(f"🥤 Detecting cool drink cans: {image_path}")
        
        if not self.model_loaded:
            print("❌ No model loaded successfully")
            return [], []
        
        try:
            # Check if image exists and is readable
            if not os.path.exists(image_path):
                print(f"❌ Image file not found: {image_path}")
                return [], []
        
            test_img = cv2.imread(image_path)
            if test_img is None:
                print(f"❌ Could not read image: {image_path}")
                return [], []
        
            print(f"📸 Image loaded: {test_img.shape}")
        
            # Run detection with optimized settings for cans
            print(f"🔍 Running can-optimized detection with confidence: {self.conf_threshold}")
        
            # Use ultralytics interface with can-specific settings
            results = self.model(image_path, conf=self.conf_threshold, iou=0.3, verbose=True)
        
            detection_data = []
            all_crops = []
        
            print(f"📊 Got {len(results)} result objects")
        
            for i, result in enumerate(results):
                print(f"Processing result {i}")
            
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    print(f"Found {len(boxes)} boxes")
                
                    if len(boxes) > 0:
                        # Get detection data
                        bboxes = boxes.xyxy.cpu().numpy()
                        confidences = boxes.conf.cpu().numpy()
                        class_ids = boxes.cls.cpu().numpy()
                    
                        print(f"Raw detections: {len(bboxes)} objects")
                        print(f"Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
                    
                        # Get class names
                        if hasattr(self.model, 'names'):
                            class_names = [self.model.names[int(cls_id)] for cls_id in class_ids]
                        else:
                            class_names = [self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else 'unknown' for cls_id in class_ids]
                    
                        print(f"Classes detected: {set(class_names)}")
                    
                        # Apply can-focused filtering with size validation
                        filtered_indices = []
                    
                        for idx, (bbox, conf, class_name) in enumerate(zip(bboxes, confidences, class_names)):
                            x1, y1, x2, y2 = bbox
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = height / width if width > 0 else 0
                        
                            # Can-specific filtering
                            is_can_like = False
                        
                            # Accept bottles/cans with very low confidence
                            if class_name in ['bottle', 'cup', 'wine glass', 'bowl'] and conf > 0.03:
                                is_can_like = True
                            # Accept books as potential cans (common misclassification)
                            elif class_name == 'book' and conf > 0.05 and 1.5 < aspect_ratio < 4.0:
                                is_can_like = True
                            # Accept other objects if they look can-like
                            elif conf > 0.1 and 1.2 < aspect_ratio < 5.0 and width > 30 and height > 50:
                                is_can_like = True
                        
                            # Size validation for individual cans
                            if is_can_like and width > 20 and height > 40 and width < 200 and height < 400:
                                filtered_indices.append(idx)
                                print(f"✅ Can detected: {class_name} ({conf:.3f}) - {width:.0f}x{height:.0f}, AR:{aspect_ratio:.2f}")
                            elif is_can_like:
                                print(f"⚠️ Rejected size: {class_name} ({conf:.3f}) - {width:.0f}x{height:.0f}")
                    
                        print(f"Filtered to {len(filtered_indices)} can detections")
                    
                        if filtered_indices:
                            # Apply NMS to remove overlapping detections
                            filtered_bboxes = bboxes[filtered_indices]
                            filtered_confidences = confidences[filtered_indices]
                        
                            # Apply NMS
                            keep_indices = self._apply_nms(filtered_bboxes, filtered_confidences, iou_threshold=0.3)
                        
                            final_bboxes = filtered_bboxes[keep_indices]
                            final_indices = [filtered_indices[i] for i in keep_indices]
                        
                            print(f"After NMS: {len(final_bboxes)} unique cans")
                        
                            # Extract crops with precise bounding boxes
                            crops = self.extract_crops_from_detections_precise(image_path, final_bboxes, output_dir)
                            all_crops.extend(crops)
                        
                            # Create detection data
                            for j, idx in enumerate(final_indices):
                                bbox = bboxes[idx]
                                conf = confidences[idx]
                                class_name = class_names[idx]
                            
                                detection_data.append({
                                    'crop_path': crops[j] if j < len(crops) else None,
                                    'bbox': bbox.tolist(),
                                    'confidence': float(conf),
                                    'class_name': class_name,
                                    'class_id': int(class_ids[idx]),
                                    'image_id': i
                                })
                    else:
                        print("⚠️ No can-like detections found")
            else:
                print("⚠️ No boxes found in result")
        
            print(f"✅ Final can detection count: {len(all_crops)} individual cans")
            return detection_data, all_crops
        
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    
    def detect_with_fallback(self, image_path, output_dir='temp/crops'):
        """Optimized detection focused on individual cans - NO GRID FALLBACK"""
        print("🎯 Starting optimized can detection...")
        
        # Store original confidence threshold
        original_conf = self.conf_threshold
        
        # Primary detection with multiple confidence levels
        detection_data, crops = self.detect(image_path, output_dir)
        
        if len(crops) < 3:
            print(f"Only {len(crops)} cans detected, trying lower confidence...")
        
            # Try ultra-low confidence for missed cans
            self.conf_threshold = 0.01  # Ultra-low for cans
        
            fallback_data, fallback_crops = self.detect(image_path, output_dir)
        
            if len(fallback_crops) > len(crops):
                print(f"✅ Lower confidence found {len(fallback_crops)} cans")
                detection_data = fallback_data
                crops = fallback_crops
        
        # Restore original threshold
        self.conf_threshold = original_conf

        # NO GRID FALLBACK - only real detections
        print(f"🎯 Final result: {len(crops)} real can detections (NO MOCK SQUARES)")
        return detection_data, crops
    
    
    def _grid_based_detection(self, image_path, output_dir, grid_size=(6, 8)):
        """REMOVED - No more grid fallback creating fake squares"""
        print("🚫 Grid fallback disabled - using only real detections")
        return []
    
    def extract_crops_from_detections(self, image_path, bboxes, output_dir):
        """Extract crops from detected bounding boxes"""
        img = cv2.imread(image_path)
        crops = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Add some padding around the detection
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = img[y1:y2, x1:x2]
            crop_path = f"{output_dir}/yolov7_crop_{i}.jpg"
            cv2.imwrite(crop_path, crop)
            crops.append(crop_path)
        
        return crops

    def _apply_nms(self, bboxes, confidences, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        try:
            import torch
            import torchvision.ops as ops
        
            boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            scores_tensor = torch.tensor(confidences, dtype=torch.float32)
        
            keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
            return keep_indices.numpy()
        except:
            # Fallback: return all indices
            return list(range(len(bboxes)))

    def extract_crops_from_detections_precise(self, image_path, bboxes, output_dir):
        """Extract precise crops for individual cans"""
        img = cv2.imread(image_path)
        crops = []
    
        os.makedirs(output_dir, exist_ok=True)
    
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
        
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
        
            # Minimal padding for precise cropping
            padding = 2
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
        
            crop = img[y1:y2, x1:x2]
            crop_path = f"{output_dir}/can_crop_{i:04d}.jpg"
            cv2.imwrite(crop_path, crop)
            crops.append(crop_path)
        
            print(f"✅ Extracted precise can crop {i}: {crop.shape}")
    
        return crops
