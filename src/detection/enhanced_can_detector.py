import torch
import cv2
import json
import numpy as np
import os
from ultralytics import YOLO

class EnhancedCanDetector:
    def __init__(self, model_path='yolov8x.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = 0.01  # Ultra-low for cans
        
        print(f"🥤 Initializing Enhanced Can/Tin Detector on {self.device}")
        
        # Load YOLOv8x for best detection
        try:
            self.model = YOLO('yolov8x.pt')  # Largest model for best accuracy
            print("✅ YOLOv8x loaded for enhanced can detection")
        except:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8n loaded as fallback")
        
        # Can/tin specific detection classes
        self.can_classes = ['bottle', 'cup', 'wine glass', 'bowl']
        self.all_classes = [
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
    
    def detect_cans_and_tins(self, image_path, output_dir='temp/crops'):
        """Enhanced detection specifically for cans and tins"""
        print(f"🔍 Enhanced Can/Tin Detection: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return [], []
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return [], []
        
        h, w = img.shape[:2]
        print(f"📸 Image dimensions: {w}x{h}")
        
        # Run detection with multiple confidence levels
        all_detections = []
        all_crops = []
        
        # Try multiple confidence thresholds
        confidence_levels = [0.01, 0.005, 0.001]
        
        for conf_level in confidence_levels:
            print(f"🔍 Trying confidence level: {conf_level}")
            
            results = self.model(image_path, conf=conf_level, iou=0.2, verbose=False)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    bboxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    # Get class names
                    class_names = [self.model.names[int(cls_id)] for cls_id in class_ids]
                    
                    print(f"📊 Found {len(bboxes)} objects at confidence {conf_level}")
                    
                    # Enhanced filtering for cans and tins
                    for idx, (bbox, conf, class_name) in enumerate(zip(bboxes, confidences, class_names)):
                        x1, y1, x2, y2 = bbox
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = height / width if width > 0 else 0
                        
                        # Detect both bottles AND cans/tins
                        is_can_or_tin = False
                        can_type = "unknown"
                        
                        # Bottle detection (existing)
                        if class_name == 'bottle' and conf > 0.01:
                            is_can_or_tin = True
                            can_type = "bottle"
                        
                        # Can/Tin detection (NEW - more aggressive)
                        elif class_name in ['cup', 'wine glass', 'bowl'] and conf > 0.005:
                            # These often get misclassified as cans
                            if 1.5 < aspect_ratio < 4.0:  # Can-like aspect ratio
                                is_can_or_tin = True
                                can_type = "can"
                        
                        # Book detection (cans often misclassified as books)
                        elif class_name == 'book' and 1.2 < aspect_ratio < 5.0 and conf > 0.003:
                            is_can_or_tin = True
                            can_type = "can"
                        
                        # Cell phone detection (thin cans)
                        elif class_name == 'cell phone' and 2.0 < aspect_ratio < 6.0 and conf > 0.003:
                            is_can_or_tin = True
                            can_type = "can"
                        
                        # Remote detection (rectangular cans)
                        elif class_name == 'remote' and 1.5 < aspect_ratio < 4.0 and conf > 0.003:
                            is_can_or_tin = True
                            can_type = "can"
                        
                        # ANY object with can-like properties
                        elif 1.2 < aspect_ratio < 5.0 and width > 25 and height > 40 and conf > 0.002:
                            is_can_or_tin = True
                            can_type = "can"
                        
                        # Size validation
                        if is_can_or_tin:
                            # Accept wider range of sizes for cans/tins
                            if (width > 15 and height > 30 and 
                                width < 300 and height < 500):
                                
                                # Check if already detected (avoid duplicates)
                                is_duplicate = False
                                for existing in all_detections:
                                    ex_bbox = existing['bbox']
                                    # Calculate overlap
                                    overlap = self._calculate_overlap(bbox, ex_bbox)
                                    if overlap > 0.5:  # 50% overlap threshold
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    all_detections.append({
                                        'bbox': bbox.tolist(),
                                        'confidence': float(conf),
                                        'class_name': class_name,
                                        'can_type': can_type,
                                        'aspect_ratio': aspect_ratio,
                                        'size': f"{width:.0f}x{height:.0f}"
                                    })
                                    
                                    print(f"✅ {can_type.upper()} detected: {class_name} ({conf:.3f}) - {width:.0f}x{height:.0f}, AR:{aspect_ratio:.2f}")
        
        print(f"🎯 Total unique cans/tins detected: {len(all_detections)}")
        
        # Extract crops
        if all_detections:
            bboxes = [det['bbox'] for det in all_detections]
            all_crops = self._extract_crops_precise(image_path, bboxes, output_dir)
            
            # Update detection data with crop paths
            for i, detection in enumerate(all_detections):
                detection['crop_path'] = all_crops[i] if i < len(all_crops) else None
        
        return all_detections, all_crops
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_crops_precise(self, image_path, bboxes, output_dir):
        """Extract precise crops for cans and tins"""
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
            padding = 3
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = img[y1:y2, x1:x2]
            crop_path = f"{output_dir}/enhanced_crop_{i:04d}.jpg"
            cv2.imwrite(crop_path, crop)
            crops.append(crop_path)
            
            print(f"✅ Extracted crop {i}: {crop.shape} -> {crop_path}")
        
        return crops
