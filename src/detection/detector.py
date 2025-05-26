import torch
import cv2
import json
import numpy as np
from src.utils.config import load_config
from src.utils.image_utils import extract_crops_from_detections
from src.detection.precise_can_detector import PreciseCanDetector
import os

class ProductDetector:
    def __init__(self, model_path='models/detector/yolov7-w6.pt', config_path='configs/detection.yaml', use_yolov7=True):
        self.config = load_config(config_path)
        self.conf_threshold = 0.001  # Much lower threshold for detection
        self.use_yolov7 = use_yolov7
        
        print(f"🎯 Initializing Precise ProductDetector")
        
        # Initialize precise can detector
        self.precise_detector = PreciseCanDetector()
        self.model_type = "Precise Can Detector"
    
    def detect(self, image_path, output_dir='temp/crops'):
        """Precise detection with clean, non-overlapping rectangles"""
        print(f"🎯 Precise Detection: {image_path}")
        
        # Use precise detector
        detection_data, crops = self.precise_detector.detect_cans_precisely(image_path, output_dir)
        
        print(f"✅ Precise detection: {len(crops)} clean detections")
        return detection_data, crops
    
    def _merge_overlapping_boxes(self, detection_data, iou_threshold=0.3, distance_threshold=50):
        """Merge overlapping or close bounding boxes that likely represent the same object"""
        if not detection_data:
            return detection_data
        
        print(f"🔧 Merging overlapping detections (threshold: IoU={iou_threshold}, distance={distance_threshold}px)")
        
        merged_detections = []
        used_indices = set()
        
        for i, det1 in enumerate(detection_data):
            if i in used_indices:
                continue
                
            bbox1 = det1.get('bbox', [])
            if len(bbox1) < 4:
                continue
                
            # Find all detections that should be merged with this one
            merge_group = [det1]
            merge_indices = {i}
            
            for j, det2 in enumerate(detection_data[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                bbox2 = det2.get('bbox', [])
                if len(bbox2) < 4:
                    continue
                
                # Check if boxes should be merged
                if self._should_merge_boxes(bbox1, bbox2, iou_threshold, distance_threshold):
                    merge_group.append(det2)
                    merge_indices.add(j)
            
            # Merge the group into a single detection
            if len(merge_group) > 1:
                merged_detection = self._merge_detection_group(merge_group)
                merged_detections.append(merged_detection)
                print(f"✅ Merged {len(merge_group)} detections into one")
            else:
                merged_detections.append(det1)
            
            used_indices.update(merge_indices)
        
        print(f"🔧 Merging result: {len(detection_data)} -> {len(merged_detections)} detections")
        return merged_detections

    def _should_merge_boxes(self, bbox1, bbox2, iou_threshold, distance_threshold):
        """Determine if two bounding boxes should be merged"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate IoU
        iou = self._calculate_iou(bbox1, bbox2)
        if iou > iou_threshold:
            return True
        
        # Calculate center points
        center1_x, center1_y = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        center2_x, center2_y = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # Calculate distance between centers
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        if distance < distance_threshold:
            # Check if they're aligned (vertically or horizontally)
            width1, height1 = x2_1 - x1_1, y2_1 - y1_1
            width2, height2 = x2_2 - x1_2, y2_2 - y1_2
            
            # Vertical alignment (split bottle top/bottom)
            horizontal_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            if horizontal_overlap > min(width1, width2) * 0.5:  # 50% horizontal overlap
                return True
            
            # Horizontal alignment (split bottle left/right)
            vertical_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            if vertical_overlap > min(height1, height2) * 0.5:  # 50% vertical overlap
                return True
        
        return False

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
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

    def _merge_detection_group(self, detections):
        """Merge a group of detections into a single detection"""
        # Calculate merged bounding box
        all_x1 = [det['bbox'][0] for det in detections]
        all_y1 = [det['bbox'][1] for det in detections]
        all_x2 = [det['bbox'][2] for det in detections]
        all_y2 = [det['bbox'][3] for det in detections]
        
        merged_bbox = [
            min(all_x1),  # x1
            min(all_y1),  # y1
            max(all_x2),  # x2
            max(all_y2)   # y2
        ]
        
        # Use highest confidence detection as base
        best_detection = max(detections, key=lambda x: x.get('confidence', 0))
        
        # Create merged detection
        merged_detection = best_detection.copy()
        merged_detection['bbox'] = merged_bbox
        merged_detection['merged_from'] = len(detections)
        
        return merged_detection

    def detect_with_fallback(self, image_path, output_dir='temp/crops'):
        """Precise detection with merging of split detections"""
        print("🎯 Starting detection with split detection merging...")
        
        # Primary: Precise detection
        detection_data, crops = self.detect(image_path, output_dir)
        
        # Apply bounding box merging to fix split detections
        if detection_data:
            print("🔧 Applying bounding box merging...")
            merged_detection_data = self._merge_overlapping_boxes(detection_data)
            
            # Re-extract crops with merged bounding boxes
            if len(merged_detection_data) != len(detection_data):
                print(f"🔧 Re-extracting crops after merging...")
                merged_crops = self._extract_merged_crops(image_path, merged_detection_data, output_dir)
                detection_data = merged_detection_data
                crops = merged_crops
        
        # Fallback logic (existing)
        if len(crops) == 0:
            print("❌ No detections found, trying ultra-aggressive detection...")
            
            # Try with ultra-low confidence and accept more classes
            try:
                from ultralytics import YOLO
                fallback_model = YOLO('yolov8x.pt')
                
                # Ultra-low confidence detection
                results = fallback_model(image_path, conf=0.0001, iou=0.05, verbose=True)
                
                fallback_detections = []
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        bboxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy()
                        
                        class_names = [fallback_model.names[int(cls_id)] for cls_id in class_ids]
                        
                        print(f"🔍 Ultra-aggressive found {len(bboxes)} objects")
                        print(f"    Classes: {set(class_names)}")
                        
                        # Accept almost anything that could be a can
                        for bbox, conf, class_name in zip(bboxes, confidences, class_names):
                            x1, y1, x2, y2 = bbox
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Very basic size check
                            if width > 10 and height > 15 and width < 500 and height < 600:
                                fallback_detections.append({
                                    'bbox': bbox.tolist(),
                                    'confidence': float(conf),
                                    'class_name': class_name,
                                    'detection_type': 'ultra_aggressive',
                                    'width': width,
                                    'height': height
                                })
                
                if fallback_detections:
                    # Apply NMS to clean up
                    clean_fallback = self.precise_detector._apply_precise_nms(fallback_detections, 0.3)
                    
                    if clean_fallback:
                        # Extract crops
                        fallback_crops = self._extract_fallback_crops(image_path, clean_fallback, output_dir)
                        
                        print(f"✅ Ultra-aggressive fallback found {len(fallback_crops)} objects")
                        detection_data = clean_fallback
                        crops = fallback_crops
                        
            except Exception as e:
                print(f"⚠️ Ultra-aggressive fallback failed: {e}")
        
        print(f"🎯 Final result: {len(crops)} detections (after merging)")
        return detection_data, crops

    def _extract_merged_crops(self, image_path, detection_data, output_dir):
        """Extract crops from merged detection data"""
        img = cv2.imread(image_path)
        crops = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, detection in enumerate(detection_data):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are valid
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Add padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = img[y1:y2, x1:x2]
            
            # Name based on whether it was merged
            if detection.get('merged_from', 1) > 1:
                crop_path = f"{output_dir}/merged_crop_{i:04d}.jpg"
            else:
                crop_path = f"{output_dir}/single_crop_{i:04d}.jpg"
            
            cv2.imwrite(crop_path, crop)
            crops.append(crop_path)
            
            # Update detection with crop path
            detection['crop_path'] = crop_path
        
        return crops
    
    def _extract_fallback_crops(self, image_path, detections, output_dir):
        """Extract crops from fallback detections"""
        img = cv2.imread(image_path)
        crops = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are valid
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            crop = img[y1:y2, x1:x2]
            crop_path = f"{output_dir}/fallback_crop_{i:04d}.jpg"
            cv2.imwrite(crop_path, crop)
            crops.append(crop_path)
        
        return crops
    
    def train(self, data_yaml_path):
        """Train detector model"""
        print("Training not implemented for precise detector")
