import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

class PreciseCanDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = 0.001  # Balanced threshold
        
        print(f"🎯 Initializing Balanced Can Detector on {self.device}")
        
        # Load YOLOv8x for best accuracy
        try:
            self.model = YOLO('yolov8x.pt')
            print("✅ YOLOv8x loaded for balanced detection")
        except:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8n loaded as fallback")
        
        # Accept classes that could be cans
        self.target_classes = [
            'bottle', 'cup', 'wine glass', 'bowl', 'cell phone', 'remote', 
            'book', 'laptop', 'mouse', 'clock', 'vase', 'banana', 'hot dog'
        ]
        
    def detect_cans_precisely(self, image_path, output_dir='temp/crops'):
        """Balanced detection - not too aggressive, not too conservative"""
        print(f"🎯 Balanced Can Detection: {image_path}")
        
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
        
        # Step 1: Run detection with balanced confidence levels
        all_raw_detections = self._run_balanced_detection(image_path)
        
        # Step 2: Filter for can-like objects with balanced criteria
        can_detections = self._filter_can_detections_balanced(all_raw_detections, img.shape)
        
        # Step 3: Smart stack detection - balanced approach
        stack_separated = self._smart_stack_separation(can_detections, img)
        
        # Step 4: Apply balanced NMS
        clean_detections = self._apply_balanced_nms(stack_separated, iou_threshold=0.25)
        
        # Step 5: Validate and refine bounding boxes
        final_detections = self._refine_bounding_boxes(clean_detections, img.shape)
        
        # Step 6: Extract crops
        crops = self._extract_clean_crops(image_path, final_detections, output_dir)
        
        print(f"🎯 Balanced detection result: {len(final_detections)} bottles/cans")
        
        return final_detections, crops
    
    def _run_balanced_detection(self, image_path):
        """Run detection with balanced confidence levels"""
        all_detections = []
        
        # Balanced confidence levels - not too low, not too high
        confidence_levels = [0.001, 0.003, 0.01, 0.03, 0.1]
        
        for conf_level in confidence_levels:
            try:
                results = self.model(image_path, conf=conf_level, iou=0.15, verbose=False)
                
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        bboxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy()
                        
                        # Get class names
                        class_names = [self.model.names[int(cls_id)] for cls_id in class_ids]
                        
                        print(f"📊 At confidence {conf_level}: Found {len(bboxes)} objects")
                        
                        # Collect detections
                        for bbox, conf, class_name in zip(bboxes, confidences, class_names):
                            all_detections.append({
                                'bbox': bbox.tolist(),
                                'confidence': float(conf),
                                'class_name': class_name,
                                'conf_level': conf_level
                            })
                
                # Continue until we have reasonable coverage
                if len(all_detections) > 30:
                    print(f"✅ Found good coverage ({len(all_detections)} detections)")
                    break
                    
            except Exception as e:
                print(f"⚠️ Detection failed at confidence {conf_level}: {e}")
        
        print(f"📊 Total detections collected: {len(all_detections)}")
        return all_detections
    
    def _filter_can_detections_balanced(self, detections, img_shape):
        """Balanced filtering - not too strict, not too loose"""
        h, w = img_shape[:2]
        can_detections = []
        
        print(f"🔍 Balanced filtering of {len(detections)} detections...")
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate dimensions
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width if width > 0 else 0
            area = width * height
            
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Balanced criteria
            is_can = False
            can_type = "unknown"
            
            # Accept bottles with low confidence
            if class_name == 'bottle' and confidence > 0.003:
                if 1.0 < aspect_ratio < 8.0 and width > 15 and height > 30:
                    is_can = True
                    can_type = "bottle"
            
            # Accept cups with reasonable confidence
            elif class_name == 'cup' and confidence > 0.005:
                if 0.8 < aspect_ratio < 6.0 and width > 12 and height > 25:
                    is_can = True
                    can_type = "can"
            
            # Accept wine glasses
            elif class_name == 'wine glass' and confidence > 0.005:
                if 1.2 < aspect_ratio < 8.0 and width > 12 and height > 30:
                    is_can = True
                    can_type = "tall_can"
            
            # Accept bowls (short cans)
            elif class_name == 'bowl' and confidence > 0.01:
                if 0.4 < aspect_ratio < 3.0 and width > 20 and height > 15:
                    is_can = True
                    can_type = "short_can"
            
            # Accept other objects with moderate confidence
            elif class_name in self.target_classes and confidence > 0.01:
                if 0.8 < aspect_ratio < 6.0 and width > 15 and height > 25:
                    is_can = True
                    can_type = "detected_object"
            
            # Accept any object with can-like proportions and decent confidence
            elif confidence > 0.02 and 0.8 < aspect_ratio < 8.0 and width > 12 and height > 20:
                is_can = True
                can_type = "potential_can"
            
            # Balanced validation
            if (is_can and 
                area > 200 and  # Lower minimum area
                width < w*0.8 and height < h*0.8 and  # Allow larger objects
                x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h):
                
                can_detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_name': class_name,
                    'can_type': can_type,
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'width': width,
                    'height': height
                })
                
                print(f"✅ Accepted: {class_name} -> {can_type} ({confidence:.3f}) {width:.0f}x{height:.0f} AR:{aspect_ratio:.2f}")
        
        print(f"🔍 Balanced filtering result: {len(can_detections)} potential cans")
        return can_detections
    
    def _smart_stack_separation(self, detections, img):
        """Smart stack separation - balanced approach"""
        print("🔧 Smart stack separation...")
        
        separated_detections = []
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width if width > 0 else 0
            confidence = det['confidence']
            
            # Smart separation criteria:
            # 1. Aspect ratio suggests stacking (>3.5)
            # 2. Reasonable confidence (>0.01)
            # 3. Height suggests multiple bottles (>100px)
            # 4. Width is reasonable for bottles (20-120px)
            should_separate = (
                aspect_ratio > 3.5 and 
                confidence > 0.01 and 
                height > 100 and 
                20 < width < 120
            )
            
            if should_separate:
                # Estimate bottles based on height and aspect ratio
                if aspect_ratio > 6.0:
                    estimated_bottles = min(4, max(2, int(height / 50)))  # Very tall
                elif aspect_ratio > 4.5:
                    estimated_bottles = min(3, max(2, int(height / 60)))  # Moderately tall
                else:
                    estimated_bottles = 2  # Just split in half
                
                print(f"🔧 Separating stack: AR={aspect_ratio:.2f}, conf={confidence:.3f}, {width:.0f}x{height:.0f} -> {estimated_bottles} bottles")
                
                bottle_height = height / estimated_bottles
                
                for i in range(estimated_bottles):
                    sep_y1 = y1 + (i * bottle_height)
                    sep_y2 = y1 + ((i + 1) * bottle_height)
                    
                    separated_det = det.copy()
                    separated_det['bbox'] = [x1, sep_y1, x2, sep_y2]
                    separated_det['confidence'] = confidence * 0.85  # Slight confidence reduction
                    separated_det['can_type'] = 'separated_bottle'
                    separated_det['separation_index'] = i
                    separated_det['original_stack'] = True
                    separated_det['estimated_bottles'] = estimated_bottles
                    
                    separated_detections.append(separated_det)
                    print(f"✅ Created separated bottle {i+1}/{estimated_bottles}")
            else:
                # Keep as single detection
                separated_detections.append(det)
                if aspect_ratio > 2.5:
                    print(f"⚪ Keeping single: AR={aspect_ratio:.2f}, conf={confidence:.3f} {width:.0f}x{height:.0f}")
        
        print(f"🔧 Smart separation: {len(detections)} -> {len(separated_detections)} detections")
        return separated_detections
    
    def _apply_balanced_nms(self, detections, iou_threshold=0.25):
        """Apply balanced NMS - not too strict, not too loose"""
        if not detections:
            return []
        
        print(f"🔧 Applying balanced NMS with IoU threshold {iou_threshold}")
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        clean_detections = []
        
        for det in detections:
            # Check if this detection overlaps significantly with existing ones
            should_keep = True
            
            for existing in clean_detections:
                overlap = self._calculate_iou(det['bbox'], existing['bbox'])
                
                # Special handling for separated bottles
                if (det.get('original_stack') and existing.get('original_stack') and
                    self._are_from_same_stack(det, existing)):
                    # Allow separated bottles from same stack to coexist
                    continue
                
                # Standard overlap check
                if overlap > iou_threshold:
                    should_keep = False
                    print(f"⚠️ Removing overlapping detection (IoU: {overlap:.3f})")
                    break
            
            if should_keep:
                clean_detections.append(det)
        
        print(f"🎯 Balanced NMS result: {len(detections)} -> {len(clean_detections)} clean detections")
        return clean_detections
    
    def _are_from_same_stack(self, det1, det2):
        """Check if two detections are from the same original stack"""
        # If they have similar x-coordinates and are vertically aligned
        bbox1 = det1['bbox']
        bbox2 = det2['bbox']
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Check horizontal alignment
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        width1 = x2_1 - x1_1
        width2 = x2_2 - x1_2
        
        horizontal_overlap_ratio = x_overlap / min(width1, width2) if min(width1, width2) > 0 else 0
        
        return horizontal_overlap_ratio > 0.8  # 80% horizontal overlap
    
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
    
    def _refine_bounding_boxes(self, detections, img_shape):
        """Refine bounding boxes for better precision"""
        h, w = img_shape[:2]
        refined_detections = []
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Ensure integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure proper order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Clamp to image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Add small padding for better crop
            padding = 3
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            # Update detection with refined bbox
            refined_det = det.copy()
            refined_det['bbox'] = [x1_pad, y1_pad, x2_pad, y2_pad]
            refined_det['refined'] = True
            
            refined_detections.append(refined_det)
        
        return refined_detections
    
    def _extract_clean_crops(self, image_path, detections, output_dir):
        """Extract clean, precise crops"""
        img = cv2.imread(image_path)
        crops = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            
            # Validate crop
            if crop.size > 0 and crop.shape[0] > 8 and crop.shape[1] > 8:
                can_type = detection.get('can_type', 'can')
                is_separated = detection.get('original_stack', False)
                confidence = detection.get('confidence', 0)
                
                if is_separated:
                    sep_idx = detection.get('separation_index', 0)
                    crop_path = f"{output_dir}/stack_{can_type}_{i:03d}_part{sep_idx}_conf{confidence:.3f}.jpg"
                else:
                    crop_path = f"{output_dir}/single_{can_type}_{i:03d}_conf{confidence:.3f}.jpg"
                
                success = cv2.imwrite(crop_path, crop)
                if success:
                    crops.append(crop_path)
                    print(f"✅ Extracted crop {i}: {crop.shape} -> {crop_path}")
                else:
                    print(f"❌ Failed to save crop {i}")
            else:
                print(f"⚠️ Invalid crop {i}: {crop.shape if crop.size > 0 else 'empty'}")
        
        return crops
