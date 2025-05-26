import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

class CanTinDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = 0.001  # Ultra-low for cans/tins
        
        print(f"🥤 Initializing Specialized Can/Tin Detector on {self.device}")
        
        # Load multiple YOLO models for better detection
        self.models = []
        
        # Try YOLOv8x first (best accuracy)
        try:
            self.primary_model = YOLO('yolov8x.pt')
            self.models.append(('YOLOv8x', self.primary_model))
            print("✅ YOLOv8x loaded for can/tin detection")
        except:
            pass
        
        # Add YOLOv8l as backup
        try:
            self.secondary_model = YOLO('yolov8l.pt')
            self.models.append(('YOLOv8l', self.secondary_model))
            print("✅ YOLOv8l loaded as backup")
        except:
            pass
        
        # Fallback to YOLOv8n
        if not self.models:
            self.fallback_model = YOLO('yolov8n.pt')
            self.models.append(('YOLOv8n', self.fallback_model))
            print("✅ YOLOv8n loaded as fallback")
        
        # COCO classes that could be cans/tins
        self.can_tin_classes = [
            'bottle',           # Often misclassified
            'cup',              # Cans often detected as cups
            'wine glass',       # Tall cans
            'bowl',             # Short cans
            'cell phone',       # Rectangular cans
            'remote',           # Rectangular objects
            'book',             # Flat rectangular objects
            'laptop',           # Larger rectangular objects
            'mouse',            # Small cylindrical objects
            'scissors',         # Metallic objects
            'knife',            # Metallic objects
            'spoon',            # Metallic objects
            'fork',             # Metallic objects
            'banana',           # Curved objects
            'hot dog',          # Cylindrical food items
            'donut',            # Circular objects
            'cake',             # Cylindrical objects
            'clock',            # Circular objects
            'vase',             # Cylindrical containers
            'teddy bear',       # Sometimes misclassified
            'sports ball',      # Round objects
            'frisbee',          # Circular objects
            'baseball bat',     # Cylindrical objects
            'tennis racket',    # Objects with handles
            'skateboard',       # Rectangular objects
            'surfboard',        # Long rectangular objects
            'snowboard',        # Rectangular objects
            'kite',             # Flat objects
            'baseball glove',   # Curved objects
            'tie',              # Vertical objects
            'suitcase',         # Rectangular containers
            'handbag',          # Containers
            'backpack',         # Containers
            'umbrella',         # Cylindrical when closed
            'hair drier',       # Cylindrical appliances
            'toothbrush'        # Cylindrical objects
        ]
        
        print(f"🎯 Monitoring {len(self.can_tin_classes)} object classes for can/tin detection")
    
    def detect_cans_and_tins(self, image_path, output_dir='temp/crops'):
        """Specialized detection for cans and tins with multiple strategies"""
        print(f"🔍 Specialized Can/Tin Detection: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return [], []
        
        # Load and validate image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return [], []
        
        h, w = img.shape[:2]
        print(f"📸 Image dimensions: {w}x{h}")
        
        # Strategy 1: Multi-model detection
        all_detections = self._multi_model_detection(image_path)
        
        # Strategy 2: Shape-based detection
        shape_detections = self._shape_based_detection(img)
        
        # Strategy 3: Color-based detection for metallic objects
        color_detections = self._color_based_detection(img)
        
        # Combine all detections
        combined_detections = self._combine_detections(all_detections, shape_detections, color_detections)
        
        # Filter and validate detections
        final_detections = self._filter_can_tin_detections(combined_detections, img.shape)
        
        print(f"🎯 Total can/tin detections: {len(final_detections)}")
        
        # Extract crops
        crops = []
        if final_detections:
            crops = self._extract_precise_crops(image_path, final_detections, output_dir)
        
        return final_detections, crops
    
    def _multi_model_detection(self, image_path):
        """Run detection with multiple YOLO models"""
        all_detections = []
        
        for model_name, model in self.models:
            print(f"🔍 Running {model_name} detection...")
            
            # Try multiple confidence levels
            for conf_level in [0.001, 0.005, 0.01, 0.02]:
                try:
                    results = model(image_path, conf=conf_level, iou=0.1, verbose=False)
                    
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                            bboxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy()
                            
                            # Get class names
                            class_names = [model.names[int(cls_id)] for cls_id in class_ids]
                            
                            # Filter for can/tin classes
                            for bbox, conf, class_name in zip(bboxes, confidences, class_names):
                                if class_name in self.can_tin_classes:
                                    x1, y1, x2, y2 = bbox
                                    width = x2 - x1
                                    height = y2 - y1
                                    aspect_ratio = height / width if width > 0 else 0
                                    
                                    # Can/tin shape validation
                                    if self._is_can_tin_shape(width, height, aspect_ratio, class_name):
                                        all_detections.append({
                                            'bbox': bbox.tolist(),
                                            'confidence': float(conf),
                                            'class_name': class_name,
                                            'model': model_name,
                                            'detection_type': self._classify_can_tin_type(class_name, aspect_ratio),
                                            'aspect_ratio': aspect_ratio
                                        })
                                        
                                        print(f"✅ {model_name} detected: {class_name} -> CAN/TIN ({conf:.3f})")
                
                except Exception as e:
                    print(f"⚠️ {model_name} detection failed at conf {conf_level}: {e}")
                    continue
        
        return all_detections
    
    def _shape_based_detection(self, img):
        """Detect cylindrical shapes that could be cans/tins"""
        print("🔍 Shape-based can/tin detection...")
        
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circles (top/bottom of cans)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # Create bounding box for circular detection
                    # Assume can height is 2-4 times the radius
                    can_height = r * 3
                    can_width = r * 2
                    
                    x1 = max(0, x - can_width // 2)
                    y1 = max(0, y - can_height // 2)
                    x2 = min(img.shape[1], x + can_width // 2)
                    y2 = min(img.shape[0], y + can_height // 2)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 0.6,
                        'class_name': 'circular_can',
                        'detection_type': 'shape_based',
                        'circle_center': (x, y),
                        'circle_radius': r
                    })
                    
                    print(f"✅ Shape detection: Circular can at ({x}, {y}) radius {r}")
            
            # Detect rectangles (side view of cans)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for rectangular shapes (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Can-like aspect ratio and size
                    if (1.5 < aspect_ratio < 5.0 and 
                        w > 20 and h > 40 and 
                        w < 200 and h < 400):
                        
                        detections.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.5,
                            'class_name': 'rectangular_can',
                            'detection_type': 'shape_based',
                            'aspect_ratio': aspect_ratio
                        })
                        
                        print(f"✅ Shape detection: Rectangular can {w}x{h} AR:{aspect_ratio:.2f}")
        
        except Exception as e:
            print(f"⚠️ Shape-based detection failed: {e}")
        
        return detections
    
    def _color_based_detection(self, img):
        """Detect metallic colors typical of cans/tins"""
        print("🔍 Color-based metallic detection...")
        
        detections = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define metallic color ranges
            metallic_ranges = [
                # Silver/aluminum
                ([0, 0, 180], [180, 30, 255]),
                # Gold/yellow metallic
                ([15, 100, 100], [35, 255, 255]),
                # Red metallic (Coca-Cola)
                ([0, 120, 70], [10, 255, 255]),
                # Blue metallic (Pepsi)
                ([100, 150, 50], [130, 255, 255]),
            ]
            
            for i, (lower, upper) in enumerate(metallic_ranges):
                lower = np.array(lower)
                upper = np.array(upper)
                
                # Create mask for this color range
                mask = cv2.inRange(hsv, lower, upper)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Filter by area (cans should have reasonable size)
                    if 500 < area < 50000:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = h / w if w > 0 else 0
                        
                        # Can-like proportions
                        if 1.2 < aspect_ratio < 6.0:
                            detections.append({
                                'bbox': [x, y, x + w, y + h],
                                'confidence': 0.4,
                                'class_name': f'metallic_can_{i}',
                                'detection_type': 'color_based',
                                'aspect_ratio': aspect_ratio,
                                'area': area
                            })
                            
                            print(f"✅ Color detection: Metallic can {w}x{h} area:{area}")
        
        except Exception as e:
            print(f"⚠️ Color-based detection failed: {e}")
        
        return detections
    
    def _is_can_tin_shape(self, width, height, aspect_ratio, class_name):
        """Validate if detected object has can/tin-like shape"""
        # Size validation
        if width < 15 or height < 30 or width > 300 or height > 500:
            return False
        
        # Aspect ratio validation based on class
        if class_name == 'bottle':
            return 1.5 < aspect_ratio < 6.0
        elif class_name in ['cup', 'wine glass']:
            return 1.2 < aspect_ratio < 5.0
        elif class_name == 'bowl':
            return 0.5 < aspect_ratio < 2.0
        elif class_name in ['cell phone', 'remote', 'book']:
            return 1.0 < aspect_ratio < 4.0
        else:
            return 1.0 < aspect_ratio < 6.0
    
    def _classify_can_tin_type(self, class_name, aspect_ratio):
        """Classify the type of can/tin based on detection"""
        if aspect_ratio > 3.0:
            return 'tall_can'
        elif aspect_ratio < 1.5:
            return 'short_can'
        elif class_name == 'bottle':
            return 'bottle_can'
        else:
            return 'standard_can'
    
    def _combine_detections(self, yolo_detections, shape_detections, color_detections):
        """Combine detections from all methods"""
        all_detections = []
        
        # Add YOLO detections (highest priority)
        all_detections.extend(yolo_detections)
        
        # Add shape detections
        all_detections.extend(shape_detections)
        
        # Add color detections
        all_detections.extend(color_detections)
        
        print(f"📊 Combined detections: {len(yolo_detections)} YOLO + {len(shape_detections)} shape + {len(color_detections)} color = {len(all_detections)} total")
        
        return all_detections
    
    def _filter_can_tin_detections(self, detections, img_shape):
        """Filter and remove duplicate detections"""
        if not detections:
            return []
        
        h, w = img_shape[:2]
        filtered = []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Check if this detection overlaps significantly with existing ones
            is_duplicate = False
            for existing in filtered:
                overlap = self._calculate_overlap(bbox, existing['bbox'])
                if overlap > 0.3:  # 30% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        print(f"🔧 Filtered detections: {len(detections)} -> {len(filtered)}")
        return filtered
    
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
    
    def _extract_precise_crops(self, image_path, detections, output_dir):
        """Extract precise crops for detected cans/tins"""
        img = cv2.imread(image_path)
        crops = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Add minimal padding
            padding = 3
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = img[y1:y2, x1:x2]
            
            # Name crop based on detection type
            detection_type = detection.get('detection_type', 'can')
            crop_path = f"{output_dir}/can_tin_{detection_type}_{i:04d}.jpg"
            
            cv2.imwrite(crop_path, crop)
            crops.append(crop_path)
            
            print(f"✅ Extracted {detection_type} crop {i}: {crop.shape} -> {crop_path}")
        
        return crops
