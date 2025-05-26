import os
import re
from collections import Counter
import cv2
import pandas as pd
import numpy as np
# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

class BrandClassifier:
    def __init__(self):
        print("🏷️ Initializing Brand Classifier")
        
        # Brand detection patterns (case-insensitive)
        self.brand_patterns = {
            'COCA_COLA': [
                r'coca.?cola', r'coke', r'coca', r'cola',
                r'cc', r'cocacola'
            ],
            'PEPSI': [
                r'pepsi', r'peps', r'pepsi.?cola'
            ],
            'SPRITE': [
                r'sprite', r'spri', r'spnt', r'spne', r'spnl',
                r'lemon.?lime', r'citrus'
            ],
            'FANTA': [
                r'fanta', r'fanva', r'orange', r'fant'
            ],
            'SEVEN_UP': [
                r'7.?up', r'seven.?up', r'7up', r'sevenup'
            ],
            'MOUNTAIN_DEW': [
                r'mountain.?dew', r'mtn.?dew', r'dew'
            ],
            'RED_BULL': [
                r'red.?bull', r'redbull', r'energy'
            ],
            'MONSTER': [
                r'monster', r'energy'
            ],
            'TANGO': [
                r'tango', r'tang'
            ],
            'DR_PEPPER': [
                r'dr.?pepper', r'doctor.?pepper', r'pepper'
            ]
        }
        
        # Color-based detection (backup method)
        self.color_patterns = {
            'COCA_COLA': ['red', 'crimson'],
            'PEPSI': ['blue', 'navy'],
            'SPRITE': ['green', 'lime'],
            'FANTA': ['orange', 'yellow'],
            'SEVEN_UP': ['green', 'lime'],
            'MOUNTAIN_DEW': ['yellow', 'green']
        }
        
        print("✅ Brand classifier initialized with pattern matching")
    
    def classify_by_ocr(self, ocr_text):
        """Classify brand based on OCR text"""
        if not ocr_text:
            return None, 0.0
        
        ocr_lower = ocr_text.lower().strip()
        
        # Direct pattern matching
        for brand, patterns in self.brand_patterns.items():
            for pattern in patterns:
                if re.search(pattern, ocr_lower):
                    confidence = self._calculate_confidence(pattern, ocr_lower)
                    print(f"🎯 OCR Match: '{ocr_text}' -> {brand} (conf: {confidence:.3f})")
                    return brand, confidence
        
        return None, 0.0
    
    def classify_by_visual_features(self, crop_path):
        """Classify based on visual features (color, shape)"""
        try:
            import cv2
            import numpy as np
            
            img = cv2.imread(crop_path)
            if img is None:
                return None, 0.0
            
            # Analyze dominant colors
            dominant_colors = self._get_dominant_colors(img)
            
            # Match colors to brands
            for brand, colors in self.color_patterns.items():
                for color in colors:
                    if self._color_matches(dominant_colors, color):
                        print(f"🎨 Color Match: {crop_path} -> {brand} (color: {color})")
                        return brand, 0.6  # Medium confidence for color matching
            
            return None, 0.0
            
        except Exception as e:
            print(f"⚠️ Visual classification failed: {e}")
            return None, 0.0
    
    def classify_crops(self, crop_paths, ocr_texts=None):
        """Classify all crops and return brand counts"""
        print(f"🔍 Classifying {len(crop_paths)} crops for brand identification")
        
        classifications = []
        
        for i, crop_path in enumerate(crop_paths):
            # Get OCR text for this crop
            ocr_text = ""
            if ocr_texts and i < len(ocr_texts):
                ocr_text = ocr_texts[i] or ""
            
            # Try OCR-based classification first
            brand, confidence = self.classify_by_ocr(ocr_text)
            
            # If OCR fails, try visual classification
            if not brand or confidence < 0.3:
                visual_brand, visual_conf = self.classify_by_visual_features(crop_path)
                if visual_brand and visual_conf > confidence:
                    brand, confidence = visual_brand, visual_conf
            
            # Store classification result
            classification = {
                'crop_path': crop_path,
                'brand': brand or 'UNKNOWN',
                'confidence': confidence,
                'ocr_text': ocr_text,
                'method': 'ocr' if brand and confidence > 0.5 else 'visual' if brand else 'unknown'
            }
            
            classifications.append(classification)
            
            if brand:
                print(f"✅ Classified {i+1}: {brand} (conf: {confidence:.3f}) - '{ocr_text}'")
            else:
                print(f"⚪ Unclassified {i+1}: UNKNOWN - '{ocr_text}'")
        
        return classifications
    
    def count_brands(self, classifications):
        """Count products by brand"""
        brand_counts = Counter()
        
        for cls in classifications:
            brand = cls.get('brand', 'UNKNOWN')
            confidence = cls.get('confidence', 0)
            
            # Only count confident classifications
            if confidence > 0.2:  # Lower threshold for real-world conditions
                brand_counts[brand] += 1
        
        # Convert to regular dict and sort
        final_counts = dict(brand_counts.most_common())
        
        print(f"📊 Final Brand Counts: {final_counts}")
        return final_counts
    
    def _calculate_confidence(self, pattern, text):
        """Calculate confidence based on pattern match quality"""
        # Exact match gets highest confidence
        if pattern in text:
            return 0.9
        
        # Partial match gets medium confidence
        pattern_clean = pattern.replace(r'\.?', '').replace(r'\?', '')
        if pattern_clean in text:
            return 0.7
        
        # Fuzzy match gets lower confidence
        return 0.5
    
    def _get_dominant_colors(self, img):
        """Extract dominant colors from image"""
        try:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Reshape to list of pixels
            pixels = img_rgb.reshape(-1, 3)
            
            # Get unique colors and their counts
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Get top 3 most common colors
            top_indices = np.argsort(counts)[-3:]
            dominant_colors = unique_colors[top_indices]
            
            return dominant_colors
            
        except Exception as e:
            print(f"Color analysis failed: {e}")
            return []
    
    def _color_matches(self, dominant_colors, target_color):
        """Check if any dominant color matches target color"""
        color_ranges = {
            'red': ([150, 0, 0], [255, 100, 100]),
            'blue': ([0, 0, 150], [100, 100, 255]),
            'green': ([0, 150, 0], [100, 255, 100]),
            'orange': ([200, 100, 0], [255, 200, 100]),
            'yellow': ([200, 200, 0], [255, 255, 150])
        }
        
        if target_color not in color_ranges:
            return False
        
        min_color, max_color = color_ranges[target_color]
        
        for color in dominant_colors:
            if (min_color[0] <= color[0] <= max_color[0] and
                min_color[1] <= color[1] <= max_color[1] and
                min_color[2] <= color[2] <= max_color[2]):
                return True
        
        return False
