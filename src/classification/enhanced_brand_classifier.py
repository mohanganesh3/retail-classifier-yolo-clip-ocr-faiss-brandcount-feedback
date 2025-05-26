import os
import re
from collections import Counter
import cv2
import pandas as pd
import numpy as np

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

class EnhancedBrandClassifier:
    def __init__(self):
        print("🏷️ Initializing Enhanced Brand Classifier")
        
        # Enhanced brand detection patterns (case-insensitive)
        self.brand_patterns = {
            'COCA-COLA': [
                r'coca.?cola', r'coke', r'coca', r'cola',
                r'cc', r'cocacola', r'classic'
            ],
            'PEPSI': [
                r'pepsi', r'peps', r'pepsi.?cola', r'pepsi.?max'
            ],
            'SPRITE': [
                r'sprite', r'spri', r'spnt', r'spne', r'spnl',
                r'lemon.?lime', r'citrus'
            ],
            'FANTA': [
                r'fanta', r'fanva', r'orange', r'fant'
            ],
            'SEVEN-UP': [
                r'7.?up', r'seven.?up', r'7up', r'sevenup', r'seven'
            ],
            'MOUNTAIN-DEW': [
                r'mountain.?dew', r'mtn.?dew', r'dew'
            ],
            'RED-BULL': [
                r'red.?bull', r'redbull', r'energy'
            ],
            'MONSTER': [
                r'monster', r'energy'
            ],
            'TANGO': [
                r'tango', r'tang'
            ],
            'DR-PEPPER': [
                r'dr.?pepper', r'doctor.?pepper', r'pepper'
            ]
        }
        
        # Visual color patterns for brand identification
        self.brand_colors = {
            'COCA-COLA': [(255, 0, 0), (220, 20, 20), (200, 0, 0)],  # Red variants
            'PEPSI': [(0, 0, 255), (30, 30, 200), (0, 50, 255)],     # Blue variants
            'SPRITE': [(0, 255, 0), (50, 255, 50), (0, 200, 0)],     # Green variants
            'FANTA': [(255, 165, 0), (255, 140, 0), (255, 100, 0)],  # Orange variants
            'SEVEN-UP': [(0, 255, 0), (50, 255, 50), (100, 255, 100)] # Green variants
        }
        
        print("✅ Enhanced brand classifier initialized with comprehensive pattern matching")
    
    def classify_crops_with_brands(self, crop_paths, ocr_texts=None):
        """Classify crops and return actual brand names"""
        print(f"🔍 Enhanced brand classification for {len(crop_paths)} crops")
        
        classifications = []
        
        for i, crop_path in enumerate(crop_paths):
            # Get OCR text for this crop
            ocr_text = ""
            if ocr_texts and i < len(ocr_texts):
                ocr_text = ocr_texts[i] or ""
            
            # Try OCR-based classification first
            brand_name, ocr_confidence = self.classify_by_ocr_enhanced(ocr_text)
            
            # If OCR fails or low confidence, try visual classification
            if brand_name == 'UNKNOWN' or ocr_confidence < 0.5:
                visual_brand, visual_confidence = self.classify_by_visual_features_enhanced(crop_path)
                if visual_brand != 'UNKNOWN' and visual_confidence > ocr_confidence:
                    brand_name, final_confidence = visual_brand, visual_confidence
                else:
                    final_confidence = ocr_confidence
            else:
                final_confidence = ocr_confidence
            
            # Create classification result with actual brand name
            classification = {
                'crop_path': crop_path,
                'sku_id': brand_name,  # This is the key - use brand name as SKU ID
                'brand': brand_name,
                'confidence': final_confidence,
                'ocr_text': ocr_text,
                'method': 'enhanced_brand_classification'
            }
            
            classifications.append(classification)
            
            if brand_name != 'UNKNOWN':
                print(f"✅ Classified crop {i+1}: {brand_name} (conf: {final_confidence:.3f}) - '{ocr_text}'")
            else:
                print(f"⚪ Unclassified crop {i+1}: UNKNOWN - '{ocr_text}'")
        
        return classifications
    
    def classify_by_ocr_enhanced(self, ocr_text):
        """Enhanced OCR-based brand classification"""
        if not ocr_text:
            return 'UNKNOWN', 0.0
        
        ocr_lower = ocr_text.lower().strip()
        
        # Direct pattern matching with confidence scoring
        for brand, patterns in self.brand_patterns.items():
            for pattern in patterns:
                if re.search(pattern, ocr_lower):
                    confidence = self._calculate_ocr_confidence(pattern, ocr_lower, ocr_text)
                    print(f"🎯 OCR Brand Match: '{ocr_text}' -> {brand} (conf: {confidence:.3f})")
                    return brand, confidence
        
        return 'UNKNOWN', 0.0
    
    def classify_by_visual_features_enhanced(self, crop_path):
        """Enhanced visual classification based on colors and shapes"""
        try:
            img = cv2.imread(crop_path)
            if img is None:
                return 'UNKNOWN', 0.0
            
            # Analyze dominant colors
            dominant_colors = self._get_dominant_colors_enhanced(img)
            
            # Match colors to brands with confidence scoring
            best_brand = 'UNKNOWN'
            best_confidence = 0.0
            
            for brand, brand_color_list in self.brand_colors.items():
                color_confidence = self._calculate_color_match_confidence(dominant_colors, brand_color_list)
                
                if color_confidence > best_confidence:
                    best_brand = brand
                    best_confidence = color_confidence
            
            if best_confidence > 0.3:  # Threshold for visual classification
                print(f"🎨 Visual Brand Match: {crop_path} -> {best_brand} (conf: {best_confidence:.3f})")
                return best_brand, best_confidence
            
            return 'UNKNOWN', 0.0
            
        except Exception as e:
            print(f"⚠️ Visual classification failed: {e}")
            return 'UNKNOWN', 0.0
    
    def _calculate_ocr_confidence(self, pattern, text_lower, original_text):
        """Calculate confidence based on OCR match quality"""
        # Exact match gets highest confidence
        if pattern in text_lower:
            # Bonus for longer matches
            match_length = len(pattern)
            text_length = len(original_text)
            length_bonus = min(0.3, match_length / text_length)
            return min(0.95, 0.7 + length_bonus)
        
        # Partial match gets medium confidence
        pattern_clean = pattern.replace(r'\.?', '').replace(r'\?', '')
        if pattern_clean in text_lower:
            return 0.6
        
        # Fuzzy match gets lower confidence
        return 0.4
    
    def _get_dominant_colors_enhanced(self, img):
        """Extract dominant colors with better analysis"""
        try:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing
            small_img = cv2.resize(img_rgb, (50, 50))
            
            # Reshape to list of pixels
            pixels = small_img.reshape(-1, 3)
            
            # Use k-means clustering to find dominant colors
            from sklearn.cluster import KMeans
            
            # Find 5 dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            return dominant_colors
            
        except Exception as e:
            print(f"Color analysis failed: {e}")
            # Fallback to simple method
            return self._get_dominant_colors_simple(img)
    
    def _get_dominant_colors_simple(self, img):
        """Simple dominant color extraction fallback"""
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img_rgb.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            top_indices = np.argsort(counts)[-5:]  # Top 5 colors
            return unique_colors[top_indices]
        except:
            return np.array([[128, 128, 128]])  # Gray fallback
    
    def _calculate_color_match_confidence(self, dominant_colors, brand_colors):
        """Calculate confidence based on color matching"""
        if len(dominant_colors) == 0:
            return 0.0
        
        max_confidence = 0.0
        
        for dominant_color in dominant_colors:
            for brand_color in brand_colors:
                # Calculate color distance
                distance = np.linalg.norm(dominant_color - np.array(brand_color))
                
                # Convert distance to confidence (closer = higher confidence)
                # Max distance for RGB is ~441 (sqrt(255^2 * 3))
                confidence = max(0.0, 1.0 - (distance / 441.0))
                
                # Apply threshold - colors need to be reasonably close
                if distance < 100:  # Threshold for "similar" colors
                    confidence *= 1.5  # Boost confidence for close matches
                
                max_confidence = max(max_confidence, confidence)
        
        return min(1.0, max_confidence)
    
    def count_brands(self, classifications):
        """Count products by brand from enhanced classifications"""
        brand_counts = Counter()
        
        for cls in classifications:
            brand = cls.get('brand', 'UNKNOWN')
            confidence = cls.get('confidence', 0)
            
            # Only count confident classifications
            if confidence > 0.3 and brand != 'UNKNOWN':
                brand_counts[brand] += 1
        
        # Convert to regular dict and sort
        final_counts = dict(brand_counts.most_common())
        
        print(f"📊 Enhanced Brand Counts: {final_counts}")
        return final_counts
