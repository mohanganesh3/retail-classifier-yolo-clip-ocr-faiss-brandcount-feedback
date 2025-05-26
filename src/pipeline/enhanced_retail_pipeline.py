import json
import os
from collections import Counter
from src.detection.precise_can_detector import PreciseCanDetector
from src.classification.classifier import ProductClassifier
from src.classification.brand_manager import BrandManager
from src.visualization.brand_annotator import BrandAnnotator
from src.ocr.ocr_processor import OCRProcessor
from src.utils.config import ensure_directories

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

class EnhancedRetailPipeline:
    def __init__(self, use_ocr=True):
        ensure_directories()
        
        print("🚀 Initializing Enhanced Retail Pipeline with Brand Management")
        
        # Core components
        self.detector = PreciseCanDetector()
        self.brand_manager = BrandManager()
        self.brand_annotator = BrandAnnotator(self.brand_manager)
        
        # Classification
        try:
            self.classifier = ProductClassifier()
            self.classifier_available = self.classifier.index_available
            if self.classifier_available:
                print("✅ Classifier with FAISS index loaded successfully")
            else:
                print("⚠️ Classifier loaded but no FAISS index available")
        except Exception as e:
            print(f"❌ Could not initialize classifier: {e}")
            self.classifier = None
            self.classifier_available = False
        
        # OCR
        self.use_ocr = use_ocr
        if use_ocr:
            try:
                self.ocr_processor = OCRProcessor()
                print("✅ OCR processor initialized successfully")
            except Exception as e:
                print(f"⚠️ Could not initialize OCR: {e}")
                self.ocr_processor = None
                self.use_ocr = False
        else:
            self.ocr_processor = None
        
        print("✅ Enhanced pipeline ready with brand management and user interaction")
    
    def process_image(self, image_path, output_dir='temp/crops'):
        """Process image with brand classification and user interaction"""
        print(f"🔍 Processing image with brand management: {image_path}")
        
        # Check if vector database needs rebuilding
        if self.brand_manager.needs_rebuild():
            print("🔄 Vector database needs rebuilding...")
            if self.brand_manager.rebuild_vector_database():
                # Reinitialize classifier
                try:
                    self.classifier = ProductClassifier()
                    self.classifier_available = self.classifier.index_available
                except:
                    pass
        
        # Step 1: Detection
        print("Step 1: Detecting products...")
        detection_data, crop_paths = self.detector.detect_cans_precisely(image_path, output_dir)
        print(f"✅ Detected {len(crop_paths)} products")
        
        # Initialize results
        results = {
            'detection_data': detection_data,
            'crop_paths': crop_paths,
            'total_products': len(crop_paths),
            'image_path': image_path
        }
        
        if not crop_paths:
            print("⚠️ No products detected")
            results['brand_counts'] = {}
            return results
        
        # Step 2: OCR
        ocr_texts = []
        if self.use_ocr and self.ocr_processor:
            print("Step 2: OCR text extraction...")
            try:
                ocr_texts = self.ocr_processor.process_crops(crop_paths)
                print(f"✅ OCR processed {len(ocr_texts)} crops")
                results['ocr_texts'] = ocr_texts
            except Exception as e:
                print(f"❌ OCR failed: {e}")
                ocr_texts = [""] * len(crop_paths)
        else:
            ocr_texts = [""] * len(crop_paths)
        
        # Step 3: Classification
        classifications = []
        if self.classifier and self.classifier_available:
            print("Step 3: Classifying products...")
            try:
                classifications = self.classifier.classify_crops(crop_paths, ocr_texts)
                print(f"✅ Classified {len(classifications)} products")
            except Exception as e:
                print(f"❌ Classification failed: {e}")
                classifications = self._generate_fallback_classifications(crop_paths, ocr_texts)
        else:
            print("Step 3: Using fallback classification...")
            classifications = self._generate_fallback_classifications(crop_paths, ocr_texts)
        
        # Step 4: Brand Management with User Interaction
        print("Step 4: Brand classification and counting...")
        try:
            # First pass - automatic classification for high confidence items
            brand_counts = Counter()
            uncertain_crops = []
            
            for i, classification in enumerate(classifications):
                sku_id = classification.get('sku_id', 'UNKNOWN')
                confidence = classification.get('confidence', 0)
                crop_path = classification.get('crop_path', '')
                
                if sku_id in self.brand_manager.brand_mapping and confidence >= self.brand_manager.confidence_threshold:
                    # High confidence classification
                    brand_name = self.brand_manager.brand_mapping[sku_id]
                    brand_counts[brand_name] += 1
                    print(f"✅ Auto-classified: {brand_name} (conf: {confidence:.3f})")
                else:
                    # Low confidence or unknown - needs user input
                    uncertain_crops.append({
                        'crop_path': crop_path,
                        'predicted_brand': self.brand_manager.brand_mapping.get(sku_id, 'UNKNOWN'),
                        'confidence': confidence,
                        'index': i,
                        'sku_id': sku_id
                    })
                    print(f"❓ Needs user input: {sku_id} (conf: {confidence:.3f})")
            
            results['automatic_brand_counts'] = dict(brand_counts)
            results['brand_counts'] = dict(brand_counts)  # Will be updated with user feedback
            results['classifications'] = classifications
            results['uncertain_crops'] = uncertain_crops
            
            if uncertain_crops:
                print(f"⚠️ Found {len(uncertain_crops)} uncertain classifications requiring user input")
                results['needs_user_feedback'] = True
            else:
                results['needs_user_feedback'] = False
                print(f"✅ All products automatically classified")
            
        except Exception as e:
            print(f"❌ Brand classification failed: {e}")
            # Fallback to SKU counts
            sku_counts = self._count_by_sku(classifications)
            results['brand_counts'] = self.brand_manager.get_brand_display_names(sku_counts)
            results['classifications'] = classifications
            results['uncertain_crops'] = []
            results['needs_user_feedback'] = False
        
        # Step 5: Create Branded Annotation
        print("Step 5: Creating branded annotation...")
        try:
            annotated_img = self.brand_annotator.create_branded_annotation(
                image_path, detection_data, classifications, results['brand_counts']
            )
            
            if annotated_img:
                # Save annotated image
                annotation_path = 'temp/branded_annotation.png'
                self.brand_annotator.save_annotated_image(annotated_img, annotation_path)
                results['annotated_image_path'] = annotation_path
            
        except Exception as e:
            print(f"❌ Annotation failed: {e}")
        
        # Step 6: Generate Summary
        summary = self._generate_enhanced_summary(results)
        results['summary'] = summary
        
        # Save results
        try:
            with open('enhanced_brand_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("💾 Results saved to enhanced_brand_results.json")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        # Print final brand counts
        print(f"\n🎯 FINAL BRAND COUNTS:")
        if results['brand_counts']:
            for brand, count in results['brand_counts'].items():
                print(f"   {brand}: {count}")
        else:
            print(f"   Total detected products: {len(crop_paths)}")
        
        return results
    
    def _generate_fallback_classifications(self, crop_paths, ocr_texts):
        """Generate fallback classifications when FAISS is unavailable"""
        classifications = []
        
        for i, crop_path in enumerate(crop_paths):
            ocr_text = ocr_texts[i] if i < len(ocr_texts) else ""
            
            # Try to classify based on OCR
            brand = self._classify_by_ocr_text(ocr_text)
            
            classifications.append({
                'crop_path': crop_path,
                'sku_id': brand,
                'confidence': 0.5 if brand != 'UNKNOWN' else 0.1,
                'ocr_text': ocr_text,
                'method': 'ocr_fallback'
            })
        
        return classifications
    
    def _classify_by_ocr_text(self, ocr_text):
        """Classify based on OCR text patterns"""
        if not ocr_text:
            return 'UNKNOWN'
        
        text_lower = ocr_text.lower()
        
        # Brand patterns
        if any(word in text_lower for word in ['coca', 'coke', 'cola']):
            return 'coca_cola'
        elif 'pepsi' in text_lower:
            return 'pepsi'
        elif 'sprite' in text_lower:
            return 'sprite'
        elif 'fanta' in text_lower:
            return 'fanta'
        elif any(word in text_lower for word in ['7up', 'seven']):
            return 'seven_up'
        elif any(word in text_lower for word in ['mountain', 'dew']):
            return 'mountain_dew'
        elif 'red bull' in text_lower or 'redbull' in text_lower:
            return 'red_bull'
        elif 'monster' in text_lower:
            return 'monster'
        elif 'tango' in text_lower:
            return 'tango'
        else:
            return 'UNKNOWN'
    
    def _count_by_sku(self, classifications):
        """Count products by SKU"""
        sku_counts = Counter()
        
        for classification in classifications:
            sku_id = classification.get('sku_id', 'UNKNOWN')
            confidence = classification.get('confidence', 0)
            
            if confidence > 0.3:  # Only count confident classifications
                sku_counts[sku_id] += 1
        
        return dict(sku_counts)
    
    def _generate_enhanced_summary(self, results):
        """Generate enhanced summary with brand information"""
        summary = {
            'total_detections': len(results.get('detection_data', [])),
            'total_products': results.get('total_products', 0),
            'brands_detected': len(results.get('brand_counts', {})),
            'brand_counts': results.get('brand_counts', {}),
            'ocr_enabled': self.use_ocr,
            'classifier_available': self.classifier_available,
            'vector_db_needs_rebuild': self.brand_manager.needs_rebuild()
        }
        
        # Add brand diversity metrics
        if results.get('brand_counts'):
            total_products = sum(results['brand_counts'].values())
            summary['brand_diversity'] = len(results['brand_counts']) / total_products if total_products > 0 else 0
            summary['dominant_brand'] = max(results['brand_counts'].items(), key=lambda x: x[1])
        
        return summary

    def process_user_feedback_batch(self, feedback_results):
        """Process batch user feedback and return brand counts"""
        return self.brand_manager.process_user_feedback_batch(feedback_results)
