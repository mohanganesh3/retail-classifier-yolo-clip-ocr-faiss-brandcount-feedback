import json
import os
from collections import Counter
from src.detection.detector import ProductDetector
from src.classification.classifier import ProductClassifier
from src.ocr.ocr_processor import OCRProcessor
from src.utils.config import ensure_directories

# Fix OpenMP conflicts at import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

class RetailShelfPipeline:
    def __init__(self, 
                 detector_model='models/detector/yolov7-w6.pt',
                 use_ocr=True,
                 use_yolov7=True,
                 confidence_threshold=0.35):  # Changed from 0.6 to 0.25
        ensure_directories()
        
        self.detector = ProductDetector(detector_model, use_yolov7=use_yolov7)
        self.use_ocr = use_ocr
        self.use_yolov7 = use_yolov7
        self.confidence_threshold = confidence_threshold
        
        # Try to initialize classifier, but don't fail if prototypes aren't built
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
        
        # Initialize OCR processor
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
    
    def process_image(self, image_path, output_dir='temp/crops'):
        """Process a single shelf image through the complete pipeline with user feedback"""
        model_name = "YOLOv7-W6" if self.use_yolov7 else "YOLOv8"
        
        print(f"🚀 Processing image with {model_name} and user feedback: {image_path}")
        
        # Step 1: Detection with bounding box merging
        print(f"Step 1: Detecting products with {model_name} and merging split detections...")
        detection_data, crop_paths = self.detector.detect_with_fallback(image_path, output_dir)
        print(f"✅ Detected {len(crop_paths)} products (after merging)")
        
        # Initialize results structure
        results = {
            'detection_data': detection_data,
            'classifications': [],
            'sku_counts': {},
            'total_products': len(crop_paths),
            'ocr_enabled': self.use_ocr,
            'classifier_available': self.classifier_available,
            'model_used': model_name,
            'image_path': image_path,
            'confidence_threshold': self.confidence_threshold
        }
        
        if not crop_paths:
            print("⚠️ No products detected")
            return results
        
        # Step 2: OCR (optional)
        ocr_texts = None
        if self.use_ocr and self.ocr_processor:
            print("Step 2: Extracting text from crops...")
            try:
                ocr_texts = self.ocr_processor.process_crops(crop_paths)
                print(f"✅ OCR processed {len(ocr_texts)} crops")
                results['ocr_texts'] = ocr_texts
            except Exception as e:
                print(f"❌ OCR processing failed: {e}")
                ocr_texts = None
        
        # Step 3: Classification with user feedback integration
        classifications = []
        uncertain_classifications = []
        
        if self.classifier and self.classifier_available:
            print("Step 3: Classifying products with feedback integration...")
            try:
                # Set OpenMP environment before classification
                os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                os.environ['OMP_NUM_THREADS'] = '1'
                
                # Get classifications with confidence scoring
                classifications, low_confidence = self.classifier.classify_crops_with_feedback(
                    crop_paths, ocr_texts, self.confidence_threshold
                )
                
                # Separate high and low confidence classifications
                high_confidence = []
                for i, cls in enumerate(classifications):
                    confidence = cls.get('confidence', 0)
                    if confidence >= self.confidence_threshold:
                        high_confidence.append(cls)
                    else:
                        uncertain_classifications.append({
                            'index': i,
                            'crop_path': cls.get('crop_path', ''),
                            'predicted_brand': cls.get('sku_id', 'UNKNOWN'),
                            'confidence': confidence,
                            'ocr_text': cls.get('ocr_text', ''),
                            'classification': cls
                        })
                
                print(f"✅ Classified {len(classifications)} products")
                print(f"📊 Confidence breakdown: {len(high_confidence)} high, {len(uncertain_classifications)} need review")
                
            except Exception as e:
                print(f"❌ Classification failed: {e}")
                # Generate fallback classifications for counting
                classifications = self._generate_detection_based_classifications(crop_paths, detection_data, ocr_texts)
        else:
            print("Step 3: Skipping classification (no classifier available)")
            classifications = self._generate_detection_based_classifications(crop_paths, detection_data, ocr_texts)
        
        # Store uncertain classifications for user feedback
        results['uncertain_classifications'] = uncertain_classifications
        results['needs_user_feedback'] = len(uncertain_classifications) > 0
        
        print(f"🔍 CLASSIFICATION METHOD USED:")
        if self.classifier and self.classifier_available:
            print(f"   ✅ FAISS Index Available: {len(self.classifier.labels)} SKUs")
            print(f"   ✅ OCR Enabled: {self.use_ocr}")
            print(f"   📊 Primary Method: OCR + FAISS with user feedback")
        else:
            print(f"   ⚠️ FAISS Index: Not available")
            print(f"   ✅ OCR Enabled: {self.use_ocr}")
            print(f"   📊 Primary Method: OCR + Detection fallback")

        if classifications:
            methods_used = [cls.get('method', 'unknown') for cls in classifications]
            method_counts = Counter(methods_used)
            print(f"   📈 Classification breakdown: {dict(method_counts)}")
        
        # Step 4: Enhanced Product Counting (only high confidence initially)
        print("Step 4: Counting products by type...")
        high_confidence_classifications = [cls for cls in classifications 
                                         if cls.get('confidence', 0) >= self.confidence_threshold]
        
        product_counts = self._count_products_enhanced(high_confidence_classifications, detection_data, ocr_texts)
        
        # Step 5: Post-processing (NMS, filtering)
        print("Step 5: Applying post-processing...")
        filtered_counts, filtered_detections = self._apply_post_processing(
            product_counts, detection_data, high_confidence_classifications
        )
        
        # Update results
        results.update({
            'classifications': classifications,
            'high_confidence_classifications': high_confidence_classifications,
            'sku_counts': filtered_counts,
            'product_counts': product_counts,
            'raw_sku_counts': product_counts,
            'filtered_detections': len(filtered_detections),
            'total_products': sum(filtered_counts.values()) if filtered_counts else len(crop_paths)
        })
        
        # Step 6: Count brands (only from confident classifications initially)
        confident_brand_counts = self._count_brands_from_classifications(high_confidence_classifications)
        results['brand_counts'] = confident_brand_counts
        
        # Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        # Save final results
        try:
            with open('final_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to final_results.json")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        print(f"🎯 Pipeline complete: {results['total_products']} products, {len(filtered_counts)} unique types")
        
        if uncertain_classifications:
            print(f"⚠️ {len(uncertain_classifications)} products need user review for improved accuracy")
        
        return results
    
    def process_user_feedback(self, feedback_data, original_results):
        """Process user feedback and update results"""
        print(f"🤔 Processing user feedback for {len(feedback_data)} items...")
        
        if not self.classifier:
            print("❌ No classifier available for feedback processing")
            return original_results
        
        # Process feedback through classifier
        updated_classifications, new_brands = self.classifier.process_user_feedback(feedback_data)
        
        # Update original classifications with user feedback
        updated_results = original_results.copy()
        
        # Replace uncertain classifications with user-confirmed ones
        all_classifications = updated_results.get('classifications', [])
        
        for feedback in feedback_data:
            index = feedback.get('index', -1)
            if 0 <= index < len(all_classifications):
                # Find corresponding updated classification
                for updated_cls in updated_classifications:
                    if updated_cls.get('crop_path') == feedback.get('crop_path'):
                        all_classifications[index] = updated_cls
                        break
        
        updated_results['classifications'] = all_classifications
        
        # Recalculate brand counts with user feedback
        all_confident_classifications = [cls for cls in all_classifications 
                                       if cls.get('confidence', 0) >= 0.8]  # Include user-confirmed
        
        updated_brand_counts = self._count_brands_from_classifications(all_confident_classifications)
        updated_results['brand_counts'] = updated_brand_counts
        
        # Update metadata
        updated_results['user_feedback_processed'] = True
        updated_results['new_brands_created'] = new_brands
        updated_results['uncertain_classifications'] = []  # Clear after processing
        updated_results['needs_user_feedback'] = False
        
        # Recalculate totals
        updated_results['total_products'] = sum(updated_brand_counts.values()) if updated_brand_counts else len(updated_results.get('detection_data', []))
        
        print(f"✅ User feedback processed. New brands: {new_brands}")
        
        # Save updated results
        try:
            with open('final_results_with_feedback.json', 'w') as f:
                json.dump(updated_results, f, indent=2, default=str)
            print(f"💾 Updated results saved to final_results_with_feedback.json")
        except Exception as e:
            print(f"⚠️ Could not save updated results: {e}")
        
        return updated_results
    
    def _generate_detection_based_classifications(self, crop_paths, detection_data, ocr_texts):
        """Generate classifications based on detection classes and OCR when FAISS is unavailable"""
        classifications = []
        
        print("📊 Generating detection-based classifications...")
        
        for i, crop_path in enumerate(crop_paths):
            # Get detection class
            detection_class = 'bottle'
            confidence = 0.5
            
            if i < len(detection_data):
                detection_class = detection_data[i].get('class_name', 'bottle')
                confidence = detection_data[i].get('confidence', 0.5)
            
            # Use OCR to determine product type
            product_type = self._classify_by_ocr_and_detection(detection_class, ocr_texts[i] if ocr_texts and i < len(ocr_texts) else "")
            
            classifications.append({
                'crop_path': crop_path,
                'sku_id': product_type,
                'confidence': float(confidence),
                'detection_class': detection_class,
                'ocr_text': ocr_texts[i] if ocr_texts and i < len(ocr_texts) else '',
                'method': 'detection_ocr_based'
            })
        
        print(f"✅ Generated {len(classifications)} detection-based classifications")
        return classifications
    
    def _classify_by_ocr_and_detection(self, detection_class, ocr_text):
        """Classify product based on OCR text and detection class"""
        ocr_lower = ocr_text.lower() if ocr_text else ""
        
        # Brand detection based on OCR
        if 'pepsi' in ocr_lower:
            return 'PEPSI'
        elif 'coca' in ocr_lower or 'coke' in ocr_lower:
            return 'COCA_COLA'
        elif 'sprite' in ocr_lower or 'spri' in ocr_lower or 'spnt' in ocr_lower:
            return 'SPRITE'
        elif 'fanta' in ocr_lower or 'fanva' in ocr_lower:
            return 'FANTA'
        elif 'tango' in ocr_lower:
            return 'TANGO'
        elif '7up' in ocr_lower or 'up' in ocr_lower:
            return '7UP'
        elif 'mountain' in ocr_lower or 'dew' in ocr_lower:
            return 'MOUNTAIN_DEW'
        elif 'red bull' in ocr_lower or 'redbull' in ocr_lower:
            return 'RED_BULL'
        elif 'monster' in ocr_lower:
            return 'MONSTER'
        elif 'energy' in ocr_lower:
            return 'ENERGY_DRINK'
        elif 'water' in ocr_lower:
            return 'WATER'
        elif 'juice' in ocr_lower:
            return 'JUICE'
        else:
            # Fallback based on detection class
            if detection_class == 'bottle':
                return 'BOTTLE_UNKNOWN'
            elif detection_class == 'cup':
                return 'CUP_UNKNOWN'
            else:
                return 'PRODUCT_UNKNOWN'
    
    def _count_products_enhanced(self, classifications, detection_data, ocr_texts):
        """Enhanced product counting with multiple methods"""
        product_counts = {}
        
        # Method 1: Count by classification
        if classifications:
            for cls in classifications:
                product_type = cls.get('sku_id', 'UNKNOWN')
                confidence = cls.get('confidence', 0)
                
                # Only count confident classifications
                if confidence > 0.2:  # Lower threshold for detection-based
                    if product_type in product_counts:
                        product_counts[product_type] += 1
                    else:
                        product_counts[product_type] = 1
        
        # Method 2: If no classifications, count by detection class
        if not product_counts and detection_data:
            detection_classes = [d.get('class_name', 'unknown') for d in detection_data]
            class_counts = Counter(detection_classes)
            
            for class_name, count in class_counts.items():
                product_counts[class_name.upper()] = count
        
        # Method 3: Fallback to total count
        if not product_counts:
            total_detections = len(detection_data) if detection_data else 0
            if total_detections > 0:
                product_counts['DETECTED_PRODUCTS'] = total_detections
        
        print(f"📊 Product counts: {product_counts}")
        return product_counts
    
    def _apply_post_processing(self, product_counts, detection_data, classifications):
        """Apply post-processing filters with improved NMS"""
        # Apply Non-Maximum Suppression to remove duplicate detections
        filtered_detections = self._apply_nms_enhanced(detection_data)
        
        # Adjust counts based on filtered detections
        if len(filtered_detections) != len(detection_data):
            # Recalculate counts proportionally
            ratio = len(filtered_detections) / len(detection_data) if detection_data else 1
            filtered_counts = {}
            
            for product_type, count in product_counts.items():
                adjusted_count = max(1, int(count * ratio))
                filtered_counts[product_type] = adjusted_count
        else:
            filtered_counts = product_counts.copy()
        
        print(f"📊 Filtered product counts: {filtered_counts}")
        return filtered_counts, filtered_detections
    
    def _apply_nms_enhanced(self, detection_data, iou_threshold=0.3):
        """Enhanced Non-Maximum Suppression for retail environments"""
        if not detection_data:
            return detection_data
        
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        valid_detections = []
        
        for det in detection_data:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                boxes.append(bbox)
                scores.append(det.get('confidence', 0))
                valid_detections.append(det)
        
        if not boxes:
            return detection_data
        
        try:
            import torch
            import torchvision.ops as ops
            
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            
            # Apply NMS with lower threshold for retail (products can be close)
            keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
            
            # Filter detections
            filtered_detections = [valid_detections[i] for i in keep_indices.tolist()]
            
            print(f"🔧 NMS: {len(detection_data)} -> {len(filtered_detections)} detections")
            return filtered_detections
            
        except Exception as e:
            print(f"⚠️ NMS failed: {e}, using original detections")
            return detection_data
    
    def _generate_summary(self, results):
        """Generate a summary of the processing results"""
        summary = {
            'total_detections': len(results.get('detection_data', [])),
            'total_classifications': len(results.get('classifications', [])),
            'unique_products': len(results.get('sku_counts', {})),
            'total_products': results.get('total_products', 0),
            'model_used': results.get('model_used', 'Unknown'),
            'classifier_available': results.get('classifier_available', False),
            'ocr_enabled': results.get('ocr_enabled', False),
            'needs_user_feedback': results.get('needs_user_feedback', False),
            'uncertain_count': len(results.get('uncertain_classifications', []))
        }
        
        # Add quality metrics
        if results.get('classifications'):
            confidences = [c.get('confidence', 0) for c in results['classifications']]
            summary['avg_confidence'] = sum(confidences) / len(confidences)
            summary['high_confidence_count'] = sum(1 for c in confidences if c > 0.7)
        
        return summary
    
    def process_batch(self, image_paths):
        """Process multiple images"""
        all_results = []
        
        for image_path in image_paths:
            result = self.process_image(image_path)
            all_results.append({
                'image_path': image_path,
                'results': result
            })
        
        return all_results

    def _count_brands_from_classifications(self, classifications):
        """Count brands from classifications"""
        brand_counts = {}
        
        for cls in classifications:
            sku_id = cls.get('sku_id', 'UNKNOWN')
            confidence = cls.get('confidence', 0)
        
            # Only count confident classifications - lowered threshold
            if confidence > 0.15 and sku_id != 'UNKNOWN':  # Changed from 0.5 to 0.15
                # Convert SKU to brand name
                brand_name = self._sku_to_brand_name(sku_id)
                if brand_name in brand_counts:
                    brand_counts[brand_name] += 1
                else:
                    brand_counts[brand_name] = 1
        
        return brand_counts

    def _sku_to_brand_name(self, sku_id):
        """Convert SKU ID to brand name"""
        sku_lower = sku_id.lower()
        
        if 'coca' in sku_lower or 'coke' in sku_lower:
            return 'COCA-COLA'
        elif 'pepsi' in sku_lower:
            return 'PEPSI'
        elif 'sprite' in sku_lower:
            return 'SPRITE'
        elif 'fanta' in sku_lower:
            return 'FANTA'
        elif 'seven' in sku_lower or '7up' in sku_lower:
            return 'SEVEN-UP'
        elif 'mountain' in sku_lower or 'dew' in sku_lower:
            return 'MOUNTAIN-DEW'
        elif 'red' in sku_lower and 'bull' in sku_lower:
            return 'RED-BULL'
        elif 'monster' in sku_lower:
            return 'MONSTER'
        elif 'tango' in sku_lower:
            return 'TANGO'
        else:
            return sku_id.upper()
