import torch
import json
import numpy as np
from transformers import DistilBertTokenizer
from src.classification.models import MultiModalClassifier
from src.utils.image_utils import preprocess_image_classification
from src.utils.faiss_utils import load_faiss_index, search_faiss_index
from src.utils.config import load_config
import os

class ProductClassifier:
    def __init__(self, config_path='configs/classification.yaml'):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config['model']['text_backbone']
        )
        
        # Check if FAISS index exists
        index_path = self.config['output']['index_path']
        labels_path = self.config['output']['labels_path']
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        if os.path.exists(index_path) and os.path.exists(labels_path):
            try:
                # Load FAISS index and labels
                self.index = load_faiss_index(index_path)
                self.labels = torch.load(labels_path)
                self.index_available = True
                print(f"✅ Loaded FAISS index with {len(self.labels)} SKUs")
                
                # Validate index
                if len(self.labels) == 0:
                    print("⚠️ Warning: Empty labels list")
                    self.index_available = False
                    
            except Exception as e:
                print(f"❌ Error loading FAISS index: {e}")
                self.index = None
                self.labels = []
                self.index_available = False
        else:
            print("⚠️ FAISS index not found. Please build prototypes first.")
            print(f"Expected index: {index_path}")
            print(f"Expected labels: {labels_path}")
            self.index = None
            self.labels = []
            self.index_available = False
        
        # Initialize model
        num_classes = len(self.labels) if self.labels else 1000  # Default fallback
        self.model = MultiModalClassifier(num_classes).to(self.device)
        self.model.eval()
        
        print(f"Classifier initialized: index_available={self.index_available}")
    
    def classify_crops(self, crop_paths, ocr_texts=None):
        """Classify crops using combined image + OCR embeddings with FAISS search"""
        if not self.index_available:
            print("❌ No FAISS index available. Please build prototypes first.")
            return self._fallback_classification(crop_paths, ocr_texts)
        
        print(f"🔍 Classifying {len(crop_paths)} crops using combined embeddings...")
        
        # Initialize embedding models
        self._init_embedding_models()
        
        classifications = []
        
        for i, crop_path in enumerate(crop_paths):
            try:
                # Get OCR text for this crop
                ocr_text = ""
                if ocr_texts and i < len(ocr_texts):
                    ocr_text = ocr_texts[i] or ""
            
                # Extract combined embedding
                combined_embedding = self._extract_combined_embedding(crop_path, ocr_text)
            
                if combined_embedding is None:
                    print(f"⚠️ Could not extract embedding for crop {i+1}")
                    classifications.append(self._create_unknown_classification(crop_path, ocr_text))
                    continue
            
                # Search FAISS index
                distances, indices = search_faiss_index(self.index, combined_embedding.reshape(1, -1), k=3)
            
                if len(indices[0]) > 0:
                    # Get top prediction
                    top_idx = indices[0][0]
                    top_distance = distances[0][0]
                    predicted_sku = self.labels[top_idx]
                
                    # Convert distance to confidence (lower distance = higher confidence)
                    confidence = max(0.0, 1.0 / (1.0 + top_distance))
                
                    # Accept predictions with reasonable confidence
                    if confidence > 0.25:  # Changed from 0.4 to 0.25 for lower threshold
                        classifications.append({
                            'crop_path': crop_path,
                            'sku_id': str(predicted_sku),
                            'brand': str(predicted_sku),
                            'confidence': float(confidence),
                            'distance': float(top_distance),
                            'ocr_text': ocr_text,
                            'method': 'combined_embedding'
                        })
                        print(f"✅ Classified crop {i+1}: {predicted_sku} (conf: {confidence:.3f}) - '{ocr_text}'")
                    else:
                        print(f"⚠️ Low confidence for crop {i+1}: {confidence:.3f}")
                        classifications.append(self._create_unknown_classification(crop_path, ocr_text))
                else:
                    print(f"⚠️ No matches found for crop {i+1}")
                    classifications.append(self._create_unknown_classification(crop_path, ocr_text))
                
            except Exception as e:
                print(f"❌ Error classifying {crop_path}: {e}")
                classifications.append(self._create_unknown_classification(crop_path, ocr_text))
    
        print(f"✅ Combined embedding classification complete: {len(classifications)} crops processed")
        return classifications

    def _init_embedding_models(self):
        """Initialize embedding models for classification"""
        if not hasattr(self, 'clip_model'):
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("✅ CLIP model loaded for classification")
            except:
                self.clip_model = None
                self.clip_processor = None
    
        if not hasattr(self, 'text_encoder'):
            try:
                from sentence_transformers import SentenceTransformer
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ SentenceTransformer loaded for classification")
            except:
                self.text_encoder = None

    def _extract_combined_embedding(self, crop_path, ocr_text):
        """Extract combined image + text embedding for a crop"""
        try:
            # Extract image embedding
            image_embedding = self._extract_image_embedding_for_classification(crop_path)
            if image_embedding is None:
                return None
        
            # Extract text embedding
            text_embedding = self._extract_text_embedding_for_classification(ocr_text)
            if text_embedding is None:
                text_embedding = np.zeros(384)  # Default size
        
            # Combine embeddings (same method as in prototype building)
            image_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
            text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
            combined = np.concatenate([image_norm, text_norm])
        
            return combined
        
        except Exception as e:
            print(f"❌ Combined embedding extraction failed: {e}")
            return None

    def _extract_image_embedding_for_classification(self, image_path):
        """Extract image embedding for classification"""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
        
            if self.clip_model and self.clip_processor:
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    return image_features.cpu().numpy().flatten()
            else:
                # Fallback to existing method
                img_tensor = preprocess_image_classification(image_path).to(self.device)
                with torch.no_grad():
                    _, features = self.model(img_tensor)
                    return features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"❌ Image embedding failed: {e}")
            return None

    def _extract_text_embedding_for_classification(self, text):
        """Extract text embedding for classification"""
        if not text:
            return np.zeros(384)
    
        try:
            if self.text_encoder:
                return self.text_encoder.encode(text)
            else:
                # Fallback to DistilBERT
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
            
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    return embedding.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"❌ Text embedding failed: {e}")
            return np.zeros(384)

    def _create_unknown_classification(self, crop_path, ocr_text):
        """Create unknown classification entry"""
        return {
            'crop_path': crop_path,
            'sku_id': 'UNKNOWN',
            'brand': 'UNKNOWN',
            'confidence': 0.0,
            'ocr_text': ocr_text,
            'method': 'unknown'
        }

    def _fallback_classification(self, crop_paths, ocr_texts):
        """Fallback classification when no FAISS index available"""
        print("📊 Using fallback OCR-based classification...")
    
        brand_patterns = {
            'COCA_COLA': ['coca', 'coke', 'cola', 'cocacola'],
            'PEPSI': ['pepsi', 'peps'],
            'SPRITE': ['sprite', 'spri', 'spnt', 'spne', 'spnl'],
            'FANTA': ['fanta', 'fanva', 'orange'],
            'SEVEN_UP': ['7up', 'seven', 'up'],
            'TANGO': ['tango', 'tang'],
        }
    
        classifications = []
        for i, crop_path in enumerate(crop_paths):
            ocr_text = ocr_texts[i] if ocr_texts and i < len(ocr_texts) else ""
            brand = self._classify_by_ocr_patterns(ocr_text, brand_patterns)
            confidence = 0.8 if brand != 'UNKNOWN' else 0.1
        
            classifications.append({
                'crop_path': crop_path,
                'sku_id': brand,
                'brand': brand,
                'confidence': confidence,
                'ocr_text': ocr_text,
                'method': 'ocr_fallback'
            })
    
        return classifications

    def _classify_by_ocr_patterns(self, ocr_text, brand_patterns):
        """Classify using OCR pattern matching"""
        if not ocr_text:
            return 'UNKNOWN'
    
        ocr_lower = ocr_text.lower().strip()
        for brand, patterns in brand_patterns.items():
            for pattern in patterns:
                if pattern in ocr_lower:
                    return brand
        return 'UNKNOWN'

    def _classify_by_ocr(self, ocr_text, brand_patterns):
        """Classify brand based on OCR text patterns"""
        if not ocr_text:
            return 'UNKNOWN'
        
        ocr_lower = ocr_text.lower().strip()
        
        # Check each brand pattern
        for brand, patterns in brand_patterns.items():
            for pattern in patterns:
                if pattern in ocr_lower:
                    print(f"🎯 OCR Brand Match: '{ocr_text}' -> {brand}")
                    return brand
        
        return 'UNKNOWN'

    def classify_crops_with_feedback(self, crop_paths, ocr_texts=None, confidence_threshold=0.25):  # Changed from 0.6 to 0.25
        """Classify crops with user feedback for low-confidence detections"""
        print(f"🔍 Classifying {len(crop_paths)} crops with user feedback integration...")
        
        # Get initial classifications
        classifications = self.classify_crops(crop_paths, ocr_texts)
        
        # Separate high and low confidence classifications
        high_confidence = []
        low_confidence = []
        
        for i, classification in enumerate(classifications):
            confidence = classification.get('confidence', 0)
            
            if confidence >= confidence_threshold:
                high_confidence.append(classification)
            else:
                low_confidence.append({
                    'index': i,
                    'crop_path': classification.get('crop_path', ''),
                    'predicted_brand': classification.get('sku_id', 'UNKNOWN'),
                    'confidence': confidence,
                    'ocr_text': classification.get('ocr_text', ''),
                    'classification': classification
                })
        
        print(f"📊 Classification confidence: {len(high_confidence)} high, {len(low_confidence)} need feedback")
        
        # Store low confidence items for user feedback
        self.pending_feedback = low_confidence
        
        return classifications, low_confidence

    def get_existing_brands(self):
        """Get list of existing brand directories"""
        brands = ['UNKNOWN']
        classification_dir = 'data/classification'
        
        if os.path.exists(classification_dir):
            for item in os.listdir(classification_dir):
                item_path = os.path.join(classification_dir, item)
                if os.path.isdir(item_path):
                    brands.append(item)
        
        return sorted(brands)

    def process_user_feedback(self, feedback_data):
        """Process user feedback and update dataset"""
        print(f"🤔 Processing user feedback for {len(feedback_data)} items...")
        
        updated_classifications = []
        new_brands_created = []
        
        for feedback in feedback_data:
            crop_path = feedback.get('crop_path', '')
            selected_brand = feedback.get('selected_brand', '')
            new_brand_name = feedback.get('new_brand_name', '')
            original_classification = feedback.get('classification', {})
            
            if selected_brand == 'NEW_BRAND' and new_brand_name:
                # Create new brand directory and move image
                success, brand_dir = self._create_new_brand_directory(new_brand_name, crop_path)
                
                if success:
                    new_brands_created.append(new_brand_name)
                    
                    # Update classification
                    updated_classification = original_classification.copy()
                    updated_classification['sku_id'] = new_brand_name
                    updated_classification['confidence'] = 1.0  # User confirmed
                    updated_classification['method'] = 'user_feedback'
                    updated_classification['user_created'] = True
                    
                    updated_classifications.append(updated_classification)
                    print(f"✅ Created new brand: {new_brand_name}")
                else:
                    print(f"❌ Failed to create brand directory for: {new_brand_name}")
                    
            elif selected_brand and selected_brand != 'UNKNOWN':
                # Move to existing brand directory
                success = self._move_to_existing_brand(selected_brand, crop_path)
                
                if success:
                    # Update classification
                    updated_classification = original_classification.copy()
                    updated_classification['sku_id'] = selected_brand
                    updated_classification['confidence'] = 1.0  # User confirmed
                    updated_classification['method'] = 'user_feedback'
                    
                    updated_classifications.append(updated_classification)
                    print(f"✅ Moved to existing brand: {selected_brand}")
    
        # Update metadata if new brands were created
        if new_brands_created:
            self._update_metadata_with_new_brands(new_brands_created)
        
        return updated_classifications, new_brands_created

    def _create_new_brand_directory(self, brand_name, crop_path):
        """Create new brand directory and move crop image"""
        try:
            # Clean brand name for directory
            clean_name = brand_name.replace(' ', '_').replace('-', '_').upper()
            brand_dir = f"data/classification/{clean_name}"
            
            # Create directory
            os.makedirs(brand_dir, exist_ok=True)
            
            # Move/copy crop image
            if os.path.exists(crop_path):
                import shutil
                filename = os.path.basename(crop_path)
                new_path = os.path.join(brand_dir, filename)
                shutil.copy2(crop_path, new_path)
                
                print(f"📁 Created brand directory: {brand_dir}")
                print(f"📸 Moved image: {crop_path} -> {new_path}")
                
                return True, brand_dir
            else:
                print(f"❌ Crop image not found: {crop_path}")
                return False, None
            
        except Exception as e:
            print(f"❌ Error creating brand directory: {e}")
            return False, None

    def _move_to_existing_brand(self, brand_name, crop_path):
        """Move crop to existing brand directory"""
        try:
            brand_dir = f"data/classification/{brand_name}"
            
            if not os.path.exists(brand_dir):
                print(f"❌ Brand directory does not exist: {brand_dir}")
                return False
        
            if os.path.exists(crop_path):
                import shutil
                filename = os.path.basename(crop_path)
                new_path = os.path.join(brand_dir, filename)
                shutil.copy2(crop_path, new_path)
                
                print(f"📸 Moved to existing brand: {crop_path} -> {new_path}")
                return True
            else:
                print(f"❌ Crop image not found: {crop_path}")
                return False
            
        except Exception as e:
            print(f"❌ Error moving to existing brand: {e}")
            return False

    def _update_metadata_with_new_brands(self, new_brands):
        """Update metadata CSV with new brands"""
        try:
            import pandas as pd
        
            meta_path = 'data/meta.csv'
        
            # Load or create metadata
            if os.path.exists(meta_path):
                df = pd.read_csv(meta_path)
            else:
                df = pd.DataFrame(columns=['SKU_ID', 'name', 'brand', 'flavor', 'category'])
        
            # Add new brands
            for brand in new_brands:
                if not df[df['SKU_ID'] == brand].empty:
                    continue  # Brand already exists
            
                new_entry = {
                    'SKU_ID': brand,
                    'name': brand.replace('_', ' ').title(),
                    'brand': brand.replace('_', ' ').title(),
                    'flavor': 'Original',
                    'category': 'Beverages'
                }
            
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        
            # Save updated metadata
            os.makedirs('data', exist_ok=True)
            df.to_csv(meta_path, index=False)
        
            print(f"✅ Updated metadata with {len(new_brands)} new brands")
        
        except Exception as e:
            print(f"❌ Error updating metadata: {e}")
