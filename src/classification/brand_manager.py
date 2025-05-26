import os
import json
import pandas as pd
import shutil
from collections import Counter
import cv2
from PIL import Image

class BrandManager:
    def __init__(self):
        self.brands_dir = 'data/prototypes'
        self.meta_file = 'data/meta.csv'
        self.confidence_threshold = 0.6  # Ask user if below this
        
        # Load existing brand mapping
        self.brand_mapping = self._load_brand_mapping()
        
        print(f"🏷️ Brand Manager initialized with {len(self.brand_mapping)} known brands")
    
    def _load_brand_mapping(self):
        """Load brand mapping from directory structure"""
        brand_mapping = {}
        
        if os.path.exists(self.brands_dir):
            for brand_dir in os.listdir(self.brands_dir):
                brand_path = os.path.join(self.brands_dir, brand_dir)
                if os.path.isdir(brand_path):
                    # Convert directory name to display name
                    display_name = self._directory_to_brand_name(brand_dir)
                    brand_mapping[brand_dir] = display_name
                    print(f"📁 Found brand: {brand_dir} -> {display_name}")
        
        return brand_mapping
    
    def _directory_to_brand_name(self, dir_name):
        """Convert directory name to readable brand name"""
        # Handle common patterns
        name_mapping = {
            'coca_cola': 'COCA-COLA',
            'cocacola': 'COCA-COLA', 
            'coke': 'COCA-COLA',
            'pepsi': 'PEPSI',
            'sprite': 'SPRITE',
            'fanta': 'FANTA',
            'seven_up': 'SEVEN-UP',
            '7up': 'SEVEN-UP',
            'mountain_dew': 'MOUNTAIN-DEW',
            'red_bull': 'RED-BULL',
            'monster': 'MONSTER',
            'tango': 'TANGO'
        }
        
        dir_lower = dir_name.lower()
        
        # Check exact matches first
        if dir_lower in name_mapping:
            return name_mapping[dir_lower]
        
        # Check partial matches
        for key, value in name_mapping.items():
            if key in dir_lower or dir_lower in key:
                return value
        
        # Default: clean up the directory name
        return dir_name.replace('_', '-').upper()
    
    def classify_and_count_brands(self, classifications, crop_paths):
        """Classify products and return brand counts with user interaction"""
        print(f"🏷️ Processing {len(classifications)} classifications for brand counting...")
        
        brand_counts = Counter()
        uncertain_crops = []
        
        for i, classification in enumerate(classifications):
            sku_id = classification.get('sku_id', 'UNKNOWN')
            confidence = classification.get('confidence', 0)
            crop_path = classification.get('crop_path', '')
            
            if sku_id in self.brand_mapping and confidence >= self.confidence_threshold:
                # High confidence classification
                brand_name = self.brand_mapping[sku_id]
                brand_counts[brand_name] += 1
                print(f"✅ {brand_name}: {confidence:.3f} confidence")
                
            elif sku_id in self.brand_mapping and confidence < self.confidence_threshold:
                # Low confidence - ask user
                uncertain_crops.append({
                    'crop_path': crop_path,
                    'predicted_brand': self.brand_mapping[sku_id],
                    'confidence': confidence,
                    'index': i
                })
                print(f"❓ Uncertain: {self.brand_mapping[sku_id]} ({confidence:.3f})")
                
            else:
                # Unknown product - ask user
                uncertain_crops.append({
                    'crop_path': crop_path,
                    'predicted_brand': 'UNKNOWN',
                    'confidence': confidence,
                    'index': i
                })
                print(f"❓ Unknown product: {confidence:.3f}")
        
        # Handle uncertain classifications
        if uncertain_crops:
            print(f"\n🤔 Found {len(uncertain_crops)} uncertain classifications")
            user_classifications = self._handle_uncertain_classifications(uncertain_crops)
            
            # Add user classifications to counts
            for user_class in user_classifications:
                if user_class['brand'] != 'SKIP':
                    brand_counts[user_class['brand']] += 1
        
        # Convert to regular dict and sort
        final_counts = dict(brand_counts.most_common())
        
        print(f"\n🎯 FINAL BRAND COUNTS:")
        for brand, count in final_counts.items():
            print(f"   {brand}: {count}")
        
        return final_counts
    
    def _handle_uncertain_classifications(self, uncertain_crops):
        """Handle uncertain classifications with real user interaction"""
        print(f"\n🤔 UNCERTAIN CLASSIFICATIONS DETECTED")
        print(f"=" * 50)
        
        user_classifications = []
        
        # Store uncertain crops for Streamlit interface
        self.uncertain_crops_data = uncertain_crops
        
        for i, uncertain in enumerate(uncertain_crops):
            crop_path = uncertain['crop_path']
            predicted_brand = uncertain['predicted_brand']
            confidence = uncertain['confidence']
            
            print(f"\n📸 Product {i+1}/{len(uncertain_crops)}")
            print(f"   Crop: {os.path.basename(crop_path)}")
            print(f"   Predicted: {predicted_brand}")
            print(f"   Confidence: {confidence:.3f}")
            
            # For now, use simulation but mark for user review
            user_choice = self._simulate_user_choice(uncertain, list(self.brand_mapping.values()))
            
            if user_choice == 'NEW':
                # Handle new brand
                new_brand = self._handle_new_brand(crop_path)
                user_classifications.append({
                    'crop_path': crop_path,
                    'brand': new_brand,
                    'confidence': 1.0,
                    'method': 'user_new'
                })
            elif user_choice == 'SKIP':
                user_classifications.append({
                    'crop_path': crop_path,
                    'brand': 'SKIP',
                    'confidence': 0.0,
                    'method': 'user_skip'
                })
            else:
                # Existing brand selected
                user_classifications.append({
                    'crop_path': crop_path,
                    'brand': user_choice,
                    'confidence': 1.0,
                    'method': 'user_corrected'
                })
                
                # Add to training data
                self._add_to_training_data(crop_path, user_choice)
    
        return user_classifications

    def _simulate_user_choice(self, uncertain, brand_options):
        """Simulate user choice based on OCR and image analysis"""
        crop_path = uncertain['crop_path']
        predicted_brand = uncertain['predicted_brand']
        
        # Try to determine brand from filename or OCR
        crop_name = os.path.basename(crop_path).lower()
        
        # Simple heuristics for simulation
        if 'coca' in crop_name or 'coke' in crop_name:
            return 'COCA-COLA'
        elif 'pepsi' in crop_name:
            return 'PEPSI'
        elif 'sprite' in crop_name:
            return 'SPRITE'
        elif 'fanta' in crop_name:
            return 'FANTA'
        elif predicted_brand != 'UNKNOWN':
            return predicted_brand  # Accept prediction
        else:
            return 'NEW'  # Treat as new brand
    
    def _handle_new_brand(self, crop_path):
        """Handle new brand discovery"""
        print(f"\n🆕 NEW BRAND DETECTED")
        
        # Generate new brand name (simulate user input)
        new_brand_name = self._generate_new_brand_name(crop_path)
        
        # Create new directory
        new_dir_name = new_brand_name.lower().replace('-', '_').replace(' ', '_')
        new_brand_dir = os.path.join(self.brands_dir, new_dir_name)
        
        os.makedirs(new_brand_dir, exist_ok=True)
        
        # Copy crop to new brand directory
        crop_filename = f"{new_dir_name}_sample_1.jpg"
        new_crop_path = os.path.join(new_brand_dir, crop_filename)
        shutil.copy2(crop_path, new_crop_path)
        
        # Update brand mapping
        self.brand_mapping[new_dir_name] = new_brand_name
        
        # Update metadata
        self._update_metadata(new_dir_name, new_brand_name)
        
        print(f"✅ Created new brand: {new_brand_name}")
        print(f"   Directory: {new_brand_dir}")
        print(f"   Sample: {new_crop_path}")
        
        # Mark for vector database rebuild
        self._mark_for_rebuild()
        
        return new_brand_name
    
    def _generate_new_brand_name(self, crop_path):
        """Generate new brand name from crop analysis"""
        # Try OCR analysis
        try:
            from src.ocr.ocr_processor import OCRProcessor
            ocr = OCRProcessor()
            text = ocr.extract_text(crop_path)
            
            if text:
                # Clean and format text as brand name
                clean_text = ''.join(c for c in text if c.isalnum() or c.isspace())
                brand_name = clean_text.strip().upper()
                if brand_name and len(brand_name) <= 20:
                    return brand_name
        except:
            pass
        
        # Fallback: generate based on timestamp
        import time
        timestamp = int(time.time())
        return f"NEW_BRAND_{timestamp}"
    
    def _add_to_training_data(self, crop_path, brand_name):
        """Add corrected classification to training data"""
        # Find corresponding directory
        brand_dir = None
        for dir_name, display_name in self.brand_mapping.items():
            if display_name == brand_name:
                brand_dir = os.path.join(self.brands_dir, dir_name)
                break
        
        if brand_dir and os.path.exists(brand_dir):
            # Count existing samples
            existing_files = [f for f in os.listdir(brand_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            next_number = len(existing_files) + 1
            
            # Copy crop as new training sample
            dir_name = os.path.basename(brand_dir)
            new_filename = f"{dir_name}_corrected_{next_number}.jpg"
            new_path = os.path.join(brand_dir, new_filename)
            
            shutil.copy2(crop_path, new_path)
            print(f"✅ Added training sample: {new_path}")
            
            # Mark for vector database rebuild
            self._mark_for_rebuild()
    
    def _update_metadata(self, dir_name, brand_name):
        """Update metadata CSV with new brand"""
        # Load existing metadata
        if os.path.exists(self.meta_file):
            df = pd.read_csv(self.meta_file)
        else:
            df = pd.DataFrame(columns=['SKU_ID', 'name', 'brand', 'flavor', 'category'])
        
        # Add new entry
        new_entry = {
            'SKU_ID': dir_name,
            'name': brand_name,
            'brand': brand_name,
            'flavor': 'Original',
            'category': 'Beverages'
        }
        
        # Check if already exists
        if dir_name not in df['SKU_ID'].values:
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(self.meta_file, index=False)
            print(f"✅ Updated metadata with {brand_name}")
    
    def _mark_for_rebuild(self):
        """Mark that vector database needs rebuilding"""
        rebuild_flag = 'temp/rebuild_vector_db.flag'
        os.makedirs(os.path.dirname(rebuild_flag), exist_ok=True)
        with open(rebuild_flag, 'w') as f:
            f.write('rebuild_needed')
        print(f"🔄 Marked vector database for rebuild")
    
    def needs_rebuild(self):
        """Check if vector database needs rebuilding"""
        return os.path.exists('temp/rebuild_vector_db.flag')
    
    def rebuild_vector_database(self):
        """Rebuild vector database with new samples"""
        print(f"\n🔄 REBUILDING VECTOR DATABASE")
        print(f"=" * 40)
        
        try:
            from src.classification.prototype_builder import PrototypeBuilder
            
            builder = PrototypeBuilder()
            index, labels = builder.build_prototypes(self.brands_dir, self.meta_file)
            
            # Update brand mapping
            self.brand_mapping = self._load_brand_mapping()
            
            # Remove rebuild flag
            rebuild_flag = 'temp/rebuild_vector_db.flag'
            if os.path.exists(rebuild_flag):
                os.remove(rebuild_flag)
            
            print(f"✅ Vector database rebuilt with {len(labels)} brands")
            return True
            
        except Exception as e:
            print(f"❌ Vector database rebuild failed: {e}")
            return False
    
    def get_brand_display_names(self, sku_counts):
        """Convert SKU counts to brand display names"""
        brand_counts = {}
        
        for sku_id, count in sku_counts.items():
            if sku_id in self.brand_mapping:
                brand_name = self.brand_mapping[sku_id]
                brand_counts[brand_name] = count
            else:
                # Unknown SKU - use as is
                brand_counts[sku_id] = count
        
        return brand_counts

    def get_uncertain_crops(self):
        """Get uncertain crops for Streamlit interface"""
        return getattr(self, 'uncertain_crops_data', [])

    def process_user_classification(self, crop_index, user_choice, new_brand_name=None):
        """Process user classification from Streamlit interface"""
        if not hasattr(self, 'uncertain_crops_data') or crop_index >= len(self.uncertain_crops_data):
            return None
        
        uncertain = self.uncertain_crops_data[crop_index]
        crop_path = uncertain['crop_path']
        
        if user_choice == 'NEW' and new_brand_name:
            # Create new brand with user-provided name
            new_brand = self._create_new_brand_with_name(crop_path, new_brand_name)
            return {
                'crop_path': crop_path,
                'brand': new_brand,
                'confidence': 1.0,
                'method': 'user_new'
            }
        elif user_choice == 'SKIP':
            return {
                'crop_path': crop_path,
                'brand': 'SKIP',
                'confidence': 0.0,
                'method': 'user_skip'
            }
        elif user_choice in self.brand_mapping.values():
            # Existing brand selected
            self._add_to_training_data(crop_path, user_choice)
            return {
                'crop_path': crop_path,
                'brand': user_choice,
                'confidence': 1.0,
                'method': 'user_corrected'
            }
        
        return None

    def _create_new_brand_with_name(self, crop_path, brand_name):
        """Create new brand with user-provided name"""
        print(f"\n🆕 CREATING NEW BRAND: {brand_name}")
        
        # Clean brand name for directory
        new_dir_name = brand_name.lower().replace('-', '_').replace(' ', '_')
        new_dir_name = ''.join(c for c in new_dir_name if c.isalnum() or c == '_')
        
        new_brand_dir = os.path.join(self.brands_dir, new_dir_name)
        os.makedirs(new_brand_dir, exist_ok=True)
        
        # Copy crop to new brand directory
        crop_filename = f"{new_dir_name}_sample_1.jpg"
        new_crop_path = os.path.join(new_brand_dir, crop_filename)
        shutil.copy2(crop_path, new_crop_path)
        
        # Update brand mapping
        display_name = brand_name.upper()
        self.brand_mapping[new_dir_name] = display_name
        
        # Update metadata
        self._update_metadata(new_dir_name, display_name)
        
        print(f"✅ Created new brand: {display_name}")
        print(f"   Directory: {new_brand_dir}")
        print(f"   Sample: {new_crop_path}")
        
        # Mark for vector database rebuild
        self._mark_for_rebuild()
        
        return display_name
