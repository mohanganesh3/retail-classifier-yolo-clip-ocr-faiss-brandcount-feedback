import easyocr
import torch
from transformers import DistilBertModel, DistilBertTokenizer
import cv2
import numpy as np

class OCRProcessor:
    def __init__(self, languages=['en']):
        print("🔧 Initializing OCR processor...")
        try:
            self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
            print("✅ EasyOCR initialized successfully")
        except Exception as e:
            print(f"❌ EasyOCR initialization failed: {e}")
            self.reader = None
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.text_model.to(self.device)
            self.text_model.eval()
            print("✅ Text embedding model initialized")
        except Exception as e:
            print(f"❌ Text model initialization failed: {e}")
            self.tokenizer = None
            self.text_model = None
    
    def extract_text(self, image_path):
        """Enhanced text extraction for can labels"""
        if not self.reader:
            return ""
    
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return ""
        
            # Try multiple preprocessing approaches
            preprocessed_images = []
        
            # Original image
            preprocessed_images.append(img)
        
            # Enhanced preprocessing
            enhanced = self._preprocess_for_ocr(img)
            preprocessed_images.append(enhanced)
        
            # Try different rotations for can labels
            for angle in [0, 90, 180, 270]:
                if angle > 0:
                    h, w = img.shape[:2]
                    center = (w//2, h//2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(img, matrix, (w, h))
                    preprocessed_images.append(rotated)
        
            all_texts = []
        
            # Try OCR on all preprocessed versions
            for i, proc_img in enumerate(preprocessed_images):
                try:
                    results = self.reader.readtext(proc_img, detail=1)
                
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3 and len(text.strip()) > 1:  # Filter short/low confidence text
                            clean_text = text.strip()
                            if clean_text and clean_text not in all_texts:
                                all_texts.append(clean_text)
                                print(f"📝 OCR found (v{i}): '{clean_text}' (conf: {confidence:.3f})")
                            
                except Exception as e:
                    print(f"OCR failed on version {i}: {e}")
                    continue
        
            # Combine all found texts
            if all_texts:
                combined_text = ' '.join(all_texts)
                print(f"📝 Combined OCR result: '{combined_text}' from {image_path}")
                return combined_text
        
            return ""
        
        except Exception as e:
            print(f"OCR error for {image_path}: {e}")
            return ""
    
    def _preprocess_for_ocr(self, img):
        """Enhanced OCR preprocessing for can labels"""
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
        
            # Resize if too small
            h, w = gray.shape
            if h < 100 or w < 100:
                scale = max(100/h, 100/w)
                new_h, new_w = int(h * scale), int(w * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
        
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
        
            # Sharpen for text clarity
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
        
            # Multiple threshold approaches
            # 1. Otsu threshold
            _, thresh1 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
            # 2. Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
            # 3. Inverted threshold for dark text on light background
            _, thresh3 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
            # Return the best threshold (you could try all three)
            return thresh1
        
        except Exception as e:
            print(f"OCR preprocessing failed: {e}")
            return img
    
    def extract_text_embedding(self, text):
        """Extract text embedding using DistilBERT"""
        if not text or not self.text_model or not self.tokenizer:
            return None
        
        try:
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
                return embedding.cpu().numpy()
                
        except Exception as e:
            print(f"Text embedding error: {e}")
            return None
    
    def process_crops(self, crop_paths):
        """Process multiple crops and extract text with detailed logging"""
        ocr_results = []
        
        print(f"🔍 Processing OCR for {len(crop_paths)} crops...")
        
        successful_extractions = 0
        
        for i, crop_path in enumerate(crop_paths):
            text = self.extract_text(crop_path)
            ocr_results.append(text)
            
            if text:
                successful_extractions += 1
                print(f"✅ OCR {i+1}/{len(crop_paths)}: Found text in {crop_path}")
            else:
                print(f"⚪ OCR {i+1}/{len(crop_paths)}: No text in {crop_path}")
        
        print(f"📊 OCR Summary: {successful_extractions}/{len(crop_paths)} crops contained readable text")
        return ocr_results
    
    def extract_text_features(self, crop_paths):
        """Extract both text and text embeddings from crops"""
        results = []
        
        for crop_path in crop_paths:
            text = self.extract_text(crop_path)
            embedding = self.extract_text_embedding(text) if text else None
            
            results.append({
                'crop_path': crop_path,
                'text': text,
                'embedding': embedding,
                'has_text': bool(text)
            })
        
        return results
