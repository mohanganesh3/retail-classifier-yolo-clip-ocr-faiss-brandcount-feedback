import torch
import os
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer
from src.classification.models import MultiModalClassifier
from src.utils.image_utils import preprocess_image_classification
from src.utils.faiss_utils import build_faiss_index, save_faiss_index
from src.utils.config import load_config
from src.ocr.ocr_processor import OCRProcessor

class PrototypeBuilder:
    def __init__(self, config_path='configs/classification.yaml'):
        self.config = load_config(config_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config['model']['text_backbone']
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize OCR processor for text extraction
        self.ocr_processor = OCRProcessor()
        
        print(f"🔧 PrototypeBuilder initialized on {self.device}")
    
    
    def build_prototypes(self, prototype_dir, meta_csv):
        """Build prototype embeddings with combined image + OCR features"""
        # Load metadata
        meta_df = pd.read_csv(meta_csv)
        
        # Initialize models for embedding extraction
        from transformers import CLIPProcessor, CLIPModel
        import torch.nn.functional as F
        
        # Load CLIP for better image embeddings
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ CLIP model loaded for image embeddings")
        except:
            print("⚠️ CLIP not available, using MobileNet")
            self.clip_model = None
            self.clip_processor = None
        
        # Initialize text model for OCR embeddings
        from sentence_transformers import SentenceTransformer
        try:
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded for text embeddings")
        except:
            print("⚠️ Using DistilBERT for text embeddings")
            self.text_encoder = None
        
        prototypes = []
        labels = []
        
        print(f"🏗️ Building combined embeddings for {len(meta_df)} SKUs...")
        
        for idx, row in meta_df.iterrows():
            sku_id = row['SKU_ID']
            sku_path = os.path.join(prototype_dir, str(sku_id))
            
            if not os.path.exists(sku_path):
                print(f"⚠️ Warning: No prototype images found for SKU {sku_id}")
                continue
        
            # Process all images for this SKU
            image_embeddings = []
            text_embeddings = []
            image_files = [f for f in os.listdir(sku_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"📸 Processing {len(image_files)} images for {sku_id}...")
            
            for img_file in image_files[:10]:  # Max 10 images per SKU
                img_path = os.path.join(sku_path, img_file)
                try:
                    # Extract image embedding
                    image_embedding = self._extract_image_embedding(img_path)
                    if image_embedding is not None:
                        image_embeddings.append(image_embedding)
                
                    # Extract OCR text and create text embedding
                    ocr_text = self.ocr_processor.extract_text(img_path)
                    if ocr_text:
                        text_embedding = self._extract_text_embedding(ocr_text)
                        if text_embedding is not None:
                            text_embeddings.append(text_embedding)
                        print(f"📝 OCR for {img_file}: '{ocr_text}'")
                    else:
                        # Use metadata as fallback text
                        metadata_text = f"{row.get('name', '')} {row.get('brand', '')} {row.get('flavor', '')}"
                        text_embedding = self._extract_text_embedding(metadata_text)
                        if text_embedding is not None:
                            text_embeddings.append(text_embedding)
                
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")
                    continue
        
            if not image_embeddings:
                print(f"⚠️ Warning: No valid images for SKU {sku_id}")
                continue
        
            # Average embeddings
            avg_image_embedding = np.mean(image_embeddings, axis=0)
        
            if text_embeddings:
                avg_text_embedding = np.mean(text_embeddings, axis=0)
            else:
                # Fallback: create text embedding from metadata
                metadata_text = f"{row.get('name', '')} {row.get('brand', '')} {row.get('flavor', '')}"
                avg_text_embedding = self._extract_text_embedding(metadata_text)
                if avg_text_embedding is None:
                    avg_text_embedding = np.zeros(384)  # Default size
        
            # Combine image and text embeddings
            combined_embedding = self._combine_embeddings(avg_image_embedding, avg_text_embedding)
        
            prototypes.append(combined_embedding)
            labels.append(sku_id)
        
            print(f"✅ Built combined embedding for {sku_id} (image + text)")
    
        if not prototypes:
            raise ValueError("No valid prototypes were created!")
    
        # Build FAISS index
        prototypes_array = np.array(prototypes)
        print(f"📊 Combined embedding array shape: {prototypes_array.shape}")
    
        index = build_faiss_index(
            prototypes_array,
            dimension=prototypes_array.shape[1],
            index_type=self.config['faiss']['index_type']
        )
    
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config['output']['index_path']), exist_ok=True)
    
        # Save index and labels
        save_faiss_index(index, self.config['output']['index_path'])
        torch.save(labels, self.config['output']['labels_path'])
    
        print(f"✅ Built combined embeddings for {len(labels)} SKUs")
        print(f"💾 FAISS index saved to: {self.config['output']['index_path']}")
        print(f"💾 Labels saved to: {self.config['output']['labels_path']}")
    
        return index, labels

    def _extract_image_embedding(self, image_path):
        """Extract image embedding using CLIP or MobileNet"""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
        
            if self.clip_model and self.clip_processor:
                # Use CLIP for better embeddings
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    return image_features.cpu().numpy().flatten()
            else:
                # Fallback to MobileNet
                img_tensor = preprocess_image_classification(image_path).to(self.device)
                with torch.no_grad():
                    model = MultiModalClassifier(1000).to(self.device)
                    model.eval()
                    _, features = model(img_tensor)
                    return features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"❌ Image embedding failed for {image_path}: {e}")
            return None

    def _extract_text_embedding(self, text):
        """Extract text embedding using SentenceTransformer or DistilBERT"""
        if not text:
            return np.zeros(384)  # Default embedding size
    
        try:
            if self.text_encoder:
                # Use SentenceTransformer for better text embeddings
                embedding = self.text_encoder.encode(text)
                return embedding
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
            print(f"❌ Text embedding failed for '{text}': {e}")
            return np.zeros(384)

    def _combine_embeddings(self, image_embedding, text_embedding):
        """Combine image and text embeddings"""
        # Normalize embeddings
        image_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
        text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
    
        # Concatenate normalized embeddings
        combined = np.concatenate([image_norm, text_norm])
    
        return combined
