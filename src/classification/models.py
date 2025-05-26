import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from transformers import DistilBertModel, DistilBertTokenizer

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes, visual_dim=576, text_dim=768, embedding_dim=1024):
        super().__init__()
        
        # Visual backbone
        self.visual_backbone = mobilenet_v3_small(pretrained=True)
        self.visual_backbone.classifier = nn.Identity()  # Remove final classifier
        
        # Text backbone
        self.text_backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Projection layers
        self.visual_projection = nn.Linear(visual_dim, embedding_dim)
        self.text_projection = nn.Linear(text_dim, embedding_dim)
        
        # Final classifier
        self.classifier = nn.Linear(embedding_dim * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, images, text_inputs=None):
        # Visual features
        visual_features = self.visual_backbone(images)
        visual_features = self.visual_projection(visual_features)
        
        if text_inputs is not None:
            # Text features
            text_outputs = self.text_backbone(**text_inputs)
            text_features = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_projection(text_features)
            
            # Combine features - ensure consistent dimensions
            combined_features = torch.cat([visual_features, text_features], dim=1)
        else:
            # Use only visual features - pad to match expected dimension
            combined_features = torch.cat([visual_features, torch.zeros_like(visual_features)], dim=1)
        
        combined_features = self.dropout(combined_features)
        output = self.classifier(combined_features)
        
        return output, combined_features
