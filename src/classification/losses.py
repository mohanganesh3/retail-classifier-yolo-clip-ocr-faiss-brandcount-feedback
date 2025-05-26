import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, label):
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        
        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, label)

class MetaContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """Meta-contrastive loss for few-shot learning"""
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)
        
        loss = -mean_log_prob_pos.mean()
        return loss

class HierarchicalAuxLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, features, labels, hierarchy_labels=None):
        """Hierarchical auxiliary loss for brand/category classification"""
        if hierarchy_labels is None:
            return torch.tensor(0.0, device=features.device)
        
        # Simple auxiliary loss for hierarchical classification
        aux_loss = F.cross_entropy(features, hierarchy_labels)
        return self.weight * aux_loss
