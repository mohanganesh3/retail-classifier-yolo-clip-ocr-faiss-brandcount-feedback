import faiss
import numpy as np
import torch
import pickle

def build_faiss_index(embeddings, dimension=512, index_type='flat_l2'):
    """Build FAISS index from embeddings"""
    if index_type == 'flat_l2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'hnsw':
        index = faiss.IndexHNSWFlat(dimension, 32)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    index.add(embeddings.astype(np.float32))
    return index

def load_faiss_index(index_path):
    """Load FAISS index from file"""
    return faiss.read_index(index_path)

def save_faiss_index(index, index_path):
    """Save FAISS index to file"""
    faiss.write_index(index, index_path)

def search_faiss_index(index, query_embedding, k=3):
    """Search FAISS index for similar embeddings"""
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return distances, indices
