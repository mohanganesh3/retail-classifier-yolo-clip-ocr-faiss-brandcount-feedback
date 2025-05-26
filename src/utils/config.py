import yaml
import os

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw_images',
        'data/annotations', 
        'data/prototypes',
        'models/detector',
        'models/classifier',
        'models/faiss_index',
        'temp/crops',
        'runtime'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
