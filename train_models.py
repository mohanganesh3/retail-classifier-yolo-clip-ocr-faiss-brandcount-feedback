#!/usr/bin/env python3
"""
Training script for the retail shelf monitoring system
"""

import argparse
import os
from src.detection.detector import ProductDetector
from src.classification.prototype_builder import PrototypeBuilder
from src.utils.config import ensure_directories

def train_detector(data_yaml_path):
    """Train YOLOv8 detector"""
    print("Training YOLOv8 detector...")
    detector = ProductDetector()
    detector.train(data_yaml_path)
    print("Detector training complete!")

def build_prototypes(prototype_dir, meta_csv):
    """Build classification prototypes"""
    print("Building classification prototypes...")
    builder = PrototypeBuilder()
    index, labels = builder.build_prototypes(prototype_dir, meta_csv)
    print(f"Built prototypes for {len(labels)} SKUs")

def main():
    parser = argparse.ArgumentParser(description="Train retail monitoring models")
    parser.add_argument('--mode', choices=['detector', 'prototypes', 'all'], 
                       default='all', help='Training mode')
    parser.add_argument('--data_yaml', default='data/dataset.yaml',
                       help='YOLO dataset YAML file')
    parser.add_argument('--prototype_dir', default='data/prototypes',
                       help='Prototype images directory')
    parser.add_argument('--meta_csv', default='data/meta.csv',
                       help='Metadata CSV file')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    if args.mode in ['detector', 'all']:
        if os.path.exists(args.data_yaml):
            train_detector(args.data_yaml)
        else:
            print(f"Warning: Dataset YAML not found at {args.data_yaml}")
    
    if args.mode in ['prototypes', 'all']:
        if os.path.exists(args.prototype_dir) and os.path.exists(args.meta_csv):
            build_prototypes(args.prototype_dir, args.meta_csv)
        else:
            print(f"Warning: Prototype data not found")
            print(f"Prototype dir: {args.prototype_dir}")
            print(f"Meta CSV: {args.meta_csv}")

if __name__ == "__main__":
    main()
