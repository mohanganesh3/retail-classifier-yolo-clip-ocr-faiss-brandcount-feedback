#!/usr/bin/env python3
"""
Inference script for processing images through the retail monitoring pipeline
"""

import argparse
import json
from src.pipeline.retail_pipeline import RetailShelfPipeline

def main():
    parser = argparse.ArgumentParser(description="Process retail shelf images")
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', default='results.json', help='Output JSON file')
    parser.add_argument('--detector_model', default='models/detector/yolov8n.pt',
                       help='Detector model path')
    parser.add_argument('--no_ocr', action='store_true', help='Disable OCR')
    parser.add_argument('--batch', nargs='+', help='Process multiple images')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RetailShelfPipeline(
        detector_model=args.detector_model,
        use_ocr=not args.no_ocr
    )
    
    # Process images
    if args.batch:
        print(f"Processing {len(args.batch)} images...")
        results = pipeline.process_batch(args.batch)
    else:
        print(f"Processing image: {args.image}")
        results = pipeline.process_image(args.image)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
