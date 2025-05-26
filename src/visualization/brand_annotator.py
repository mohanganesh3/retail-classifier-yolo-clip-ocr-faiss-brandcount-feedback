import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class BrandAnnotator:
    def __init__(self, brand_manager):
        self.brand_manager = brand_manager
        
        # Brand colors for visualization
        self.brand_colors = {
            'COCA-COLA': (220, 20, 60),    # Crimson Red
            'PEPSI': (0, 100, 200),        # Pepsi Blue
            'SPRITE': (50, 205, 50),       # Lime Green
            'FANTA': (255, 140, 0),        # Dark Orange
            'SEVEN-UP': (124, 252, 0),     # Lawn Green
            'MOUNTAIN-DEW': (255, 215, 0), # Gold
            'RED-BULL': (255, 69, 0),      # Red Orange
            'MONSTER': (0, 255, 0),        # Bright Green
            'TANGO': (255, 165, 0),        # Orange
            'UNKNOWN': (128, 128, 128),    # Gray
        }
        
        print("🎨 Brand Annotator initialized with brand-specific colors")
    
    def create_branded_annotation(self, image_path, detection_data, classifications, brand_counts):
        """Create annotated image with brand labels and counts"""
        print(f"🎨 Creating branded annotation for {len(detection_data)} detections")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            label_font = ImageFont.truetype("arial.ttf", 18)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw brand counts at the top
        self._draw_brand_summary(draw, brand_counts, title_font, pil_img.width)
        
        # Annotate each detection with brand label
        for i, detection in enumerate(detection_data):
            bbox = detection.get('bbox', [])
            
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # Get brand for this detection
                brand_name = 'UNKNOWN'
                confidence = 0.0
                
                if i < len(classifications):
                    classification = classifications[i]
                    sku_id = classification.get('sku_id', 'UNKNOWN')
                    confidence = classification.get('confidence', 0)
                    
                    if sku_id in self.brand_manager.brand_mapping:
                        brand_name = self.brand_manager.brand_mapping[sku_id]
                
                # Get brand color
                color = self.brand_colors.get(brand_name, self.brand_colors['UNKNOWN'])
                
                # Draw bounding box with brand color
                line_width = 3 if confidence > 0.6 else 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                
                # Draw brand label prominently on rectangle
                label_text = f"{brand_name}"
                if confidence > 0:
                    if confidence < 0.6:
                        label_text += f" (?{confidence:.2f})"  # Question mark for low confidence
                    else:
                        label_text += f" ({confidence:.2f})"

                # Make label more prominent - draw directly on rectangle
                label_font_size = max(14, min(24, int((x2-x1)/8)))  # Scale font with box size
                try:
                    label_font_large = ImageFont.truetype("arial.ttf", label_font_size)
                except:
                    label_font_large = label_font

                # Calculate text size for centering
                try:
                    bbox_text = draw.textbbox((0, 0), label_text, font=label_font_large)
                    label_width = bbox_text[2] - bbox_text[0]
                    label_height = bbox_text[3] - bbox_text[1]
                except:
                    label_width, label_height = len(label_text) * 8, 20

                # Position label in center of rectangle
                label_x = x1 + (x2 - x1 - label_width) // 2
                label_y = y1 + (y2 - y1 - label_height) // 2

                # Draw semi-transparent background for text
                padding = 5
                draw.rectangle([label_x - padding, label_y - padding, 
                               label_x + label_width + padding, label_y + label_height + padding], 
                             fill=(*color, 180))

                # Draw label text in white
                draw.text((label_x, label_y), label_text, fill=(255, 255, 255), font=label_font_large)

                # Add warning indicator for low confidence
                if confidence < 0.6:
                    # Draw warning triangle
                    warning_size = 15
                    warning_x = x1 + 5
                    warning_y = y1 + 5
                    triangle_points = [
                        (warning_x, warning_y + warning_size),
                        (warning_x + warning_size//2, warning_y),
                        (warning_x + warning_size, warning_y + warning_size)
                    ]
                    draw.polygon(triangle_points, fill=(255, 255, 0), outline=(255, 0, 0), width=2)
                    draw.text((warning_x + 3, warning_y + 3), "!", fill=(255, 0, 0), font=small_font)
                
                # Draw detection number
                number_text = str(i + 1)
                circle_size = 20
                circle_x = x2 - circle_size - 5
                circle_y = y1 + 5
                
                if circle_x > 0 and circle_y > 0:
                    draw.ellipse([circle_x, circle_y, circle_x + circle_size, circle_y + circle_size], 
                               fill=color, outline=(255, 255, 255), width=2)
                    
                    # Center number in circle
                    try:
                        bbox_num = draw.textbbox((0, 0), number_text, font=small_font)
                        num_width = bbox_num[2] - bbox_num[0]
                        num_height = bbox_num[3] - bbox_num[1]
                        
                        draw.text((circle_x + (circle_size - num_width) // 2, 
                                 circle_y + (circle_size - num_height) // 2), 
                                number_text, fill=(255, 255, 255), font=small_font)
                    except:
                        pass
        
        print(f"✅ Created branded annotation with {len(detection_data)} labeled products")
        return pil_img
    
    def _draw_brand_summary(self, draw, brand_counts, font, img_width):
        """Draw brand count summary at the top of the image"""
        if not brand_counts:
            return
        
        # Create summary text
        summary_lines = []
        summary_lines.append("BRAND COUNTS:")
        
        for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"{brand}: {count}")
        
        # Calculate summary box size
        max_width = 0
        total_height = 0
        line_heights = []
        
        for line in summary_lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
            except:
                line_width = len(line) * 12
                line_height = 20
            
            max_width = max(max_width, line_width)
            line_heights.append(line_height)
            total_height += line_height + 5
        
        # Draw summary background
        summary_x = 10
        summary_y = 10
        summary_width = max_width + 20
        summary_height = total_height + 10
        
        draw.rectangle([summary_x, summary_y, summary_x + summary_width, summary_y + summary_height], 
                     fill=(0, 0, 0, 180))
        
        # Draw summary text
        current_y = summary_y + 10
        for i, line in enumerate(summary_lines):
            if i == 0:
                # Title line in white
                draw.text((summary_x + 10, current_y), line, fill=(255, 255, 255), font=font)
            else:
                # Brand lines in brand colors
                brand_name = line.split(':')[0]
                color = self.brand_colors.get(brand_name, (255, 255, 255))
                draw.text((summary_x + 10, current_y), line, fill=color, font=font)
            
            current_y += line_heights[i] + 5
    
    def save_annotated_image(self, annotated_img, output_path):
        """Save annotated image"""
        try:
            annotated_img.save(output_path, 'PNG', quality=95)
            print(f"💾 Saved branded annotation: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save annotation: {e}")
            return False
