import streamlit as st
import os
import json
from PIL import Image
import pandas as pd
from src.pipeline.retail_pipeline import RetailShelfPipeline
from src.pipeline.enhanced_retail_pipeline import EnhancedRetailPipeline
from src.classification.prototype_builder import PrototypeBuilder
from src.utils.config import ensure_directories
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import io

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Page config
st.set_page_config(
    page_title="Retail Shelf Monitor",
    page_icon="🛒",
    layout="wide"
)

def main():
    st.title("🛒 Retail Shelf Monitoring System")
    st.markdown("AI-powered product detection and brand classification")
    
    # Sidebar
    st.sidebar.title("Settings")
    st.sidebar.subheader("🔧 Detection Settings")
    use_ocr = st.sidebar.checkbox("Enable OCR", value=True, help="Extract text from products")
    show_debug = st.sidebar.checkbox("Show Debug Info", value=False, help="Display detailed detection information")
    
    # Add this in the sidebar of the main function
    if st.sidebar.button("🔄 Update Classifier", help="Rebuild classifier with new brands"):
        if rebuild_classifier_if_needed():
            st.sidebar.success("Classifier updated!")
        else:
            st.sidebar.error("Update failed!")
    
    # Initialize directories
    ensure_directories()
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Image Processing", "Build Prototypes", "View Results"]
    )
    
    if mode == "Image Processing":
        image_processing_page(use_ocr, show_debug)
    elif mode == "Build Prototypes":
        build_prototypes_page()
    elif mode == "View Results":
        view_results_page()

def image_processing_page(use_ocr=True, show_debug=False):
    st.header("📸 Product Detection & Brand Classification")
    
    # Initialize session state for persistent data
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'image_path' not in st.session_state:
        st.session_state.image_path = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload shelf image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a retail shelf with products"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        image_path = f"temp/{uploaded_file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Store in session state
        st.session_state.image_path = image_path
        
        # Display image
        st.subheader("Uploaded Image")
        image = Image.open(image_path)
        st.image(image, caption="Shelf Image", use_column_width=True)
        
        # Process button
        if st.button("🚀 Analyze Products", type="primary") and not st.session_state.processing_complete:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔧 Initializing detection pipeline...")
                progress_bar.progress(20)
                
                # Initialize pipeline
                pipeline = RetailShelfPipeline(use_ocr=use_ocr)
                
                status_text.text("🔍 Detecting and classifying products...")
                progress_bar.progress(60)
                
                # Process image
                results = pipeline.process_image(image_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Store results in session state
                st.session_state.processing_results = results
                st.session_state.processing_complete = True
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                if show_debug:
                    st.exception(e)
                return
    
    # Handle results and user feedback if processing is complete
    if st.session_state.processing_complete and st.session_state.processing_results:
        results = st.session_state.processing_results
        
        # Handle uncertain classifications
        uncertain_crops = results.get('uncertain_classifications', [])
        if uncertain_crops and not results.get('user_feedback_processed', False):
            st.warning(f"⚠️ Found {len(uncertain_crops)} products that need your review!")
            
            # User feedback section
            updated_results = handle_user_feedback_persistent(uncertain_crops, results)
            if updated_results:
                st.session_state.processing_results = updated_results
                results = updated_results
        
        # Display results
        display_results(results, st.session_state.image_path, show_debug)
        
        # Reset button
        if st.button("🔄 Process New Image", type="secondary"):
            # Clear session state
            st.session_state.processing_results = None
            st.session_state.image_path = None
            st.session_state.processing_complete = False
            st.session_state.submitted_classifications = {}
            st.rerun()

def handle_user_feedback_persistent(uncertain_crops, results):
    """Handle user feedback with persistent state management"""
    st.subheader("🤔 Product Classification Review")
    st.markdown("Please classify each product individually. Each classification requires submission.")
    
    # Initialize session state for tracking submissions
    if 'submitted_classifications' not in st.session_state:
        st.session_state.submitted_classifications = {}
    if 'dynamic_brands' not in st.session_state:
        st.session_state.dynamic_brands = get_existing_brands()
    if 'custom_brand_inputs' not in st.session_state:
        st.session_state.custom_brand_inputs = {}
    if 'brand_directories_created' not in st.session_state:
        st.session_state.brand_directories_created = {}
    
    updated_counts = results.get('brand_counts', {}).copy()
    all_submitted = True
    
    for i, uncertain in enumerate(uncertain_crops):
        crop_path = uncertain['crop_path']
        predicted_brand = uncertain.get('predicted_brand', 'UNKNOWN')
        confidence = uncertain.get('confidence', 0)
        
        st.write(f"### Product {i+1} of {len(uncertain_crops)}")
        
        # Check if this classification was already submitted
        is_submitted = i in st.session_state.submitted_classifications
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if os.path.exists(crop_path):
                crop_img = Image.open(crop_path)
                st.image(crop_img, caption=f"Product {i+1}", use_container_width=True)
        
        with col2:
            st.write(f"**AI Prediction:** {predicted_brand}")
            st.write(f"**Confidence:** {confidence:.2f}")
            
            if is_submitted:
                # Show submitted classification
                submitted_brand = st.session_state.submitted_classifications[i]
                st.success(f"✅ **Classified as:** {submitted_brand}")
                st.write("Classification submitted successfully!")
            else:
                # Show classification form
                all_submitted = False
                
                # Create a container for this product's form
                with st.container():
                    # Brand selection options
                    brand_options = ['Select Brand...'] + st.session_state.dynamic_brands + ['Enter Custom Brand']
                    
                    selected_option = st.selectbox(
                        f"Select brand for Product {i+1}:",
                        options=brand_options,
                        key=f"brand_option_{i}",
                        index=0
                    )
                    
                    final_brand = None
                    custom_brand_ready = False
                    
                    if selected_option == 'Enter Custom Brand':
                        # Custom brand input section
                        st.write("**Enter New Brand Name:**")
                        
                        # Use form to prevent auto-rerun
                        with st.form(key=f"custom_brand_form_{i}"):
                            custom_brand = st.text_input(
                                f"Brand name:",
                                key=f"custom_brand_input_{i}",
                                placeholder="e.g., DR-PEPPER, MOUNTAIN-DEW, COCA-COLA",
                                help="Enter the exact brand name as you want it to appear"
                            )
                            
                            create_brand_submitted = st.form_submit_button("Create Brand Directory")
                            
                            if create_brand_submitted and custom_brand and custom_brand.strip():
                                # Create brand directory
                                brand_name_clean = custom_brand.upper().strip()
                                success, result = create_brand_directory_for_product(
                                    brand_name_clean, crop_path, i
                                )
                                
                                if success:
                                    # Store that this brand was created
                                    st.session_state.brand_directories_created[i] = brand_name_clean
                                    
                                    # Add to dynamic brands list
                                    if brand_name_clean not in st.session_state.dynamic_brands:
                                        st.session_state.dynamic_brands.append(brand_name_clean)
                                        st.session_state.dynamic_brands.sort()
                                    
                                    st.success(f"✅ Created brand directory for: {brand_name_clean}")
                                    st.info("Now you can submit the classification below.")
                                    custom_brand_ready = True
                                    final_brand = brand_name_clean
                                else:
                                    st.error(f"❌ Failed to create brand: {result}")
                        
                        # Check if brand directory was already created for this product
                        if i in st.session_state.brand_directories_created:
                            final_brand = st.session_state.brand_directories_created[i]
                            custom_brand_ready = True
                            st.success(f"✅ Brand directory ready: {final_brand}")
                        
                    elif selected_option != 'Select Brand...':
                        final_brand = selected_option
                        
                        # Show brand directory status
                        brand_dir_name = final_brand.replace('-', '_').replace(' ', '_')
                        brand_dir = f"data/prototypes/{brand_dir_name}"
                        
                        if os.path.exists(brand_dir):
                            image_count = count_images_in_directory(brand_dir)
                            if image_count < 10:
                                st.info(f"📸 Brand directory has {image_count}/10 images. This image will be added to improve training.")
                            else:
                                st.info(f"📸 Brand directory has {image_count} images (sufficient for training).")
                        else:
                            st.warning(f"⚠️ Brand directory doesn't exist yet. Will be created.")
                        
                        st.info(f"Will classify as: **{final_brand}**")
                    
                    # Individual submit button (outside the form)
                    submit_disabled = (final_brand is None)
                    
                    if st.button(f"✅ Submit Classification for Product {i+1}", 
                               key=f"submit_{i}", 
                               type="primary",
                               disabled=submit_disabled):
                        
                        if final_brand:
                            # Process this individual classification with smart image addition
                            success = process_individual_classification_with_training(
                                i, final_brand, crop_path, selected_option == 'Enter Custom Brand'
                            )
                            
                            if success:
                                # Store in session state
                                st.session_state.submitted_classifications[i] = final_brand
                                
                                # Update counts
                                updated_counts[final_brand] = updated_counts.get(final_brand, 0) + 1
                                
                                st.success(f"✅ Product {i+1} classified as {final_brand}")
                                st.rerun()  # Refresh to show submitted state
                            else:
                                st.error("Failed to process classification")
                        else:
                            st.error("Please select or enter a brand name")
        
        st.divider()  # Visual separator between products
    
    # Show overall progress
    submitted_count = len(st.session_state.submitted_classifications)
    total_count = len(uncertain_crops)
    
    progress = submitted_count / total_count if total_count > 0 else 0
    st.progress(progress)
    st.write(f"**Progress:** {submitted_count}/{total_count} products classified")
    
    # Final completion check
    if all_submitted and submitted_count == total_count:
        st.success("🎉 All products have been classified!")
        
        # Check if classifier needs rebuilding
        needs_rebuild = check_if_classifier_needs_rebuild()
        if needs_rebuild:
            st.info("🔄 New training images were added. Consider rebuilding the classifier for better accuracy.")
            if st.button("🚀 Rebuild Classifier Now", type="secondary"):
                if rebuild_classifier_if_needed():
                    st.success("✅ Classifier rebuilt with new training data!")
        
        # Update results with all classifications
        results['brand_counts'] = updated_counts
        results['user_feedback_processed'] = True
        results['uncertain_classifications'] = []
        results['needs_user_feedback'] = False
        
        # Complete classification process
        if st.button("🔄 Complete Classification Process", type="primary"):
            st.success("Classification process completed!")
            return results
    
    elif submitted_count > 0:
        st.info(f"⏳ {total_count - submitted_count} more products need classification")
    
    return None

def count_images_in_directory(directory_path):
    """Count the number of image files in a directory"""
    if not os.path.exists(directory_path):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    count = 0
    
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            count += 1
    
    return count

def create_brand_directory_for_product(brand_name, crop_path, product_index):
    """Create brand directory and copy product image"""
    try:
        # Clean brand name for directory
        brand_dir_name = brand_name.replace('-', '_').replace(' ', '_')
        brand_dir = f"data/prototypes/{brand_dir_name}"
        
        # Create directory
        os.makedirs(brand_dir, exist_ok=True)
        print(f"📁 Created directory: {brand_dir}")
        
        # Copy crop image to brand directory
        if os.path.exists(crop_path):
            import shutil
            crop_filename = f"{brand_dir_name}_user_sample_{product_index}.jpg"
            destination_path = os.path.join(brand_dir, crop_filename)
            shutil.copy2(crop_path, destination_path)
            print(f"📸 Copied image: {crop_path} -> {destination_path}")
        else:
            print(f"⚠️ Crop image not found: {crop_path}")
        
        # Update metadata
        update_metadata_with_new_brand(brand_dir_name, brand_name)
        print(f"📝 Updated metadata for: {brand_name}")
        
        return True, brand_dir
        
    except Exception as e:
        print(f"❌ Error creating brand directory: {e}")
        return False, str(e)

def process_individual_classification_with_training(index, brand_name, crop_path, is_custom_brand):
    """Process individual classification and add image to training set if needed"""
    try:
        # Clean brand name for directory
        brand_dir_name = brand_name.replace('-', '_').replace(' ', '_')
        brand_dir = f"data/prototypes/{brand_dir_name}"
        
        # Ensure brand directory exists
        if not os.path.exists(brand_dir):
            print(f"📁 Creating brand directory: {brand_dir}")
            os.makedirs(brand_dir, exist_ok=True)
            
            # Update metadata for new brand
            update_metadata_with_new_brand(brand_dir_name, brand_name)
        
        # Check current image count in brand directory
        current_image_count = count_images_in_directory(brand_dir)
        
        # Add image to training set if directory has less than 10 images
        if current_image_count < 10:
            if os.path.exists(crop_path):
                import shutil
                import time
                
                # Create unique filename with timestamp
                timestamp = int(time.time())
                crop_filename = f"{brand_dir_name}_training_{current_image_count + 1}_{timestamp}.jpg"
                destination_path = os.path.join(brand_dir, crop_filename)
                
                # Copy image to brand directory
                shutil.copy2(crop_path, destination_path)
                
                print(f"📸 Added training image: {crop_path} -> {destination_path}")
                print(f"📊 Brand directory now has {current_image_count + 1} images")
                
                # Log the training addition
                log_training_addition(brand_name, crop_path, destination_path, current_image_count + 1)
                
                # Mark that classifier needs rebuilding
                mark_classifier_for_rebuild()
                
            else:
                print(f"⚠️ Crop image not found: {crop_path}")
        else:
            print(f"📊 Brand directory already has {current_image_count} images (sufficient)")
        
        # Log the classification
        classification_log = {
            'index': index,
            'brand': brand_name,
            'crop_path': crop_path,
            'is_custom': is_custom_brand,
            'training_image_added': current_image_count < 10,
            'brand_image_count': current_image_count + (1 if current_image_count < 10 else 0),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save to log file
        log_file = 'temp/classification_log.json'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(classification_log)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing classification: {e}")
        return False

def log_training_addition(brand_name, source_path, destination_path, new_count):
    """Log when a training image is added"""
    training_log = {
        'brand': brand_name,
        'source_path': source_path,
        'destination_path': destination_path,
        'new_image_count': new_count,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    log_file = 'temp/training_additions.json'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(training_log)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

def mark_classifier_for_rebuild():
    """Mark that the classifier needs rebuilding due to new training data"""
    rebuild_flag_file = 'temp/classifier_needs_rebuild.flag'
    with open(rebuild_flag_file, 'w') as f:
        f.write(pd.Timestamp.now().isoformat())

def check_if_classifier_needs_rebuild():
    """Check if classifier needs rebuilding"""
    rebuild_flag_file = 'temp/classifier_needs_rebuild.flag'
    return os.path.exists(rebuild_flag_file)

def clear_rebuild_flag():
    """Clear the rebuild flag after classifier is rebuilt"""
    rebuild_flag_file = 'temp/classifier_needs_rebuild.flag'
    if os.path.exists(rebuild_flag_file):
        os.remove(rebuild_flag_file)

def get_existing_brands():
    """Get list of existing brands from prototype directories"""
    brands = []
    prototype_dir = 'data/prototypes'
    
    if os.path.exists(prototype_dir):
        for item in os.listdir(prototype_dir):
            if os.path.isdir(os.path.join(prototype_dir, item)):
                # Convert directory name to brand name
                brand_name = item.replace('_', '-').upper()
                brands.append(brand_name)
    
    return sorted(brands)

def update_metadata_with_new_brand(brand_dir_name, brand_display_name):
    """Update metadata CSV with new brand information"""
    import pandas as pd
    
    meta_path = 'data/meta.csv'
    
    # Create or load existing metadata
    if os.path.exists(meta_path):
        try:
            df = pd.read_csv(meta_path)
        except:
            df = pd.DataFrame(columns=['SKU_ID', 'name', 'brand', 'flavor', 'category'])
    else:
        df = pd.DataFrame(columns=['SKU_ID', 'name', 'brand', 'flavor', 'category'])
    
    # Add new brand entry
    new_entry = {
        'SKU_ID': brand_dir_name,
        'name': brand_display_name,
        'brand': brand_display_name,
        'flavor': 'Original',
        'category': 'Beverages'
    }
    
    # Check if brand already exists
    if not df[df['SKU_ID'] == brand_dir_name].empty:
        return  # Brand already exists
    
    # Add new entry
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Save updated metadata
    os.makedirs('data', exist_ok=True)
    df.to_csv(meta_path, index=False)
    
    print(f"✅ Updated metadata with new brand: {brand_display_name}")

def display_results(results, image_path, show_debug=False):
    """Display processing results with individual classification tracking"""
    
    # Main metrics
    total_products = results.get('total_products', 0)
    brand_counts = results.get('brand_counts', {})
    uncertain_crops = results.get('uncertain_classifications', [])
    
    st.subheader("🎯 Detection Results")
    
    if total_products == 0:
        st.error("❌ No products detected!")
        return
    
    # Show classification status
    if uncertain_crops and not results.get('user_feedback_processed', False):
        submitted_count = len(st.session_state.get('submitted_classifications', {}))
        total_uncertain = len(uncertain_crops)
        
        if submitted_count > 0:
            st.info(f"📊 Classification Progress: {submitted_count}/{total_uncertain} products classified")
        else:
            st.warning(f"⚠️ {total_uncertain} products need individual classification")
    else:
        st.success(f"✅ {total_products} products detected and classified")
    
    # Brand counts display
    if brand_counts:
        st.subheader("📊 Brand Counts")
        
        # Create metrics display
        brands = list(brand_counts.keys())
        if len(brands) <= 4:
            cols = st.columns(len(brands))
        else:
            cols = st.columns(4)
        
        for i, (brand, count) in enumerate(brand_counts.items()):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.metric(brand, count)
        
        # Summary table
        brand_df = pd.DataFrame([
            {
                'Brand': brand,
                'Count': count,
                'Percentage': f"{(count/sum(brand_counts.values()))*100:.1f}%"
            }
            for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.dataframe(brand_df, use_container_width=True)
        
        total_branded = sum(brand_counts.values())
        st.info(f"**Total Classified Products:** {total_branded}")
        
    else:
        st.warning("⚠️ No brands identified yet")
    
    # Show training data status
    if show_debug:
        with st.expander("📚 Training Data Status"):
            show_training_data_status()
    
    # Show individual classification log
    if show_debug and os.path.exists('temp/classification_log.json'):
        with st.expander("📝 Individual Classification Log"):
            with open('temp/classification_log.json', 'r') as f:
                logs = json.load(f)
            
            for log in logs[-10:]:  # Show last 10 classifications
                training_added = "📸 +Training" if log.get('training_image_added', False) else ""
                st.write(f"**Product {log['index']+1}:** {log['brand']} {'(Custom)' if log['is_custom'] else ''} {training_added}")
                if log.get('brand_image_count'):
                    st.write(f"   Brand now has {log['brand_image_count']} training images")
    
    # Show annotated image
    if image_path and os.path.exists(image_path):
        annotated_img = create_simple_annotation(image_path, results.get('detection_data', []), results.get('classifications', []))
        if annotated_img:
            st.subheader("🖼️ Detection Results")
            st.image(annotated_img, caption="Detected Products with Brand Names", use_container_width=True)
    
    # Debug info
    if show_debug:
        with st.expander("🔍 Debug Information"):
            st.json(results)
    
    # Download results
    st.subheader("💾 Download Results")
    results_json = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="Download Results",
        data=results_json,
        file_name="shelf_analysis_results.json",
        mime="application/json"
    )

def show_training_data_status():
    """Show the current status of training data for each brand"""
    prototype_dir = 'data/prototypes'
    
    if not os.path.exists(prototype_dir):
        st.info("No training data directories found.")
        return
    
    brand_status = []
    
    for brand_dir in os.listdir(prototype_dir):
        brand_path = os.path.join(prototype_dir, brand_dir)
        if os.path.isdir(brand_path):
            image_count = count_images_in_directory(brand_path)
            brand_display = brand_dir.replace('_', '-').upper()
            
            status = "✅ Sufficient" if image_count >= 10 else "📸 Needs more"
            
            brand_status.append({
                'Brand': brand_display,
                'Images': image_count,
                'Status': status,
                'Progress': f"{min(image_count, 10)}/10"
            })
    
    if brand_status:
        df = pd.DataFrame(brand_status)
        st.dataframe(df, use_container_width=True)
        
        # Summary
        total_brands = len(brand_status)
        sufficient_brands = len([b for b in brand_status if b['Images'] >= 10])
        st.write(f"**Summary:** {sufficient_brands}/{total_brands} brands have sufficient training data")
    else:
        st.info("No brand directories found.")

def create_simple_annotation(image_path, detection_data, classifications=None):
    """Create simple annotated image with actual brand names"""
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, detection in enumerate(detection_data):
            bbox = detection.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                color = colors[i % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Use actual brand name from classifications if available
                brand_name = f"Product {i+1}"  # Default fallback
                
                if classifications and i < len(classifications):
                    classification = classifications[i]
                    sku_id = classification.get('sku_id', 'UNKNOWN')
                    confidence = classification.get('confidence', 0)
                    
                    # Convert SKU to display name
                    if sku_id != 'UNKNOWN' and confidence > 0.15:
                        brand_name = sku_id.replace('_', '-')
                        # Add confidence indicator for low confidence
                        if confidence < 0.5:
                            brand_name += f" ({confidence:.2f})"
                
                draw.text((x1, y1-20), brand_name, fill=color)
        
        return pil_img
    except Exception as e:
        print(f"Error creating annotation: {e}")
        return None

def build_prototypes_page():
    import torch
    st.header("🔧 Build Prototypes with OCR Integration")
    st.markdown("Build prototype embeddings for product classification using visual features + OCR text")
    
    # Check existing prototypes
    if os.path.exists('models/faiss_index/index.bin') and os.path.exists('models/faiss_index/labels.pt'):
        st.success("✅ Existing prototype index found!")
        
        try:
            labels = torch.load('models/faiss_index/labels.pt')
            st.info(f"Current index contains {len(labels)} SKUs: {', '.join(map(str, labels))}")
        except:
            st.warning("Could not read existing labels")
    
    # Show training data status
    st.subheader("📚 Current Training Data")
    show_training_data_status()
    
    # Rebuild prototypes button
    if st.button("🔄 Rebuild Prototypes with OCR", type="primary"):
        with st.spinner("Building prototype embeddings with OCR integration..."):
            try:
                builder = PrototypeBuilder()
                index, labels = builder.build_prototypes('data/prototypes', 'data/meta.csv')
                
                st.success(f"✅ Built prototypes for {len(labels)} SKUs with OCR integration")
                st.info("Prototype index saved to models/faiss_index/")
                
                # Clear rebuild flag
                clear_rebuild_flag()
                
                # Show what was built
                st.write("**SKUs processed:**")
                for label in labels:
                    st.write(f"• {label}")
                
            except Exception as e:
                st.error(f"Error building prototypes: {str(e)}")
                st.exception(e)
    
    # File upload section (keeping existing functionality)
    st.subheader("Upload New Data")
    
    # Metadata CSV
    meta_file = st.file_uploader(
        "Upload metadata CSV",
        type=['csv'],
        help="CSV with columns: SKU_ID, name, brand, flavor, shelf_position"
    )
    
    # Prototype images
    prototype_files = st.file_uploader(
        "Upload prototype images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload 3-10 images per SKU. Name format: SKU_ID_1.jpg, SKU_ID_2.jpg, etc."
    )
    
    if meta_file is not None and prototype_files:
        # Save metadata
        meta_path = "data/meta.csv"
        with open(meta_path, "wb") as f:
            f.write(meta_file.getbuffer())
        
        # Save prototype images
        prototype_dir = "data/prototypes"
        os.makedirs(prototype_dir, exist_ok=True)
        
        # Organize images by SKU
        for uploaded_file in prototype_files:
            # Extract SKU from filename (assuming format: SKU_ID_number.jpg)
            filename = uploaded_file.name
            sku_id = filename.split('_')[0]  # Simple extraction
            
            sku_dir = os.path.join(prototype_dir, sku_id)
            os.makedirs(sku_dir, exist_ok=True)
            
            file_path = os.path.join(sku_dir, filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.success(f"Uploaded {len(prototype_files)} prototype images")

def view_results_page():
    st.header("📈 View Results")
    
    # Load recent results
    result_files = ['enhanced_brand_results.json', 'final_results.json']
    
    for result_file in result_files:
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            st.subheader(f"Latest Results ({result_file})")
            display_results(results, "", True)  # Show debug info and annotation
            break
    else:
        st.info("No results available. Process an image first.")
    
    # Results history
    st.subheader("📁 Results Files")
    
    # List available result files
    result_files = []
    for file in os.listdir('.'):
        if file.endswith('_results.json') or file in ['final_results.json', 'enhanced_brand_results.json']:
            result_files.append(file)
    
    if result_files:
        selected_file = st.selectbox("Select results file", result_files)
        
        if st.button("Load Results"):
            with open(selected_file, 'r') as f:
                results = json.load(f)
            display_results(results, "", True)
    else:
        st.info("No result files found")

def rebuild_classifier_if_needed():
    """Rebuild classifier when new brands are added"""
    try:
        from src.classification.prototype_builder import PrototypeBuilder
        
        st.info("🔄 Rebuilding classifier with new brands...")
        with st.spinner("Updating classification model..."):
            builder = PrototypeBuilder()
            index, labels = builder.build_prototypes('data/prototypes', 'data/meta.csv')
            
            # Clear rebuild flag
            clear_rebuild_flag()
            
            st.success(f"✅ Classifier updated with {len(labels)} brands!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to rebuild classifier: {e}")
        return False

if __name__ == "__main__":
    main()
