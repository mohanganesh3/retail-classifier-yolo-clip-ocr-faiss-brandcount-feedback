import streamlit as st
import os
import json
from PIL import Image
import pandas as pd

# Fix OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from src.pipeline.enhanced_retail_pipeline import EnhancedRetailPipeline
from src.utils.config import ensure_directories

# Page config
st.set_page_config(
    page_title="Enhanced Retail Monitor",
    page_icon="🥤",
    layout="wide"
)

def main():
    st.title("🥤 Enhanced Retail Can/Tin Monitor")
    st.markdown("**Specialized for Coca-Cola, Pepsi, Sprite and other beverage cans/tins**")
    
    # Sidebar
    st.sidebar.title("Enhanced Detection")
    st.sidebar.markdown("✅ Detects bottles AND cans/tins")
    st.sidebar.markdown("✅ Brand identification via OCR")
    st.sidebar.markdown("✅ Accurate brand counts")
    st.sidebar.markdown("❌ No mock data")
    
    # Initialize directories
    ensure_directories()
    
    st.header("📸 Can/Tin Detection & Brand Classification")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload shelf image with cans/tins",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing Coca-Cola, Pepsi, Sprite and other beverage cans"
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        use_ocr = st.checkbox("Enable OCR for Brand Detection", value=True)
    with col2:
        show_debug = st.checkbox("Show Detection Details", value=True)
    
    if uploaded_file is not None:
        # Save uploaded file
        image_path = f"temp/{uploaded_file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display image
        st.subheader("Uploaded Image")
        image = Image.open(image_path)
        st.image(image, caption="Shelf Image", use_container_width=True)
        
        # Process button
        if st.button("🔍 Detect Cans & Classify Brands", type="primary"):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔧 Initializing enhanced detection...")
                progress_bar.progress(20)
                
                # Initialize enhanced pipeline
                pipeline = EnhancedRetailPipeline(use_ocr=use_ocr)
                
                status_text.text("🔍 Detecting cans and tins...")
                progress_bar.progress(50)
                
                # Process image
                results = pipeline.process_image(image_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Detection and classification complete!")
                
                # Check for uncertain classifications
                uncertain_crops = pipeline.brand_manager.get_uncertain_crops()
                
                if uncertain_crops:
                    st.warning(f"⚠️ Found {len(uncertain_crops)} uncertain classifications that need your review!")
                    
                    # User interaction for uncertain crops
                    handle_uncertain_classifications(pipeline.brand_manager, uncertain_crops, results)
                
                # Display results
                display_enhanced_results(results, show_debug)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if show_debug:
                    st.exception(e)

def handle_uncertain_classifications(brand_manager, uncertain_crops, results):
    """Handle user interaction for uncertain classifications"""
    st.subheader("🤔 Uncertain Classifications - Your Input Needed")
    
    user_classifications = []
    
    for i, uncertain in enumerate(uncertain_crops):
        crop_path = uncertain['crop_path']
        predicted_brand = uncertain['predicted_brand']
        confidence = uncertain['confidence']
        
        st.write(f"### Product {i+1}/{len(uncertain_crops)}")
        
        # Display crop image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if os.path.exists(crop_path):
                crop_img = Image.open(crop_path)
                st.image(crop_img, caption=f"Crop {i+1}", use_container_width=True)
            else:
                st.error("Crop image not found")
        
        with col2:
            st.write(f"**Predicted:** {predicted_brand}")
            st.write(f"**Confidence:** {confidence:.3f}")
            
            # User choice options
            brand_options = ['SKIP'] + list(brand_manager.brand_mapping.values()) + ['NEW BRAND']
            
            user_choice = st.selectbox(
                f"What brand is this product {i+1}?",
                options=brand_options,
                index=0 if predicted_brand == 'UNKNOWN' else (
                    brand_options.index(predicted_brand) if predicted_brand in brand_options else 0
                ),
                key=f"brand_choice_{i}"
            )
            
            new_brand_name = None
            if user_choice == 'NEW BRAND':
                new_brand_name = st.text_input(
                    f"Enter new brand name for product {i+1}:",
                    key=f"new_brand_{i}",
                    placeholder="e.g., DR-PEPPER"
                )
            
            # Process user choice
            if st.button(f"Confirm Choice for Product {i+1}", key=f"confirm_{i}"):
                if user_choice == 'NEW BRAND' and new_brand_name:
                    classification = brand_manager.process_user_classification(i, 'NEW', new_brand_name)
                elif user_choice != 'NEW BRAND':
                    classification = brand_manager.process_user_classification(i, user_choice)
                else:
                    st.error("Please enter a brand name for new brand")
                    continue
                
                if classification:
                    user_classifications.append(classification)
                    st.success(f"✅ Classified as {classification['brand']}")
                    
                    # Update results
                    if classification['brand'] != 'SKIP':
                        if classification['brand'] not in results['brand_counts']:
                            results['brand_counts'][classification['brand']] = 0
                        results['brand_counts'][classification['brand']] += 1
    
    # Rebuild vector database if new brands were added
    if any(cls['method'] == 'user_new' for cls in user_classifications):
        st.info("🔄 Rebuilding vector database with new brands...")
        if brand_manager.rebuild_vector_database():
            st.success("✅ Vector database rebuilt successfully!")
        else:
            st.error("❌ Failed to rebuild vector database")

def display_enhanced_results(results, show_debug=False):
    """Display enhanced detection and classification results"""
    
    # Main metrics
    total_products = results.get('total_products', 0)
    brand_counts = results.get('brand_counts', {})
    
    st.subheader("🎯 Detection Results")
    
    # Status message
    if total_products == 0:
        st.error("❌ No cans or tins detected!")
        return
    elif total_products < 5:
        st.warning(f"⚠️ Only {total_products} products detected")
    else:
        st.success(f"✅ {total_products} cans/tins detected")
    
    # Brand counts display
    if brand_counts:
        st.subheader("📊 Brand Counts")
        
        # Create columns for brand display
        brands = list(brand_counts.keys())
        if len(brands) <= 3:
            cols = st.columns(len(brands))
        else:
            cols = st.columns(3)
            # Show additional brands in rows
        
        for i, (brand, count) in enumerate(brand_counts.items()):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Format brand name for display
                display_name = brand.replace('_', ' ').title()
                st.metric(display_name, count)
        
        # Summary table
        st.subheader("📋 Detailed Brand Breakdown")
        brand_df = pd.DataFrame([
            {
                'Brand': brand.replace('_', ' ').title(),
                'Count': count,
                'Percentage': f"{(count/sum(brand_counts.values()))*100:.1f}%"
            }
            for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.dataframe(brand_df, use_container_width=True)
        
        # Total summary
        total_branded = sum(brand_counts.values())
        st.info(f"**Total Branded Products:** {total_branded} out of {total_products} detected")
        
    else:
        st.warning("⚠️ No brands could be identified from the detected products")
        st.info(f"Detected {total_products} products but could not classify brands")
    
    # OCR Results
    if results.get('ocr_texts'):
        with st.expander("📝 OCR Text Results"):
            ocr_texts = results['ocr_texts']
            text_found = [(i, text) for i, text in enumerate(ocr_texts) if text]
            
            st.write(f"**OCR Summary:** Found readable text in {len(text_found)} out of {len(ocr_texts)} products")
            
            if text_found:
                for i, (crop_idx, text) in enumerate(text_found):
                    st.write(f"**Product {crop_idx + 1}:** '{text}'")
            else:
                st.info("No readable text found in any products")
    
    # Debug information
    if show_debug:
        with st.expander("🔍 Detection Debug Information"):
            detection_data = results.get('detection_data', [])
            
            if detection_data:
                st.write(f"**Total Detections:** {len(detection_data)}")
                
                # Detection type breakdown
                can_types = {}
                confidences = []
                
                for det in detection_data:
                    can_type = det.get('can_type', 'unknown')
                    confidence = det.get('confidence', 0)
                    
                    if can_type in can_types:
                        can_types[can_type] += 1
                    else:
                        can_types[can_type] = 1
                    
                    confidences.append(confidence)
                
                # Show detection types
                st.write("**Detection Types:**")
                for can_type, count in can_types.items():
                    st.write(f"• {can_type.title()}: {count}")
                
                # Confidence stats
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    st.write(f"**Average Confidence:** {avg_conf:.3f}")
                    st.write(f"**Confidence Range:** {min(confidences):.3f} - {max(confidences):.3f}")
                
                # Classification details
                if results.get('classifications'):
                    st.write("**Classification Details:**")
                    classifications = results['classifications']
                    
                    for i, cls in enumerate(classifications[:10]):  # Show first 10
                        brand = cls.get('brand', 'UNKNOWN')
                        confidence = cls.get('confidence', 0)
                        method = cls.get('method', 'unknown')
                        ocr_text = cls.get('ocr_text', '')
                        
                        st.write(f"**Product {i+1}:** {brand} (conf: {confidence:.3f}, method: {method}) - '{ocr_text}'")
            
            else:
                st.error("No detection data available for debugging")
    
    # Download results
    st.subheader("💾 Download Results")
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Enhanced Results JSON",
        data=results_json,
        file_name="enhanced_can_detection_results.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
