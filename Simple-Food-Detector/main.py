"""
main.py - Simple Food Detection App
=====================================
Main Streamlit application for food detection using SAM + OpenAI
"""

import streamlit as st
import os
from PIL import Image
import json
from datetime import datetime

# Import our custom modules
from src.segmentation import SAMSegmenter
from src.detection import FoodDetector
from src.database_manager import DatabaseManager
from utils.visualization import draw_detections
from utils.image_utils import save_uploaded_image
from config import Config

# Page configuration
st.set_page_config(
    page_title="Simple Food Detector",
    page_icon="🍎",
    layout="wide"
)

def init_app():
    """Initialize the application components"""
    # Create data directories if they don't exist
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/crops", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    # Initialize components
    if 'sam_segmenter' not in st.session_state:
        st.session_state.sam_segmenter = SAMSegmenter()
    
    if 'food_detector' not in st.session_state:
        st.session_state.food_detector = FoodDetector()
    
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

def main():
    # Initialize app
    init_app()
    
    # App header
    st.title("🍎 Simple Food Detection System")
    st.markdown("Upload an image and detect food items automatically!")
    
    # Sidebar for stats and settings
    with st.sidebar:
        st.header("📊 Statistics")
        
        # Get database stats
        total_images = st.session_state.db_manager.get_total_images()
        total_detections = st.session_state.db_manager.get_total_detections()
        
        st.metric("Images Processed", total_images)
        st.metric("Food Items Detected", total_detections)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
        max_detections = st.slider("Max Detections", 1, 20, 10)
        
        # Model status
        st.header("🤖 Model Status")
        if st.session_state.sam_segmenter.is_loaded():
            st.success("✅ SAM Model Loaded")
        else:
            st.error("❌ SAM Model Not Found")
            st.info("Place sam_vit_h_4b8939.pth in models/ folder")
        
        if st.session_state.food_detector.is_configured():
            st.success("✅ OpenAI API Configured")
        else:
            st.error("❌ OpenAI API Not Configured")
            st.info("Set OPENAI_API_KEY in config")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo containing food items"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("🔍 Detect Food Items", type="primary", use_container_width=True):
                # Save uploaded image
                image_path = save_uploaded_image(uploaded_file, image)
                
                # Process the image
                process_image(image, image_path, uploaded_file.name, max_detections)
    
    with col2:
        st.header("📋 Recent Results")
        
        # Display recent results from database
        recent_results = st.session_state.db_manager.get_recent_results(limit=5)
        
        if recent_results:
            for result in recent_results:
                with st.expander(f"{result['filename']} - {result['timestamp'][:16]}"):
                    st.write(f"**Detected Items:** {len(result['detections'])}")
                    
                    # Show detected food items
                    for detection in result['detections']:
                        food_name = detection.get('food_name', 'Unknown')
                        confidence = detection.get('confidence', 0)
                        st.write(f"• {food_name} (confidence: {confidence:.2f})")
        else:
            st.info("No results yet. Upload an image to get started!")

def process_image(image, image_path, filename, max_detections):
    """Process uploaded image through the detection pipeline"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Segment image with SAM
        status_text.text("🔍 Segmenting image with SAM...")
        progress_bar.progress(20)
        
        segments = st.session_state.sam_segmenter.segment_image(image, max_segments=max_detections)
        
        if not segments:
            st.error("No segments found in the image. Try a different image.")
            return
        
        st.success(f"Found {len(segments)} segments!")
        
        # Step 2: Detect food in each segment
        status_text.text("🤖 Detecting food items with AI...")
        progress_bar.progress(50)
        
        detections = []
        detection_col1, detection_col2 = st.columns([1, 1])
        
        for i, segment in enumerate(segments):
            # Update progress
            progress = 50 + (40 * i // len(segments))
            progress_bar.progress(progress)
            
            # Detect food in this segment
            food_result = st.session_state.food_detector.detect_food(segment['crop'])
            
            if food_result and food_result['food_name'] != 'not_food':
                detection = {
                    'bbox': segment['bbox'],
                    'food_name': food_result['food_name'],
                    'confidence': food_result.get('confidence', 0.0),
                    'crop': segment['crop']
                }
                detections.append(detection)
                
                # Display detection in real-time
                col = detection_col1 if len(detections) % 2 == 1 else detection_col2
                with col:
                    st.image(segment['crop'], caption=f"🍽️ {food_result['food_name']}", width=150)
        
        # Step 3: Save results to database
        status_text.text("💾 Saving results...")
        progress_bar.progress(90)
        
        if detections:
            # Save to database
            st.session_state.db_manager.save_detection_result(
                filename=filename,
                image_path=image_path,
                detections=detections
            )
            
            # Create and display annotated image
            annotated_image = draw_detections(image, detections)
            
            # Save annotated result
            result_path = f"data/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            annotated_image.save(result_path)
            
            # Display final results
            progress_bar.progress(100)
            status_text.text("✅ Detection Complete!")
            
            st.success(f"🎉 Detected {len(detections)} food items!")
            st.image(annotated_image, caption="Detected Food Items", use_container_width=True)
            
            # Show detection summary
            st.subheader("🍽️ Detection Summary")
            for i, detection in enumerate(detections):
                st.write(f"{i+1}. **{detection['food_name']}** (confidence: {detection['confidence']:.2f})")
        
        else:
            st.warning("No food items detected in this image.")
            progress_bar.progress(100)
            status_text.text("✅ Processing Complete - No food found")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        progress_bar.empty()
        status_text.empty()
    
    finally:
        # Clean up progress indicators after a moment
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

if __name__ == "__main__":
    main()