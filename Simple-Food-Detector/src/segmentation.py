"""
src/segmentation.py - SAM Segmentation Logic
===========================================
Handles image segmentation using the Segment Anything Model (SAM)
"""

import os
import numpy as np
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from config import Config

class SAMSegmenter:
    """
    Wrapper class for SAM (Segment Anything Model) segmentation
    
    This class handles loading the SAM model and performing automatic
    segmentation on images to identify distinct objects/regions.
    """
    
    def __init__(self):
        """Initialize the SAM segmenter"""
        self.sam_model = None
        self.mask_generator = None
        self.is_model_loaded = False
        
        # Try to load the model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the SAM model and create mask generator"""
        try:
            # Check if model file exists
            if not os.path.exists(Config.SAM_MODEL_PATH):
                print(f"⚠️ SAM model not found at {Config.SAM_MODEL_PATH}")
                return False
            
            print("🔄 Loading SAM model... (this may take a moment)")
            
            # Load the SAM model
            self.sam_model = sam_model_registry[Config.SAM_MODEL_TYPE](
                checkpoint=Config.SAM_MODEL_PATH
            )
            
            # Move model to appropriate device (CPU or GPU)
            self.sam_model.to(Config.DEVICE)
            
            # Create automatic mask generator with optimized settings
            self.mask_generator = SamAutomaticMaskGenerator(
                self.sam_model,
                **Config.get_sam_config()
            )
            
            self.is_model_loaded = True
            print("✅ SAM model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SAM model: {str(e)}")
            self.is_model_loaded = False
            return False
    
    def is_loaded(self):
        """Check if the SAM model is loaded and ready"""
        return self.is_model_loaded
    
    def segment_image(self, image, max_segments=None):
        """
        Segment an image into distinct objects/regions
        
        Args:
            image (PIL.Image): Input image to segment
            max_segments (int, optional): Maximum number of segments to return
            
        Returns:
            list: List of segment dictionaries containing bbox, crop, and metadata
        """
        if not self.is_loaded():
            print("❌ SAM model not loaded. Cannot perform segmentation.")
            return []
        
        try:
            # Convert PIL image to numpy array for SAM
            image_np = np.array(image)
            
            # Resize image if it's too large (for performance)
            height, width = image_np.shape[:2]
            if max(height, width) > Config.MAX_IMAGE_SIZE:
                scale_factor = Config.MAX_IMAGE_SIZE / max(height, width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize using PIL for better quality
                image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                image_np = np.array(image_resized)
                
                print(f"📏 Resized image from {width}x{height} to {new_width}x{new_height}")
            else:
                scale_factor = 1.0
                image_resized = image
            
            print("🔍 Generating masks with SAM...")
            
            # Generate masks using SAM
            masks = self.mask_generator.generate(image_np)
            
            print(f"🎯 Generated {len(masks)} initial masks")
            
            # Process and filter masks
            segments = self._process_masks(masks, image_resized, scale_factor, max_segments)
            
            print(f"✅ Returning {len(segments)} filtered segments")
            return segments
            
        except Exception as e:
            print(f"❌ Error during segmentation: {str(e)}")
            return []
    
    def _process_masks(self, masks, image, scale_factor, max_segments):
        """
        Process SAM masks into usable segments
        
        Args:
            masks (list): Raw masks from SAM
            image (PIL.Image): Processed image
            scale_factor (float): Scale factor applied to image
            max_segments (int): Maximum segments to return
            
        Returns:
            list: Processed segments with bounding boxes and crops
        """
        segments = []
        
        # Sort masks by area (largest first) and take top candidates
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Limit number of masks to process for performance
        if max_segments:
            masks_sorted = masks_sorted[:max_segments * 2]  # Process extra, then filter
        
        image_np = np.array(image)
        image_width, image_height = image.size
        
        for i, mask_data in enumerate(masks_sorted):
            try:
                # Extract mask information
                mask = mask_data['segmentation']
                bbox = mask_data['bbox']  # [x, y, width, height]
                area = mask_data['area']
                
                # Scale bbox back to original coordinates if image was resized
                if scale_factor != 1.0:
                    bbox = [int(coord / scale_factor) for coord in bbox]
                
                x, y, w, h = bbox
                
                # Filter out segments that are too small or too large
                if area < Config.MIN_SEGMENT_AREA:
                    continue
                
                # Filter out segments that are too close to image edges
                if x < 10 or y < 10 or (x + w) > (image_width - 10) or (y + h) > (image_height - 10):
                    continue
                
                # Filter out segments with extreme aspect ratios
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    continue
                
                # Crop the segment from the image
                crop = image.crop((x, y, x + w, y + h))
                
                # Skip crops that are too small
                if crop.width < 20 or crop.height < 20:
                    continue
                
                # Create segment dictionary
                segment = {
                    'id': i,
                    'bbox': [x, y, w, h],
                    'area': area,
                    'crop': crop,
                    'aspect_ratio': aspect_ratio,
                    'confidence': mask_data.get('predicted_iou', 0.0)
                }
                
                segments.append(segment)
                
                # Stop if we have enough segments
                if max_segments and len(segments) >= max_segments:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error processing mask {i}: {str(e)}")
                continue
        
        return segments
    
    def visualize_segments(self, image, segments):
        """
        Create a visualization of the segments on the original image
        
        Args:
            image (PIL.Image): Original image
            segments (list): List of segments from segment_image()
            
        Returns:
            PIL.Image: Image with segment bounding boxes drawn
        """
        # Convert to OpenCV format for drawing
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes for each segment
        for i, segment in enumerate(segments):
            bbox = segment['bbox']
            x, y, w, h = bbox
            
            # Draw rectangle
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, thickness)
            
            # Add segment ID label
            label = f"Segment {segment['id']}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(image_cv, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(image_cv, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Convert back to PIL and return
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)