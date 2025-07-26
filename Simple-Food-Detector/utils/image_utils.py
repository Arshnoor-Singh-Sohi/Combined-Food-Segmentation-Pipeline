"""
utils/image_utils.py - Image Processing Helpers
===============================================
Utility functions for image handling, preprocessing, and file management

This module contains all the "plumbing" functions that handle the practical
aspects of working with images in a computer vision application. Think of
this as your image toolkit - it handles all the common operations you need
but don't want to rewrite every time.

The key principle here is DRY (Don't Repeat Yourself). Instead of having
image processing code scattered throughout your application, we centralize
it here where it can be tested, optimized, and reused.
"""

import os
import hashlib
from datetime import datetime
from PIL import Image, ImageOps, ExifTags
import numpy as np
from config import Config

def save_uploaded_image(uploaded_file, pil_image):
    """
    Save an uploaded image file to the uploads directory
    
    This function handles the practical aspects of saving uploaded files
    in a way that avoids conflicts and maintains organization. It demonstrates
    several important concepts in file handling:
    
    1. Safe filename generation (avoiding conflicts and invalid characters)
    2. Consistent directory structure
    3. Metadata preservation
    4. Error handling
    
    Args:
        uploaded_file: Streamlit uploaded file object
        pil_image (PIL.Image): The PIL image object
        
    Returns:
        str: Path to the saved image file
    """
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        
        # Generate a safe, unique filename
        # This prevents conflicts and handles special characters gracefully
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Clean the original filename to remove problematic characters
        original_name = uploaded_file.name
        name_part, ext_part = os.path.splitext(original_name)
        
        # Remove or replace characters that could cause file system issues
        safe_name = "".join(c for c in name_part if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        # Limit filename length to avoid file system issues
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        
        # Construct the final filename
        filename = f"{timestamp}_{safe_name}{ext_part.lower()}"
        filepath = os.path.join(Config.UPLOAD_DIR, filename)
        
        # Save the image with appropriate quality settings
        # JPEG with 95% quality is a good balance between size and quality
        if pil_image.mode == 'RGBA':
            # Convert RGBA to RGB for JPEG saving
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            rgb_image.paste(pil_image, mask=pil_image.split()[-1])
            rgb_image.save(filepath, 'JPEG', quality=95)
        else:
            pil_image.save(filepath, 'JPEG', quality=95)
        
        print(f"💾 Saved uploaded image: {filename}")
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving uploaded image: {str(e)}")
        raise

def preprocess_image_for_sam(image):
    """
    Preprocess an image to optimize it for SAM segmentation
    
    This function applies several preprocessing steps that can improve
    SAM's performance. These are based on computer vision best practices
    and empirical testing with SAM:
    
    1. Orientation correction (handles phone photos)
    2. Size optimization (balances quality vs processing time)
    3. Color space normalization
    4. Contrast enhancement when needed
    
    Args:
        image (PIL.Image): Input image to preprocess
        
    Returns:
        PIL.Image: Preprocessed image optimized for SAM
    """
    try:
        # Step 1: Fix image orientation based on EXIF data
        # This is crucial for photos taken with phones/cameras
        image = fix_image_orientation(image)
        
        # Step 2: Optimize image size for processing
        # SAM works well with images up to ~1024px on the longest side
        image = optimize_image_size(image, max_size=Config.MAX_IMAGE_SIZE)
        
        # Step 3: Enhance contrast if the image appears too dark or washed out
        # This can help SAM detect boundaries better
        image = enhance_contrast_if_needed(image)
        
        # Step 4: Ensure the image is in RGB format
        # SAM expects RGB input, so convert if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {str(e)}")
        # Return original image if preprocessing fails
        return image

def fix_image_orientation(image):
    """
    Fix image orientation based on EXIF data
    
    Modern cameras and phones embed orientation information in EXIF data.
    This function reads that information and rotates the image accordingly.
    Without this step, images might appear sideways or upside down.
    
    This is a great example of handling real-world data messiness in a
    clean, robust way.
    
    Args:
        image (PIL.Image): Image that might need rotation
        
    Returns:
        PIL.Image: Correctly oriented image
    """
    try:
        # Get EXIF data if it exists
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            
            # Look for orientation tag
            for tag, value in exif.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                    # Apply the appropriate rotation based on orientation value
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
                    break
    except:
        # If anything goes wrong with EXIF processing, just continue
        # It's better to have an unrotated image than a crashed program
        pass
    
    return image

def optimize_image_size(image, max_size=1024):
    """
    Resize image to optimize processing speed while maintaining quality
    
    This function demonstrates the classic tradeoff in computer vision:
    larger images provide more detail but take longer to process and use
    more memory. We find a sweet spot that gives good results efficiently.
    
    The algorithm maintains aspect ratio and only shrinks images that are
    too large, never enlarging smaller images (which would reduce quality).
    
    Args:
        image (PIL.Image): Image to potentially resize
        max_size (int): Maximum dimension (width or height) in pixels
        
    Returns:
        PIL.Image: Optimally sized image
    """
    width, height = image.size
    
    # Only resize if the image is larger than max_size
    if max(width, height) > max_size:
        # Calculate the scaling factor to fit within max_size
        scale_factor = max_size / max(width, height)
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize using high-quality resampling
        # LANCZOS provides excellent quality for downsizing
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        print(f"📏 Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return image

def enhance_contrast_if_needed(image):
    """
    Enhance image contrast if it appears too flat or washed out
    
    This function analyzes the image's histogram to determine if contrast
    enhancement would be beneficial. It's a good example of adaptive
    processing - only applying enhancements when they're actually needed.
    
    The algorithm looks at the distribution of pixel values. If most pixels
    are clustered in a narrow range (indicating low contrast), it applies
    enhancement.
    
    Args:
        image (PIL.Image): Image to potentially enhance
        
    Returns:
        PIL.Image: Enhanced image (or original if enhancement not needed)
    """
    try:
        # Convert to grayscale for histogram analysis
        gray = image.convert('L')
        
        # Calculate histogram
        histogram = gray.histogram()
        
        # Calculate the range of pixel values that contain 90% of the image
        # This ignores extreme outliers that might skew the analysis
        total_pixels = sum(histogram)
        cumulative = 0
        min_val, max_val = 0, 255
        
        # Find 5th percentile
        target = total_pixels * 0.05
        for i, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target:
                min_val = i
                break
        
        # Find 95th percentile
        cumulative = 0
        target = total_pixels * 0.05
        for i in range(255, -1, -1):
            cumulative += histogram[i]
            if cumulative >= target:
                max_val = i
                break
        
        # If the range is narrow, apply contrast enhancement
        value_range = max_val - min_val
        if value_range < 128:  # Arbitrary threshold for "low contrast"
            # Use PIL's autocontrast function, which is a simple but effective
            # method for improving contrast
            image = ImageOps.autocontrast(image, cutoff=1)
            print("🎨 Applied contrast enhancement")
    
    except:
        # If anything goes wrong, return the original image
        # Robust error handling prevents crashes from minor processing issues
        pass
    
    return image

def calculate_image_hash(image):
    """
    Calculate a perceptual hash of an image for duplicate detection
    
    This function creates a "fingerprint" of an image that can be used to
    detect duplicates or near-duplicates. It's useful for preventing the
    same image from being processed multiple times.
    
    The hash is based on image content, not file data, so it will match
    even if the same image is saved in different formats or with different
    compression.
    
    Args:
        image (PIL.Image): Image to hash
        
    Returns:
        str: Hexadecimal hash string
    """
    try:
        # Resize to a small, standard size for consistent hashing
        # This removes the influence of image size on the hash
        small_image = image.resize((8, 8), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray_image = small_image.convert('L')
        
        # Get pixel values as a list
        pixels = list(gray_image.getdata())
        
        # Calculate the average pixel value
        avg_pixel = sum(pixels) / len(pixels)
        
        # Create a binary string based on whether each pixel is above/below average
        # This creates a simple but effective perceptual hash
        hash_bits = ''.join('1' if pixel >= avg_pixel else '0' for pixel in pixels)
        
        # Convert to hexadecimal for more compact representation
        hash_value = hex(int(hash_bits, 2))[2:]  # Remove '0x' prefix
        
        return hash_value
        
    except Exception as e:
        print(f"⚠️ Error calculating image hash: {str(e)}")
        return None

def validate_image_file(filepath):
    """
    Validate that a file is a valid image and get its properties
    
    This function demonstrates defensive programming - checking that inputs
    are valid before processing them. It prevents crashes and provides
    useful error messages when things go wrong.
    
    Args:
        filepath (str): Path to the image file
        
    Returns:
        dict: Image properties if valid, None if invalid
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            return None
        
        # Try to open the image
        with Image.open(filepath) as img:
            # Basic validation - can we read the image?
            img.verify()  # This checks the image integrity
        
        # Reopen for getting properties (verify() invalidates the image)
        with Image.open(filepath) as img:
            properties = {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'file_size': os.path.getsize(filepath),
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
        
        return properties
        
    except Exception as e:
        print(f"❌ Invalid image file {filepath}: {str(e)}")
        return None

def create_thumbnail(image, size=(150, 150)):
    """
    Create a thumbnail of an image for quick previews
    
    Thumbnails are essential for responsive user interfaces. They allow you
    to show image previews without loading full-size images, which improves
    performance and user experience.
    
    Args:
        image (PIL.Image): Source image
        size (tuple): Maximum thumbnail dimensions (width, height)
        
    Returns:
        PIL.Image: Thumbnail image
    """
    try:
        # Create a copy to avoid modifying the original
        thumbnail = image.copy()
        
        # Use thumbnail() method which maintains aspect ratio
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        
        return thumbnail
        
    except Exception as e:
        print(f"⚠️ Error creating thumbnail: {str(e)}")
        return image  # Return original if thumbnail creation fails

# Convenience function for the most common use case
def prepare_image_for_detection(uploaded_file):
    """
    Complete image preparation pipeline for food detection
    
    This function combines all the preprocessing steps into a single,
    easy-to-use function. It represents the complete "image preparation
    pipeline" that takes a raw uploaded file and produces an optimized
    image ready for detection.
    
    This is a great example of creating high-level interfaces that hide
    complexity while still allowing access to individual components when
    needed.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (processed_image, saved_path) or (None, None) if failed
    """
    try:
        # Open the uploaded file as a PIL image
        image = Image.open(uploaded_file)
        
        # Apply all preprocessing steps
        processed_image = preprocess_image_for_sam(image)
        
        # Save the processed image
        saved_path = save_uploaded_image(uploaded_file, processed_image)
        
        return processed_image, saved_path
        
    except Exception as e:
        print(f"❌ Error preparing image: {str(e)}")
        return None, None