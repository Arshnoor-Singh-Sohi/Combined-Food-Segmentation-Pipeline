"""
utils/visualization.py - Drawing and Display Utilities
=====================================================
Handles all visualization tasks like drawing bounding boxes and labels

This module demonstrates the power of separation of concerns. Instead of
mixing visualization logic with detection logic (which makes code messy),
we keep all drawing and display functions in one focused place.

Think of this as your "artist" module - it knows how to make things look good
but doesn't need to understand the complex logic of segmentation or detection.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import colorsys

class FoodVisualizer:
    """
    Handles all visualization tasks for the food detection system
    
    This class encapsulates the 'how to draw' knowledge separate from the
    'what to detect' knowledge, making both easier to understand and modify.
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        # Color scheme for consistent, attractive visualizations
        self.colors = self._generate_color_palette(20)
        
        # Text settings
        self.font_scale = 0.6
        self.font_thickness = 2
        self.box_thickness = 2
        
        # Try to load a nice font, fall back to default if not available
        try:
            # You can replace this with any TTF font file you prefer
            self.pil_font = ImageFont.truetype("arial.ttf", 16)
        except:
            self.pil_font = ImageFont.load_default()
    
    def _generate_color_palette(self, num_colors):
        """
        Generate a visually pleasing color palette
        
        This uses HSV color space to create colors that are both distinct
        and aesthetically pleasing. HSV is great for this because we can
        fix saturation and value while varying hue for different colors.
        
        Args:
            num_colors (int): Number of colors to generate
            
        Returns:
            list: List of (R, G, B) color tuples
        """
        colors = []
        for i in range(num_colors):
            # Distribute hues evenly around the color wheel
            hue = i / num_colors
            
            # Use high saturation and value for vibrant colors
            saturation = 0.8 + (i % 3) * 0.1  # Slight variation in saturation
            value = 0.9
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert to 0-255 range and format for OpenCV (BGR)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
    def draw_detections(self, image, detections, show_confidence=True, show_crops=False):
        """
        Draw bounding boxes and labels for detected food items
        
        This is the main method you'll use to visualize detection results.
        It creates a clean, professional-looking annotated image.
        
        Args:
            image (PIL.Image): Original image to annotate
            detections (list): List of detection results from FoodDetector
            show_confidence (bool): Whether to show confidence scores
            show_crops (bool): Whether to show crop previews
            
        Returns:
            PIL.Image: Annotated image with bounding boxes and labels
        """
        # Convert PIL image to OpenCV format for drawing
        # PIL uses RGB, OpenCV uses BGR, so we need to convert
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Draw each detection
        for i, detection in enumerate(detections):
            # Get detection information
            bbox = detection['bbox']
            food_name = detection['food_name']
            confidence = detection.get('confidence', 0.0)
            
            # Unpack bounding box coordinates
            x, y, w, h = bbox
            
            # Choose color for this detection (cycle through our palette)
            color = self.colors[i % len(self.colors)]
            
            # Draw the bounding box rectangle
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, self.box_thickness)
            
            # Prepare the label text
            if show_confidence:
                label = f"{food_name} ({confidence:.2f})"
            else:
                label = food_name
            
            # Draw the label with background for better readability
            self._draw_label_with_background(image_cv, label, (x, y), color)
        
        # Convert back to PIL format and return
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(image_rgb)
        
        return result_image
    
    def _draw_label_with_background(self, image_cv, text, position, color):
        """
        Draw text with a colored background for better readability
        
        This is a helper method that demonstrates good UI design principles:
        - Always ensure text is readable against any background
        - Use consistent styling throughout the application
        - Make information easy to scan quickly
        
        Args:
            image_cv (numpy.ndarray): OpenCV image array
            text (str): Text to draw
            position (tuple): (x, y) position for the text
            color (tuple): BGR color for the background
        """
        x, y = position
        
        # Get text size to create appropriately sized background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        
        # Add some padding around the text
        padding = 4
        bg_width = text_width + (padding * 2)
        bg_height = text_height + baseline + (padding * 2)
        
        # Calculate background rectangle coordinates
        # Position label above the bounding box if possible
        label_y = y - bg_height if y - bg_height > 0 else y + bg_height
        
        # Draw the background rectangle
        cv2.rectangle(
            image_cv,
            (x, label_y),
            (x + bg_width, label_y + bg_height),
            color,
            -1  # Fill the rectangle
        )
        
        # Draw the text in a contrasting color
        # Use white text on dark backgrounds, black on light backgrounds
        text_color = self._get_contrasting_text_color(color)
        
        cv2.putText(
            image_cv,
            text,
            (x + padding, label_y + text_height + padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            text_color,
            self.font_thickness
        )
    
    def _get_contrasting_text_color(self, bg_color):
        """
        Choose black or white text based on background color brightness
        
        This implements a simple algorithm for ensuring text readability:
        calculate the perceived brightness of the background and choose
        the contrasting text color.
        
        Args:
            bg_color (tuple): BGR background color
            
        Returns:
            tuple: BGR color for text (black or white)
        """
        b, g, r = bg_color
        
        # Calculate perceived brightness using standard luminance formula
        # This weights colors based on human eye sensitivity
        brightness = (0.299 * r + 0.587 * g + 0.114 * b)
        
        # Use white text on dark backgrounds, black on light backgrounds
        if brightness < 128:
            return (255, 255, 255)  # White
        else:
            return (0, 0, 0)  # Black
    
    def create_detection_grid(self, detections, grid_size=(4, 3)):
        """
        Create a grid view of detected food item crops
        
        This is useful for quickly reviewing what was detected in an image.
        It's like creating a "contact sheet" of all the food items found.
        
        Args:
            detections (list): List of detection results with crop images
            grid_size (tuple): (cols, rows) for the grid layout
            
        Returns:
            PIL.Image: Grid image showing all detected crops
        """
        cols, rows = grid_size
        max_items = cols * rows
        
        # Filter detections that have crop images
        crops_with_names = []
        for detection in detections[:max_items]:
            if 'crop' in detection and detection['crop'] is not None:
                crops_with_names.append((detection['crop'], detection['food_name']))
        
        if not crops_with_names:
            # Return a placeholder image if no crops available
            placeholder = Image.new('RGB', (400, 300), color=(240, 240, 240))
            return placeholder
        
        # Calculate cell size based on the largest crop
        max_width = max(crop.width for crop, _ in crops_with_names)
        max_height = max(crop.height for crop, _ in crops_with_names)
        cell_size = max(max_width, max_height, 100)  # Minimum 100px
        
        # Create the grid image
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
        
        # Place each crop in the grid
        for i, (crop, food_name) in enumerate(crops_with_names):
            if i >= max_items:
                break
            
            # Calculate grid position
            col = i % cols
            row = i // cols
            
            # Calculate cell position
            cell_x = col * cell_size
            cell_y = row * cell_size
            
            # Resize crop to fit cell while maintaining aspect ratio
            crop_resized = self._resize_to_fit(crop, (cell_size - 20, cell_size - 40))  # Leave space for label
            
            # Center the crop in the cell
            paste_x = cell_x + (cell_size - crop_resized.width) // 2
            paste_y = cell_y + (cell_size - crop_resized.height) // 2 - 10  # Offset for label space
            
            # Paste the crop
            grid_image.paste(crop_resized, (paste_x, paste_y))
            
            # Add label below the crop
            # Note: For simplicity, we're not drawing text here, but you could
            # use PIL's ImageDraw to add labels if needed
        
        return grid_image
    
    def _resize_to_fit(self, image, max_size):
        """
        Resize image to fit within max_size while maintaining aspect ratio
        
        This is a utility function that demonstrates good image handling:
        always preserve aspect ratio to avoid distortion, and handle edge
        cases gracefully.
        
        Args:
            image (PIL.Image): Image to resize
            max_size (tuple): (max_width, max_height)
            
        Returns:
            PIL.Image: Resized image
        """
        max_width, max_height = max_size
        current_width, current_height = image.size
        
        # Calculate scale factor to fit within max_size
        scale_w = max_width / current_width
        scale_h = max_height / current_height
        scale = min(scale_w, scale_h)  # Use the smaller scale to ensure it fits
        
        # Calculate new size
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # Resize using high-quality resampling
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def create_summary_visualization(self, image, detections, stats=None):
        """
        Create a comprehensive summary visualization
        
        This creates a "dashboard" style image that shows:
        - The original image with annotations
        - Statistics about the detections
        - A preview of detected items
        
        This demonstrates how to combine multiple visualization elements
        into a single, informative display.
        
        Args:
            image (PIL.Image): Original image
            detections (list): Detection results
            stats (dict): Optional statistics to display
            
        Returns:
            PIL.Image: Comprehensive summary visualization
        """
        # This is a placeholder for a more complex visualization
        # You could implement a dashboard-style layout here
        
        # For now, just return the annotated image
        return self.draw_detections(image, detections, show_confidence=True)

# Convenience function for quick use
def draw_detections(image, detections):
    """
    Quick function to draw detections on an image
    
    This is a convenience function that creates a visualizer instance
    and draws the detections. It's perfect for simple use cases where
    you just want to quickly annotate an image.
    
    Args:
        image (PIL.Image): Image to annotate
        detections (list): Detection results
        
    Returns:
        PIL.Image: Annotated image
    """
    visualizer = FoodVisualizer()
    return visualizer.draw_detections(image, detections)