"""
src/detection.py - OpenAI Food Detection
=======================================
Handles food identification using OpenAI's Vision API

This module takes image crops from SAM segmentation and uses OpenAI's
GPT-4 Vision model to identify whether each crop contains food and what
specific food item it is.
"""

import openai
import base64
import io
import json
import time
from PIL import Image
from config import Config

class FoodDetector:
    """
    Food detection using OpenAI's Vision API
    
    This class handles communication with OpenAI to identify food items
    in image segments. It includes retry logic, error handling, and
    confidence scoring.
    """
    
    def __init__(self):
        """Initialize the food detector"""
        self.client = None
        self.is_api_configured = False
        
        # Set up OpenAI client
        self._setup_openai_client()
        
        # Define the prompt template for food detection
        self.food_detection_prompt = """
        Look at this image carefully. Is there a food item in this image?
        
        If YES - respond with ONLY the specific food name (e.g., "apple", "sandwich", "milk carton")
        If NO - respond with exactly "not_food"
        
        Be specific but concise. For example:
        - "red apple" not just "fruit"
        - "chocolate chip cookies" not just "cookies"
        - "whole wheat bread" not just "bread"
        
        Only respond with the food name or "not_food". No other text.
        """
    
    def _setup_openai_client(self):
        """Set up the OpenAI client with API key"""
        try:
            if not Config.OPENAI_API_KEY:
                print("⚠️ OpenAI API key not found in configuration")
                return False
            
            # Use new OpenAI v1.0+ syntax
            from openai import OpenAI
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            
            # Test the connection
            print("🔄 Testing OpenAI API connection...")
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            self.is_api_configured = True
            print("✅ OpenAI API configured successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to configure OpenAI API: {str(e)}")
            self.is_api_configured = False
            return False
    
    def is_configured(self):
        """Check if OpenAI API is properly configured"""
        return self.is_api_configured
    
    
    def detect_food(self, image_crop, max_retries=3):
        """
        Detect if an image crop contains food and identify the food item
        
        Args:
            image_crop (PIL.Image): Cropped image segment to analyze
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: Detection result with food_name, confidence, and metadata
                Returns None if no food is detected or if detection fails
        """
        if not self.is_configured():
            print("❌ OpenAI API not configured. Cannot detect food.")
            return None
        
        try:
            # Convert image to base64 for OpenAI API
            image_base64 = self._image_to_base64(image_crop)
            
            # Prepare the API request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.food_detection_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "low"  # Use low detail for faster processing
                            }
                        }
                    ]
                }
            ]
            
            # Make API request with retry logic
            for attempt in range(max_retries):
                try:
                    # Use new OpenAI v1.0+ syntax with self.client
                    response = self.client.chat.completions.create(
                        model=Config.OPENAI_MODEL,
                        messages=messages,
                        max_tokens=Config.OPENAI_MAX_TOKENS,
                        temperature=Config.OPENAI_TEMPERATURE
                    )
                    
                    # Extract the response
                    food_name = response.choices[0].message.content.strip().lower()
                    
                    # Process the response
                    result = self._process_detection_response(food_name, response)
                    
                    if result:
                        print(f"🍽️ Detected: {result['food_name']}")
                    else:
                        print("❌ No food detected in this segment")
                    
                    return result
                    
                except Exception as e:
                    # Handle all errors generically by examining error message
                    error_str = str(e).lower()
                    
                    # Check if it's a rate limit error
                    if "rate" in error_str and "limit" in error_str:
                        print(f"⏳ Rate limit hit, waiting before retry {attempt + 1}/{max_retries}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    
                    # Check if it's a general API error worth retrying
                    if any(keyword in error_str for keyword in ["api", "connection", "timeout", "server"]):
                        print(f"❌ OpenAI API error: {str(e)}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            return None
                    
                    # For other errors, don't retry
                    print(f"❌ Error in food detection: {str(e)}")
                    return None
            
            print(f"❌ Failed to get response after {max_retries} attempts")
            return None
            
        except Exception as e:
            print(f"❌ Error in food detection: {str(e)}")
            return None

    def _image_to_base64(self, image):
        """
        Convert PIL Image to base64 string for OpenAI API
        
        Args:
            image (PIL.Image): Image to convert
            
        Returns:
            str: Base64 encoded image string
        """
        # Resize image if it's too large (to save API costs)
        max_size = 512
        if max(image.width, image.height) > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = max_size / max(image.width, image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to JPEG format in memory
        buffer = io.BytesIO()
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = background
        
        image.save(buffer, format='JPEG', quality=85)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_base64
    
    def _process_detection_response(self, food_name, api_response):
        """
        Process the OpenAI API response into a structured result
        
        Args:
            food_name (str): Raw food name from API
            api_response: Full API response object
            
        Returns:
            dict or None: Processed detection result
        """
        # Clean up the food name
        food_name = food_name.strip().lower()
        
        # Check if it's actually food
        if food_name == "not_food" or "not food" in food_name or food_name == "":
            return None
        
        # Remove common prefixes/suffixes that might confuse the result
        food_name = food_name.replace("a ", "").replace("an ", "").replace("the ", "")
        food_name = food_name.replace("image of ", "").replace("picture of ", "")
        
        # Calculate confidence score based on response characteristics
        confidence = self._calculate_confidence(food_name, api_response)
        
        # Format the food name properly (capitalize first letter of each word)
        formatted_food_name = ' '.join(word.capitalize() for word in food_name.split())
        
        return {
            'food_name': formatted_food_name,
            'confidence': confidence,
            'raw_response': food_name,
            'api_usage': {
                'model': api_response.model,
                'tokens_used': api_response.usage.total_tokens if hasattr(api_response, 'usage') else 0
            }
        }
    
    def _calculate_confidence(self, food_name, api_response):
        """
        Calculate a confidence score for the food detection
        
        This is a heuristic-based approach since OpenAI doesn't provide
        explicit confidence scores for vision tasks.
        
        Args:
            food_name (str): Detected food name
            api_response: API response object
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Higher confidence for specific food names
        if len(food_name.split()) > 1:  # Multi-word descriptions are usually more specific
            confidence += 0.2
        
        # Lower confidence for vague terms
        vague_terms = ['food', 'item', 'object', 'thing', 'stuff']
        if any(term in food_name for term in vague_terms):
            confidence -= 0.3
        
        # Higher confidence for common food items
        common_foods = [
            'apple', 'banana', 'orange', 'bread', 'milk', 'cheese', 'egg',
            'chicken', 'beef', 'fish', 'rice', 'pasta', 'tomato', 'lettuce',
            'carrot', 'potato', 'onion', 'garlic', 'butter', 'yogurt'
        ]
        if any(food in food_name for food in common_foods):
            confidence += 0.15
        
        # Ensure confidence is between 0.0 and 1.0
        confidence = max(0.0, min(1.0, confidence))
        
        return round(confidence, 2)
    
    def batch_detect_food(self, image_crops, progress_callback=None):
        """
        Detect food in multiple image crops efficiently
        
        Args:
            image_crops (list): List of PIL Images to process
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            list: List of detection results for each crop
        """
        results = []
        
        for i, crop in enumerate(image_crops):
            if progress_callback:
                progress_callback(i, len(image_crops))
            
            result = self.detect_food(crop)
            results.append(result)
            
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        return results