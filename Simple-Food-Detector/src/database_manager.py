"""
src/database_manager.py - Simple Database Operations
===================================================
Handles all database operations for storing food detection results

This module manages a simple SQLite database that stores:
1. Original images and their metadata
2. Detection results (what food was found where)
3. Bounding box coordinates for future model training
4. Confidence scores and other metadata

The beauty of this approach is that it automatically builds your training
dataset as you use the app, giving you labeled data for training a local
model later to replace the OpenAI API.
"""

import sqlite3
import json
import os
from datetime import datetime
from config import Config

class DatabaseManager:
    """
    Simple database manager for food detection results
    
    This class handles all database operations using SQLite, which is perfect
    for this application because:
    - No complex setup required
    - File-based database that's easy to backup
    - Good performance for moderate amounts of data
    - Built into Python, no additional dependencies
    """
    
    def __init__(self):
        """Initialize the database manager and create tables if needed"""
        self.db_path = Config.DATABASE_PATH
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create tables if they don't exist
        self._create_tables()
        
        print(f"📊 Database initialized at {self.db_path}")
    
    def _create_tables(self):
        """
        Create the database tables if they don't exist
        
        We use a simple schema with just two main tables:
        1. images - stores metadata about uploaded images
        2. detections - stores individual food detections with bounding boxes
        
        This design makes it easy to export data for training later.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Images table - stores information about each uploaded image
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        original_path TEXT NOT NULL,
                        upload_timestamp DATETIME NOT NULL,
                        image_width INTEGER,
                        image_height INTEGER,
                        file_size INTEGER,
                        processing_status TEXT DEFAULT 'completed',
                        total_detections INTEGER DEFAULT 0,
                        notes TEXT
                    )
                ''')
                
                # Detections table - stores individual food item detections
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        food_name TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        bbox_x INTEGER NOT NULL,
                        bbox_y INTEGER NOT NULL,
                        bbox_width INTEGER NOT NULL,
                        bbox_height INTEGER NOT NULL,
                        detection_timestamp DATETIME NOT NULL,
                        api_tokens_used INTEGER DEFAULT 0,
                        raw_api_response TEXT,
                        crop_saved_path TEXT,
                        verified BOOLEAN DEFAULT 0,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(upload_timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_food_name ON detections(food_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(detection_timestamp)')
                
                conn.commit()
                print("✅ Database tables created/verified successfully")
                
        except Exception as e:
            print(f"❌ Error creating database tables: {str(e)}")
            raise
    
    def save_detection_result(self, filename, image_path, detections):
        """
        Save a complete detection result to the database
        
        This is the main method you'll use after processing an image.
        It stores both the image metadata and all the food detections,
        building your training dataset automatically.
        
        Args:
            filename (str): Original filename of the uploaded image
            image_path (str): Path where the image is stored
            detections (list): List of detection results from FoodDetector
            
        Returns:
            int: ID of the saved image record
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get image dimensions and file size
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
                
                file_size = os.path.getsize(image_path)
                
                # Insert image record
                cursor.execute('''
                    INSERT INTO images (
                        filename, original_path, upload_timestamp,
                        image_width, image_height, file_size, total_detections
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    filename,
                    image_path,
                    datetime.now().isoformat(),
                    width,
                    height,
                    file_size,
                    len(detections)
                ))
                
                image_id = cursor.lastrowid
                
                # Insert detection records
                for detection in detections:
                    bbox = detection['bbox']
                    
                    # Save the crop image if it exists
                    crop_path = None
                    if 'crop' in detection and detection['crop']:
                        crop_path = self._save_crop_image(detection['crop'], image_id, detection['food_name'])
                    
                    cursor.execute('''
                        INSERT INTO detections (
                            image_id, food_name, confidence,
                            bbox_x, bbox_y, bbox_width, bbox_height,
                            detection_timestamp, api_tokens_used,
                            raw_api_response, crop_saved_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        image_id,
                        detection['food_name'],
                        detection['confidence'],
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        datetime.now().isoformat(),
                        detection.get('api_usage', {}).get('tokens_used', 0),
                        json.dumps(detection.get('raw_response', '')),
                        crop_path
                    ))
                
                conn.commit()
                print(f"💾 Saved detection result: {len(detections)} items from {filename}")
                return image_id
                
        except Exception as e:
            print(f"❌ Error saving detection result: {str(e)}")
            raise
    
    def _save_crop_image(self, crop_image, image_id, food_name):
        """
        Save a cropped image segment to disk
        
        This creates individual image files for each detected food item,
        which is perfect for training data preparation.
        """
        try:
            # Create crops directory if it doesn't exist
            crops_dir = Config.CROPS_DIR
            os.makedirs(crops_dir, exist_ok=True)
            
            # Generate filename for the crop
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
            safe_food_name = "".join(c for c in food_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_food_name = safe_food_name.replace(' ', '_')
            
            crop_filename = f"img{image_id:06d}_{timestamp}_{safe_food_name}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            
            # Save the crop
            crop_image.save(crop_path, 'JPEG', quality=90)
            
            return crop_path
            
        except Exception as e:
            print(f"⚠️ Error saving crop image: {str(e)}")
            return None
    
    def get_recent_results(self, limit=10):
        """
        Get recent detection results for display in the UI
        
        Args:
            limit (int): Maximum number of results to return
            
        Returns:
            list: Recent detection results with image and detection info
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent images with their detections
                cursor.execute('''
                    SELECT 
                        i.filename,
                        i.upload_timestamp,
                        i.total_detections,
                        GROUP_CONCAT(d.food_name || ":" || d.confidence, "|") as detections_data
                    FROM images i
                    LEFT JOIN detections d ON i.id = d.image_id
                    GROUP BY i.id
                    ORDER BY i.upload_timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    filename, timestamp, total_detections, detections_data = row
                    
                    # Parse detections data
                    detections = []
                    if detections_data:
                        for detection_str in detections_data.split('|'):
                            if ':' in detection_str:
                                food_name, confidence = detection_str.split(':', 1)
                                detections.append({
                                    'food_name': food_name,
                                    'confidence': float(confidence)
                                })
                    
                    results.append({
                        'filename': filename,
                        'timestamp': timestamp,
                        'total_detections': total_detections or 0,
                        'detections': detections
                    })
                
                return results
                
        except Exception as e:
            print(f"❌ Error getting recent results: {str(e)}")
            return []
    
    def get_total_images(self):
        """Get total number of processed images"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM images')
                return cursor.fetchone()[0]
        except:
            return 0
    
    def get_total_detections(self):
        """Get total number of food detections"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM detections')
                return cursor.fetchone()[0]
        except:
            return 0
    
    def get_food_statistics(self):
        """
        Get statistics about detected food items
        
        This is useful for understanding what foods are most commonly
        detected, which can help when planning your local model training.
        
        Returns:
            list: Food items with detection counts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        food_name,
                        COUNT(*) as detection_count,
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence
                    FROM detections
                    GROUP BY food_name
                    ORDER BY detection_count DESC
                ''')
                
                return [
                    {
                        'food_name': row[0],
                        'count': row[1],
                        'avg_confidence': round(row[2], 2),
                        'min_confidence': round(row[3], 2),
                        'max_confidence': round(row[4], 2)
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            print(f"❌ Error getting food statistics: {str(e)}")
            return []
    
    def export_training_data(self, output_format='yolo'):
        """
        Export detection data in format suitable for training a local model
        
        This is the key method for your goal of eventually replacing the
        OpenAI API with a local model. It exports your accumulated data
        in standard formats used for computer vision model training.
        
        Args:
            output_format (str): Format to export ('yolo', 'coco', 'csv')
            
        Returns:
            dict: Export results with file paths and statistics
        """
        try:
            # This is a placeholder for the export functionality
            # You can implement different export formats as needed
            
            print(f"🔄 Exporting training data in {output_format} format...")
            
            # Get all detection data
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        i.filename, i.original_path, i.image_width, i.image_height,
                        d.food_name, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                        d.confidence, d.crop_saved_path
                    FROM images i
                    JOIN detections d ON i.id = d.image_id
                    ORDER BY i.id, d.id
                ''')
                
                export_data = cursor.fetchall()
            
            print(f"📊 Found {len(export_data)} detection records ready for export")
            
            # Return export summary (you can implement actual export logic here)
            return {
                'total_records': len(export_data),
                'export_format': output_format,
                'ready_for_training': len(export_data) > 100,  # Arbitrary threshold
                'message': f"Ready to export {len(export_data)} labeled examples"
            }
            
        except Exception as e:
            print(f"❌ Error exporting training data: {str(e)}")
            return None