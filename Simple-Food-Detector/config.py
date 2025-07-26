"""
config.py - Configuration Settings
==================================
Central configuration for the food detection application
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration settings"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Model Paths
    SAM_MODEL_PATH = os.getenv('SAM_MODEL_PATH', 'models/sam_vit_h_4b8939.pth')
    SAM_MODEL_TYPE = os.getenv('SAM_MODEL_TYPE', 'vit_h')
    
    # Database
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/database.db')
    
    # Processing Settings
    # MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '1024'))  # Max width/height in pixels
    MAX_IMAGE_SIZE = 512
    MIN_SEGMENT_AREA = int(os.getenv('MIN_SEGMENT_AREA', '1000'))  # Minimum area for segments
    # MAX_SEGMENTS_PER_IMAGE = int(os.getenv('MAX_SEGMENTS_PER_IMAGE', '20'))
    MAX_SEGMENTS_PER_IMAGE = 5
    
    # OpenAI Settings
    # OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-vision-preview')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '50'))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
    
    # File Storage
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'data/uploads')
    CROPS_DIR = os.getenv('CROPS_DIR', 'data/crops')
    RESULTS_DIR = os.getenv('RESULTS_DIR', 'data/results')
    
    # SAM Configuration
    # SAM_POINTS_PER_SIDE = int(os.getenv('SAM_POINTS_PER_SIDE', '16'))
    SAM_POINTS_PER_SIDE = 8
    SAM_PRED_IOU_THRESH = float(os.getenv('SAM_PRED_IOU_THRESH', '0.7'))
    SAM_STABILITY_SCORE_THRESH = float(os.getenv('SAM_STABILITY_SCORE_THRESH', '0.8'))
    
    # Device Configuration
    DEVICE = os.getenv('DEVICE', 'cpu')  # 'cpu' or 'cuda'
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        issues = []
        
        # Check if SAM model file exists
        if not os.path.exists(cls.SAM_MODEL_PATH):
            issues.append(f"SAM model not found at {cls.SAM_MODEL_PATH}")
        
        # Check OpenAI API key
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY not set")
        
        # Create directories if they don't exist
        for directory in [cls.UPLOAD_DIR, cls.CROPS_DIR, cls.RESULTS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        return issues
    
    @classmethod
    def get_sam_config(cls):
        """Get SAM model configuration as a dictionary"""
        return {
            'points_per_side': cls.SAM_POINTS_PER_SIDE,
            'pred_iou_thresh': cls.SAM_PRED_IOU_THRESH,
            'stability_score_thresh': cls.SAM_STABILITY_SCORE_THRESH,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': cls.MIN_SEGMENT_AREA
        }
    
    @classmethod
    def get_openai_config(cls):
        """Get OpenAI configuration as a dictionary"""
        return {
            'model': cls.OPENAI_MODEL,
            'max_tokens': cls.OPENAI_MAX_TOKENS,
            'temperature': cls.OPENAI_TEMPERATURE
        }

# Create a default config instance
config = Config()

# Validate configuration on import
config_issues = config.validate_config()
if config_issues:
    print("⚠️ Configuration Issues Found:")
    for issue in config_issues:
        print(f"  - {issue}")
    print("\nPlease check your .env file and model paths.")