# Simple Food Detection System

A straightforward food detection application using SAM (Segment Anything Model) + OpenAI Vision API that automatically builds a labeled dataset for future local model training.

## Why This Approach?

Unlike complex systems with multiple microservices, Docker containers, and React frontends, this project follows the KISS principle (Keep It Simple, Stupid). Everything runs in a single Streamlit app with clearly separated, focused modules.

**Key Benefits:**
- ✅ **One-file deployment** - Just run `streamlit run main.py`
- ✅ **No complex setup** - Works with basic Python environment
- ✅ **Clear separation of concerns** - Each module has one job
- ✅ **Builds training data automatically** - Every detection becomes a labeled example
- ✅ **Easy to understand and modify** - No hidden complexity
- ✅ **Cost-effective path to local model** - Collect data with API, then train local model

## Project Structure

```
food-detector/
├── main.py                    # 🎯 Main Streamlit app (START HERE)
├── config.py                  # ⚙️ Configuration settings
├── requirements.txt           # 📦 Python dependencies
├── .env.example              # 🔧 Environment template
├── 
├── src/                      # 🧠 Core business logic
│   ├── segmentation.py       # 📐 SAM image segmentation
│   ├── detection.py          # 🤖 OpenAI food identification
│   └── database_manager.py   # 💾 Data storage
│
├── utils/                    # 🛠️ Helper utilities
│   ├── image_utils.py        # 🖼️ Image processing
│   └── visualization.py     # 🎨 Drawing and display
│
├── models/                   # 🎭 AI model weights
│   └── sam_vit_h_4b8939.pth # (download this)
│
├── data/                     # 📊 Generated data
│   ├── uploads/              # Uploaded images
│   ├── crops/                # Segmented food items  
│   ├── results/              # Annotated images
│   └── database.db           # SQLite database
│
└── scripts/                  # 🚀 Setup helpers
    ├── setup.py              # Initial project setup
    └── download_sam.py       # Download SAM model
```

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
git clone <your-repo>
cd food-detector
pip install -r requirements.txt
```

### Step 2: Setup Project
```bash
python scripts/setup.py
python scripts/download_sam.py
```

### Step 3: Configure & Run
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
streamlit run main.py
```

## How It Works

### The Simple Pipeline
```
📸 Upload Image → 🔍 SAM Segments → 🤖 OpenAI Identifies → 💾 Store Data → 🎯 Train Local Model
```

1. **Upload**: User uploads a fridge/food image
2. **Segment**: SAM automatically finds objects in the image
3. **Identify**: OpenAI Vision API identifies food items in each segment
4. **Store**: Results saved to database with bounding boxes and labels
5. **Display**: Annotated image shown with detected food items
6. **Build Dataset**: Each detection adds to your training data

### The Smart Part: Automatic Dataset Building

Every time you process an image, the system automatically stores:
- ✅ **Original images** with metadata
- ✅ **Cropped food segments** as individual files
- ✅ **Bounding box coordinates** in standard format
- ✅ **Food labels** from OpenAI
- ✅ **Confidence scores** and processing metadata

This builds a high-quality labeled dataset that you can use to train a local model, eventually eliminating OpenAI API costs.

## Key Design Principles

### 1. Simplicity Over Complexity
- **Single Streamlit app** instead of separate frontend/backend
- **SQLite database** instead of complex database setup
- **File-based storage** instead of cloud storage requirements
- **Direct imports** instead of microservice communication

### 2. Separation of Concerns
- **main.py**: UI and user interaction only
- **src/**: Core business logic (segmentation, detection, data)
- **utils/**: Reusable helper functions
- **config.py**: All settings in one place

### 3. Progressive Enhancement
- Start with API-based detection (fast to deploy)
- Collect labeled data automatically
- Train local model when you have enough data
- Replace API calls with local inference (save costs)

### 4. Defensive Programming
- Graceful error handling throughout
- Fallbacks when components fail
- Clear error messages for debugging
- Robust file and data handling

## Configuration

All settings are centralized in `config.py` and can be overridden with environment variables:

```bash
# Essential settings
OPENAI_API_KEY=your_key_here
SAM_MODEL_PATH=models/sam_vit_h_4b8939.pth

# Processing tuning
MAX_IMAGE_SIZE=1024          # Resize large images for speed
MIN_SEGMENT_AREA=1000        # Filter tiny segments
MAX_SEGMENTS_PER_IMAGE=20    # Limit processing per image

# Quality settings
SAM_PRED_IOU_THRESH=0.7      # SAM prediction threshold
OPENAI_TEMPERATURE=0.1       # Lower = more consistent results
```

## Usage Examples

### Basic Detection
```python
# Just run the Streamlit app
streamlit run main.py

# Upload image, click "Detect Food Items"
# See results immediately with bounding boxes and labels
```

### Programmatic Usage
```python
from src.segmentation import SAMSegmenter
from src.detection import FoodDetector
from utils.image_utils import prepare_image_for_detection

# Initialize components
segmenter = SAMSegmenter()
detector = FoodDetector()

# Process an image
image, path = prepare_image_for_detection(uploaded_file)
segments = segmenter.segment_image(image)

# Detect food in each segment
for segment in segments:
    result = detector.detect_food(segment['crop'])
    if result:
        print(f"Found: {result['food_name']} (confidence: {result['confidence']})")
```

### Data Export for Training
```python
from src.database_manager import DatabaseManager

db = DatabaseManager()
training_data = db.export_training_data(format='yolo')
print(f"Ready to train on {training_data['total_records']} examples!")
```

## Troubleshooting

### SAM Model Issues
```bash
# If SAM model download fails:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth models/
```

### OpenAI API Issues
- ✅ Check your API key in `.env`
- ✅ Verify you have credits in your OpenAI account
- ✅ Check rate limits (we include retry logic)

### Performance Issues
- ✅ Reduce `MAX_SEGMENTS_PER_IMAGE` for faster processing
- ✅ Lower `MAX_IMAGE_SIZE` if memory is limited
- ✅ Use CPU mode if you don't have GPU (`DEVICE=cpu`)

### Database Issues
- ✅ Database is created automatically
- ✅ Delete `data/database.db` to reset all data
- ✅ Check file permissions on data directory




### This Simple Approach:
- ✅ Single Python file to run
- ✅ No containers or complex deployment
- ✅ Clear, readable code structure
- ✅ Easy to modify and extend
- ✅ Focus on solving the actual problem
- ✅ Works perfectly for the use case

## Future Enhancements

As your dataset grows, you can easily add:

1. **Local Model Training**
   ```python
   # Use your collected data to train a YOLO or similar model
   python train_local_model.py --data data/exports/yolo_format
   ```

2. **Batch Processing**
   ```python
   # Process multiple images at once
   python batch_process.py --input_dir /path/to/images
   ```

3. **Model Evaluation**
   ```python
   # Compare OpenAI vs local model performance
   python evaluate_models.py --test_set data/test_images
   ```

4. **API Endpoint** (if needed later)
   ```python
   # Add a simple Flask/FastAPI wrapper around the core logic
   # But keep the core logic in the same modules!
   ```

## Contributing

The beauty of this architecture is that it's easy to contribute to:

1. **Adding new detection methods**: Create new modules in `src/`
2. **Improving visualization**: Enhance `utils/visualization.py`
3. **Adding export formats**: Extend `database_manager.py`
4. **Performance optimization**: Tune parameters in `config.py`

Each module has a single responsibility, making changes predictable and safe.

## License

MIT License - Use this however you want!

---

**Remember**: The best code is code that solves your problem simply and reliably. This project prioritizes clarity, maintainability, and getting results over architectural complexity.

Start simple, iterate based on real needs, and avoid premature optimization!
