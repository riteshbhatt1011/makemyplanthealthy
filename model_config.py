import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Configuration for PlantVillage dataset plant disease detection model
# Based on: https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/b4e3a32f-c0bd-4060-81e9-6144231f2520/file_downloaded

# Model file path - configurable via env
# Default kept for local usage; override with MODEL_PATH in .env
MODEL_PATH = os.getenv("MODEL_PATH", "Make_healthy_plant.keras")

# Image preprocessing settings - configurable via env
# INPUT_SIZE supports "WxH" format like "161x161"
_input_size_env = os.getenv("INPUT_SIZE", "161x161")
try:
    _w, _h = _input_size_env.lower().split("x")
    INPUT_SIZE = (int(_w), int(_h))
except Exception:
    INPUT_SIZE = (161, 161)

CHANNELS = 3  # RGB images

# PlantVillage Dataset Classes (39 classes in exact order)
# ⚠️ You need to verify these are the correct 39 classes from your training
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___healthy',
    'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___healthy',
    'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'  # Added the missing 39th class
]

# Preprocessing settings
NORMALIZE = True  # Normalize pixel values (0-1)
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean values
STD = [0.229, 0.224, 0.225]   # ImageNet std values

# Prediction settings
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to show results

# Enhanced disease information database for PlantVillage classes
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'Apple scab is a serious disease of apples caused by the fungus Venturia inaequalis. It causes dark, scabby lesions on leaves and fruit.',
        'cure': [
            'Apply fungicides during growing season',
            'Remove and destroy fallen leaves and fruit',
            'Prune to improve air circulation',
            'Plant resistant varieties',
            'Maintain proper tree spacing'
        ]
    },
    'Apple___Black_rot': {
        'description': 'Black rot is a fungal disease that causes fruit rot, leaf spots, and cankers on apple trees.',
        'cure': [
            'Remove and destroy infected plant parts',
            'Apply fungicides during bloom',
            'Prune out cankers in winter',
            'Improve air circulation',
            'Control insects that spread the disease'
        ]
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar apple rust is caused by a fungus that requires both apple trees and cedar trees to complete its life cycle.',
        'cure': [
            'Remove cedar trees within 2 miles if possible',
            'Apply fungicides during growing season',
            'Plant resistant apple varieties',
            'Remove galls from cedar trees',
            'Maintain tree health'
        ]
    },
    'Apple___healthy': {
        'description': 'Your apple tree appears to be healthy with no signs of disease.',
        'cure': [
            'Continue current care routine',
            'Monitor for any changes',
            'Maintain proper watering and fertilization',
            'Prune regularly for good structure',
            'Apply dormant oil in winter'
        ]
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot is caused by Xanthomonas bacteria, creating small, dark lesions on leaves, stems, and fruit.',
        'cure': [
            'Remove and destroy infected plants',
            'Use disease-free seed',
            'Avoid overhead watering',
            'Apply copper-based bactericides',
            'Rotate crops annually'
        ]
    },
    'Tomato___Early_blight': {
        'description': 'Early blight is a common fungal disease causing dark brown spots with concentric rings on leaves.',
        'cure': [
            'Remove infected leaves',
            'Improve air circulation',
            'Apply fungicides if necessary',
            'Avoid overhead watering',
            'Maintain proper plant spacing'
        ]
    },
    'Tomato___Late_blight': {
        'description': 'Late blight is a serious disease that can quickly kill plants, characterized by water-soaked lesions.',
        'cure': [
            'Remove and destroy all infected plants',
            'Apply copper-based fungicide',
            'Improve air circulation',
            'Avoid overhead watering',
            'Plant resistant varieties'
        ]
    },
    'Tomato___healthy': {
        'description': 'Your tomato plant appears to be healthy with no signs of disease.',
        'cure': [
            'Continue current care routine',
            'Monitor for any changes',
            'Maintain proper watering',
            'Ensure adequate sunlight',
            'Regular fertilization'
        ]
    },
    'Potato___Early_blight': {
        'description': 'Early blight causes dark brown spots with concentric rings on potato leaves, reducing yield.',
        'cure': [
            'Remove infected leaves',
            'Improve air circulation',
            'Apply fungicides if necessary',
            'Avoid overhead watering',
            'Maintain proper plant spacing'
        ]
    },
    'Potato___Late_blight': {
        'description': 'Late blight is a devastating disease that can destroy entire potato crops quickly.',
        'cure': [
            'Remove and destroy infected plants immediately',
            'Apply copper-based fungicides',
            'Improve air circulation',
            'Avoid overhead watering',
            'Plant resistant varieties'
        ]
    },
    'Potato___healthy': {
        'description': 'Your potato plant appears to be healthy with no signs of disease.',
        'cure': [
            'Continue current care routine',
            'Monitor for any changes',
            'Maintain proper watering',
            'Hill soil around plants',
            'Regular fertilization'
        ]
    }
}

# Add more disease info for other classes as needed
# You can expand this database with information for all 38 classes
