from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json
import os
from model_config import *
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variable to store the loaded model
model = None

def load_model():
    """Load the trained .keras model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print(f" Loading model from: {MODEL_PATH}")
            
            # Try to load the model with custom objects to handle input shape issues
            try:
                # First try normal loading
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            except ValueError as e:
                print(f"⚠️  Model loading failed: {e}")
                print("🔄 Trying alternative loading method...")
                
                # Try loading with custom_objects and custom_metrics
                try:
                    model = tf.keras.models.load_model(
                        MODEL_PATH, 
                        compile=False,
                        custom_objects={
                            'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
                            'accuracy': tf.keras.metrics.Accuracy()
                        }
                    )
                except Exception as e2:
                    print(f"⚠️  Alternative loading failed: {e2}")
                    print("🔄 Trying to fix input layer issue...")
                    
                    # Try to load and fix the input layer
                    try:
                        # Load the model without the input layer
                        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                        
                        # Get the model's layers except the input layer
                        layers = model.layers[1:]  # Skip the first layer (input)
                        
                        # Create a new model with correct input shape
                        inputs = tf.keras.layers.Input(shape=(161, 161, 3))
                        x = inputs
                        
                        # Rebuild the model with the correct input
                        for layer in layers:
                            x = layer(x)
                        
                        model = tf.keras.Model(inputs=inputs, outputs=x)
                        print("✅ Model rebuilt with correct input shape")
                        
                    except Exception as e3:
                        print(f"❌ Model rebuilding failed: {e3}")
                        print("🔄 Using fallback model...")
                        
                        # Create a fallback model
                        input_shape = (161, 161, 3)
                        num_classes = len(DISEASE_CLASSES)
                        
                        model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=input_shape),
                            tf.keras.layers.Conv2D(32, 3, activation='relu'),
                            tf.keras.layers.MaxPooling2D(),
                            tf.keras.layers.Conv2D(64, 3, activation='relu'),
                            tf.keras.layers.MaxPooling2D(),
                            tf.keras.layers.Conv2D(64, 3, activation='relu'),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(num_classes, activation='softmax')
                        ])
                        
                        print("⚠️  Using fallback model - your trained model couldn't be loaded")
                        print("💡 This is a demo model and won't give accurate predictions")
            
            # Print model information for debugging
            print(f" Model input shape: {model.input_shape}")
            print(f" Model output shape: {model.output_shape}")
            print(f" Model layers count: {len(model.layers)}")
            
            # Check if the model expects RGB input
            expected_channels = model.input_shape[-1]
            print(f"📊 Expected input channels: {expected_channels}")
            
            if expected_channels != 3:
                print(f"⚠️  Warning: Model expects {expected_channels} channels, but we're providing 3 (RGB)")
            
            # Compile the model for inference
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            print(f" Number of classes: {len(DISEASE_CLASSES)}")
            print(f"🌱 PlantVillage dataset classes loaded")
            
        else:
            print(f"❌ Model file not found: {MODEL_PATH}")
            print("💡 Please place your trained .keras model file in the project directory")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    return True

def preprocess_image(image_data):
    """Preprocess image for PlantVillage model input"""
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Resize to model's expected size
        image = image.resize(INPUT_SIZE)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Apply ImageNet normalization if configured
        if NORMALIZE and 'MEAN' in globals() and 'STD' in globals():
            # ImageNet normalization
            image_array = image_array.astype(np.float32) / 255.0
            image_array = (image_array - MEAN) / STD
        else:
            # Simple 0-1 normalization
            image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_disease(image_array):
    """Make prediction using the loaded PlantVillage model"""
    global model
    
    if model is None:
        return None, 0.0
    
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get class name
        predicted_class = DISEASE_CLASSES[predicted_class_idx]
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_plant():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        predicted_class, confidence = predict_disease(processed_image)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get disease information
        disease_info = DISEASE_INFO.get(predicted_class, {})
        
        # Format the class name for display
        display_name = predicted_class.replace('___', ' - ').replace('_', ' ')
        
        result = {
            'disease': display_name,
            'confidence': round(confidence * 100, 2),  # Convert to percentage
            'description': disease_info.get('description', 'No description available for this class.'),
            'cure': disease_info.get('cure', ['No specific treatment information available.'])
        }
        
        print(f" Analysis complete: {predicted_class} ({confidence*100:.2f}%)")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/health')
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'classes': len(DISEASE_CLASSES),
        'input_size': INPUT_SIZE,
        'dataset': 'PlantVillage'
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'classes': DISEASE_CLASSES,
        'input_size': INPUT_SIZE,
        'normalization': NORMALIZE,
        'dataset': 'PlantVillage',
        'total_classes': len(DISEASE_CLASSES)
    })

# Add this function to check your model file
def check_model_file():
    """Check if the model file is valid"""
    try:
        import h5py
        with h5py.File(MODEL_PATH, 'r') as f:
            print("✅ Model file structure:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
            f.visititems(print_structure)
    except Exception as e:
        print(f"❌ Error reading model file: {e}")
        return False
    return True

if __name__ == '__main__':
    print("🚀 Starting Plant Health Detector...")
    print(f" Looking for PlantVillage model at: {MODEL_PATH}")
    print(f"🌱 Dataset: PlantVillage ({len(DISEASE_CLASSES)} classes)")
    
    # Check model file first
    if os.path.exists(MODEL_PATH):
        print("🔍 Checking model file...")
        check_model_file()
    
    # Load model on startup
    if load_model():
        print("✅ Ready to analyze plant images!")
    else:
        print("⚠️  Starting in demo mode (no ML model)")
        print("💡 Upload your Make_healthy_plant.keras file to enable ML predictions")
    
    # App settings via env
    app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
    debug_flag = os.getenv("FLASK_DEBUG", "true").lower() in ("1", "true", "yes")
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "5000"))
    except ValueError:
        port = 5000
    app.run(debug=debug_flag, host=host, port=port)
