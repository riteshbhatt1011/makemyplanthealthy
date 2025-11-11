#!/usr/bin/env python3
"""
Test script to verify the Plant Health Detector model and configuration
"""

import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image

# Import configuration
from model_config import *

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🧪 Testing Model Loading...")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"📊 Expected input size: {INPUT_SIZE}")
    print(f"🎨 Expected channels: {CHANNELS}")
    print(f"🏷️  Number of classes: {len(DISEASE_CLASSES)}")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    print(f"✅ Model file found: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
    
    try:
        # Load the model
        print("🔄 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Print model information
        print(f"✅ Model loaded successfully!")
        print(f"📊 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
        print(f"🏗️  Number of layers: {len(model.layers)}")
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_preprocessing():
    """Test image preprocessing"""
    print("\n🧪 Testing Image Preprocessing...")
    
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        # Resize to expected input size
        resized_image = pil_image.resize(INPUT_SIZE)
        print(f"✅ Image resized to: {resized_image.size}")
        
        # Convert to RGB
        rgb_image = resized_image.convert('RGB')
        print(f"✅ Image converted to RGB: {rgb_image.mode}")
        
        # Convert to numpy array
        image_array = np.array(rgb_image)
        print(f"✅ Numpy array shape: {image_array.shape}")
        
        # Normalize
        if NORMALIZE and 'MEAN' in globals() and 'STD' in globals():
            image_array = image_array.astype(np.float32) / 255.0
            image_array = (image_array - MEAN) / STD
            print("✅ ImageNet normalization applied")
        else:
            image_array = image_array.astype(np.float32) / 255.0
            print("✅ Simple normalization applied")
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        print(f"✅ Final input shape: {image_array.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return False

def test_prediction():
    """Test model prediction with dummy data"""
    print("\n🧪 Testing Model Prediction...")
    
    try:
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Create dummy input
        dummy_input = np.random.random((1, INPUT_SIZE[0], INPUT_SIZE[1], 3)).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        print(f"✅ Prediction successful!")
        print(f"🏷️  Predicted class index: {predicted_class_idx}")
        print(f"🏷️  Predicted class: {DISEASE_CLASSES[predicted_class_idx]}")
        print(f"📊 Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Plant Health Detector - Model Test Suite")
    print("=" * 50)
    
    # Test 1: Model Loading
    model_ok = test_model_loading()
    
    # Test 2: Preprocessing
    preprocess_ok = test_preprocessing()
    
    # Test 3: Prediction
    prediction_ok = False
    if model_ok:
        prediction_ok = test_prediction()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"✅ Model Loading: {'PASS' if model_ok else 'FAIL'}")
    print(f"✅ Preprocessing: {'PASS' if preprocess_ok else 'FAIL'}")
    print(f"✅ Prediction: {'PASS' if prediction_ok else 'FAIL'}")
    
    if model_ok and preprocess_ok and prediction_ok:
        print("\n🎉 All tests passed! Your model is ready for predictions!")
        print("💡 You can now run: python app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
    
    return model_ok and preprocess_ok and prediction_ok

if __name__ == "__main__":
    main()

