#!/usr/bin/env python3
"""
Script to fix the model input layer issue
"""

import tensorflow as tf
import numpy as np
from model_config import *

def fix_model():
    """Fix the model input layer issue"""
    print("🔧 Fixing model input layer...")
    
    try:
        # Load the original model
        print("📥 Loading original model...")
        original_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        print(f"📊 Original model input shape: {original_model.input_shape}")
        print(f"📊 Original model output shape: {original_model.output_shape}")
        
        # Get all layers except the input layer
        layers = original_model.layers[1:]  # Skip the first layer (input)
        
        # Create new input layer with correct shape
        new_input = tf.keras.layers.Input(shape=(161, 161, 3), name='new_input')
        
        # Connect the layers
        x = new_input
        for layer in layers:
            x = layer(x)
        
        # Create new model
        fixed_model = tf.keras.Model(inputs=new_input, outputs=x)
        
        print(f"✅ Fixed model input shape: {fixed_model.input_shape}")
        print(f"✅ Fixed model output shape: {fixed_model.output_shape}")
        
        # Test the fixed model
        print("🧪 Testing fixed model...")
        test_input = np.random.random((1, 161, 161, 3)).astype(np.float32)
        test_output = fixed_model.predict(test_input, verbose=0)
        print(f"✅ Test prediction shape: {test_output.shape}")
        
        # Save the fixed model
        fixed_model_path = "Make_healthy_plant_fixed.keras"
        fixed_model.save(fixed_model_path)
        print(f"💾 Fixed model saved as: {fixed_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing model: {e}")
        return False

if __name__ == "__main__":
    fix_model()
