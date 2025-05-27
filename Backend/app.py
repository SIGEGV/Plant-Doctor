from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import os
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Image size for SVM + MobileNet feature extractor models
IMG_SIZE = (224, 224)


PLANT_CONFIGS = {
    'potato': {
        'model_path': "/home/smurfy/Desktop/Plant_Disease_Detection/MAJOR_PROJECT/POTATO/MODELS/MobileNetV2/SVM/MobileNetV2_SVM.joblib",
        'class_names': [
            "Bacterial Wilt",
            "Early Blight",
            "Healthy",
            "Late Blight",
            "Leaf Roll Virus",
            "Mosaic Virus",
            "Nematode",
            "Pests",
            "Phytophthora"
        ]
    },
    'bean': {
        'model_path': "/home/smurfy/Desktop/Plant_Disease_Detection/MAJOR_PROJECT/BEAN/MODELS/MobileNetV2/SVM/MobileNetV2_SVM.joblib",
        'class_names': [
            "Angular Leaf Spot",
            "Bean Rust",
            "Healthy"
        ]
    }
}
# Tomato model and classes (Keras model)
TOMATO_MODEL_PATH = "/home/smurfy/Desktop/Plant_Disease_Detection/MAJOR_PROJECT/TOMATO/MODELS/NEW.h5"
tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)

TOMATO_CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus"
]


# Load MobileNetV2 feature extractor (shared)
feature_extractor = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
feature_extractor.trainable = False

# Load all SVM models at startup
models = {}
for plant, config in PLANT_CONFIGS.items():
    try:
        if os.path.exists(config['model_path']):
            models[plant] = joblib.load(config['model_path'])
            print(f"Loaded model for {plant}")
        else:
            print(f"Model file not found for {plant}: {config['model_path']}")
    except Exception as e:
        print(f"Error loading model for {plant}: {str(e)}")



def prepare_image(image_bytes):
    """Prepare image for MobileNetV2 + SVM pipeline (224x224)."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)  # (1, 224, 224, 3)

def prepare_tomato_image(image_bytes):
    """Prepare image for tomato Keras model (64x64)."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/api/<plant>/predict", methods=["POST"])
def predict_disease(plant):
    """Predict disease for potato, bean, etc. using SVM + MobileNet features."""
    if plant not in PLANT_CONFIGS:
        return jsonify({'error': f'Plant {plant} not supported'}), 400
    
    if plant not in models:
        return jsonify({'error': f'Model for {plant} not available'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img = prepare_image(file.read())
        features = feature_extractor(img, training=False).numpy()
        prediction = models[plant].predict(features)
        predicted_class = PLANT_CONFIGS[plant]['class_names'][prediction[0]]

        response = {
            'plant': plant,
            'prediction': predicted_class,
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/tomato/predict", methods=["POST"])
def predict_tomato():
    """Predict tomato disease using Keras model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img = prepare_tomato_image(file.read())
        prediction = tomato_model.predict(img)[0]
        predicted_idx = int(np.argmax(prediction))
        
        return jsonify({
            'plant': 'tomato',
            'prediction': TOMATO_CLASS_NAMES[predicted_idx],
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        'status': 'healthy',
        'available_plants': list(models.keys()) + ['tomato']
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
