"""
Healthcare Risk Prediction API
Flask backend server for serving ML model predictions
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load trained models
print("Loading ML models...")
try:
    diabetes_model = joblib.load('diabetes_model.pkl')
    heart_model = joblib.load('heart_disease_model.pkl')
    print("Models loaded successfully")
except FileNotFoundError:
    print("ERROR: Model files not found. Please run 'python train_models.py' first.")
    diabetes_model = None
    heart_model = None

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'diabetes_model_loaded': diabetes_model is not None,
        'heart_model_loaded': heart_model is not None
    })

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    print("Received diabetes prediction request")
    try:
        if diabetes_model is None:
            return jsonify({'error': 'Diabetes model not loaded'}), 500
        
        data = request.json
        
        # Extract features in correct order
        features = [
            float(data.get('gender', 0)),
            float(data.get('age', 0)),
            float(data.get('hypertension', 0)),
            float(data.get('heart_disease', 0)),
            float(data.get('smoking_history', 0)),
            float(data.get('bmi', 0)),
            float(data.get('HbA1c_level', 0)),
            float(data.get('blood_glucose_level', 0))
        ]
        
        # Make prediction
        features_array = np.array([features])
        prediction = diabetes_model.predict(features_array)[0]
        probability = diabetes_model.predict_proba(features_array)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'risk_probability': float(probability[1]),  # Probability of diabetes
            'risk_percentage': float(probability[1] * 100),
            'risk_level': 'High' if probability[1] > 0.5 else 'Low'
        })
    
    except Exception as e:
        print(f"Diabetes Prediction Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict/heart-disease', methods=['POST'])
def predict_heart_disease():
    print("Received heart disease prediction request")
    try:
        if heart_model is None:
            return jsonify({'error': 'Heart disease model not loaded'}), 500
        
        data = request.json
        
        # Extract features in correct order (matching dataset columns)
        features = [
            float(data.get('Chest_Pain', 0)),
            float(data.get('Shortness_of_Breath', 0)),
            float(data.get('Fatigue', 0)),
            float(data.get('Palpitations', 0)),
            float(data.get('Dizziness', 0)),
            float(data.get('Swelling', 0)),
            float(data.get('Pain_Arms_Jaw_Back', 0)),
            float(data.get('Cold_Sweats_Nausea', 0)),
            float(data.get('High_BP', 0)),
            float(data.get('High_Cholesterol', 0)),
            float(data.get('Diabetes', 0)),
            float(data.get('Smoking', 0)),
            float(data.get('Obesity', 0)),
            float(data.get('Sedentary_Lifestyle', 0)),
            float(data.get('Family_History', 0)),
            float(data.get('Chronic_Stress', 0)),
            float(data.get('Gender', 0)),
            float(data.get('Age', 0))
        ]
        
        # Make prediction
        features_array = np.array([features])
        prediction = heart_model.predict(features_array)[0]
        probability = heart_model.predict_proba(features_array)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'risk_probability': float(probability[1]),  # Probability of heart disease
            'risk_percentage': float(probability[1] * 100),
            'risk_level': 'High' if probability[1] > 0.5 else 'Low'
        })
    
    except Exception as e:
        print(f"Heart Disease Prediction Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict/combined', methods=['POST'])
def predict_combined():
    """
    Get both diabetes and heart disease predictions
    """
    print("Received combined prediction request")
    try:
        data = request.json
        
        # This approach is safer than calling other view functions
        # Extract features for diabetes
        d_features = [
            float(data.get('gender', 0)),
            float(data.get('age', 0)),
            float(data.get('hypertension', 0)),
            float(data.get('heart_disease', 0)),
            float(data.get('smoking_history', 0)),
            float(data.get('bmi', 0)),
            float(data.get('HbA1c_level', 5.7)),
            float(data.get('blood_glucose_level', 100))
        ]
        d_prob = diabetes_model.predict_proba(np.array([d_features]))[0][1]
        
        # Extract features for heart disease
        h_features = [
            float(data.get('Chest_Pain', 0)),
            float(data.get('Shortness_of_Breath', 0)),
            float(data.get('Fatigue', 0)),
            float(data.get('Palpitations', 0)),
            float(data.get('Dizziness', 0)),
            float(data.get('Swelling', 0)),
            float(data.get('Pain_Arms_Jaw_Back', 0)),
            float(data.get('Cold_Sweats_Nausea', 0)),
            float(data.get('High_BP', 0)),
            float(data.get('High_Cholesterol', 0)),
            float(data.get('Diabetes', 0)),
            float(data.get('Smoking', 0)),
            float(data.get('Obesity', 0)),
            float(data.get('Sedentary_Lifestyle', 0)),
            float(data.get('Family_History', 0)),
            float(data.get('Chronic_Stress', 0)),
            float(data.get('Gender', 0)),
            float(data.get('Age', 0))
        ]
        h_prob = heart_model.predict_proba(np.array([h_features]))[0][1]
        
        # Calculate overall risk
        overall_risk = (d_prob + h_prob) / 2
        
        return jsonify({
            'diabetes_probability': float(d_prob),
            'heart_probability': float(h_prob),
            'overall_risk_probability': float(overall_risk),
            'overall_risk_percentage': float(overall_risk * 100),
            'overall_risk_level': 'High' if overall_risk > 0.5 else 'Moderate' if overall_risk > 0.3 else 'Low'
        })
    
    except Exception as e:
        print(f"Combined Prediction Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    """Serve the frontend index.html"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve other static files (css, js, etc.)"""
    return send_from_directory('.', path)

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Healthcare Risk Prediction API Server (Unified)")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health                  - Health check")
    print("  POST /predict/diabetes        - Diabetes prediction")
    print("  POST /predict/heart-disease   - Heart disease prediction")
    print("  POST /predict/combined        - Combined prediction")
    print("\nStarting server on http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
