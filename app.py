"""
Healthcare Risk Prediction API
Flask backend server for serving ML model predictions
"""

from flask import Flask, request, jsonify
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
    print("✓ Models loaded successfully")
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
    """
    Predict diabetes risk
    Expected input features:
    - gender (0=Female, 1=Male)
    - age
    - hypertension (0 or 1)
    - heart_disease (0 or 1)
    - smoking_history (0-5)
    - bmi
    - HbA1c_level
    - blood_glucose_level
    """
    try:
        if diabetes_model is None:
            return jsonify({'error': 'Diabetes model not loaded'}), 500
        
        data = request.json
        
        # Extract features in correct order
        features = [
            data.get('gender', 0),
            data.get('age', 0),
            data.get('hypertension', 0),
            data.get('heart_disease', 0),
            data.get('smoking_history', 0),
            data.get('bmi', 0),
            data.get('HbA1c_level', 0),
            data.get('blood_glucose_level', 0)
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
        return jsonify({'error': str(e)}), 400

@app.route('/predict/heart-disease', methods=['POST'])
def predict_heart_disease():
    """
    Predict heart disease risk
    Expected input features (all 0 or 1):
    - Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations, Dizziness
    - Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea
    - High_BP, High_Cholesterol, Diabetes, Smoking, Obesity
    - Sedentary_Lifestyle, Family_History, Chronic_Stress
    - Gender (0=Female, 1=Male)
    - Age
    """
    try:
        if heart_model is None:
            return jsonify({'error': 'Heart disease model not loaded'}), 500
        
        data = request.json
        
        # Extract features in correct order (matching dataset columns)
        features = [
            data.get('Chest_Pain', 0),
            data.get('Shortness_of_Breath', 0),
            data.get('Fatigue', 0),
            data.get('Palpitations', 0),
            data.get('Dizziness', 0),
            data.get('Swelling', 0),
            data.get('Pain_Arms_Jaw_Back', 0),
            data.get('Cold_Sweats_Nausea', 0),
            data.get('High_BP', 0),
            data.get('High_Cholesterol', 0),
            data.get('Diabetes', 0),
            data.get('Smoking', 0),
            data.get('Obesity', 0),
            data.get('Sedentary_Lifestyle', 0),
            data.get('Family_History', 0),
            data.get('Chronic_Stress', 0),
            data.get('Gender', 0),
            data.get('Age', 0)
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
        return jsonify({'error': str(e)}), 400

@app.route('/predict/combined', methods=['POST'])
def predict_combined():
    """
    Get both diabetes and heart disease predictions
    Combines features from both models
    """
    try:
        data = request.json
        
        # Get diabetes prediction
        diabetes_result = predict_diabetes()
        diabetes_data = diabetes_result.get_json()
        
        # Get heart disease prediction
        heart_result = predict_heart_disease()
        heart_data = heart_result.get_json()
        
        # Calculate overall risk
        overall_risk = (diabetes_data.get('risk_probability', 0) + 
                       heart_data.get('risk_probability', 0)) / 2
        
        return jsonify({
            'diabetes': diabetes_data,
            'heart_disease': heart_data,
            'overall_risk_probability': overall_risk,
            'overall_risk_percentage': overall_risk * 100,
            'overall_risk_level': 'High' if overall_risk > 0.5 else 'Moderate' if overall_risk > 0.3 else 'Low'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Healthcare Risk Prediction API Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health                  - Health check")
    print("  POST /predict/diabetes        - Diabetes prediction")
    print("  POST /predict/heart-disease   - Heart disease prediction")
    print("  POST /predict/combined        - Combined prediction")
    print("\nStarting server on http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
