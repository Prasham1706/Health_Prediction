# Healthcare AI Risk Prediction System

An ML-powered healthcare risk prediction system that uses trained Random Forest models to predict diabetes and heart disease risks based on patient data.

## Features

- **Diabetes Prediction**: ML model trained on 100K+ patient records
- **Heart Disease Prediction**: ML model trained on 70K+ patient records
- **Real-time Predictions**: Flask REST API serving model predictions
- **Interactive Frontend**: Modern glassmorphism UI with animated results
- **Comprehensive Analysis**: Risk factor breakdown and personalized recommendations

## Project Structure

```
ai_project/
├── diabetes_prediction_dataset.csv       # Training data for diabetes model
├── heart_disease_risk_dataset_earlymed.csv  # Training data for heart disease model
├── train_models.py                       # ML model training script
├── app.py                                # Flask API server
├── index.html                            # Frontend interface
├── script.js                             # Frontend JavaScript (API integration)
├── styles.css                            # Styling
├── requirements.txt                      # Python dependencies
├── diabetes_model.pkl                    # Trained diabetes model (generated)
└── heart_disease_model.pkl               # Trained heart disease model (generated)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- Flask & Flask-CORS (API server)
- scikit-learn (ML models)
- pandas & numpy (data processing)
- joblib (model serialization)

### 2. Train the Models

```bash
python train_models.py
```

This will:

- Load and preprocess both CSV datasets
- Train Random Forest classifiers
- Evaluate model performance
- Save trained models as `.pkl` files

Expected output:

- Diabetes Model Accuracy: ~96%
- Heart Disease Model Accuracy: ~85-90%

### 3. Start the API Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Open the Frontend

Open `index.html` in your web browser or use a local server:

```bash
# Using Python's built-in server
python -m http.server 8000
```

Then navigate to `http://localhost:8000`

## API Endpoints

### Health Check

```
GET /health
```

### Diabetes Prediction

```
POST /predict/diabetes
Content-Type: application/json

{
  "gender": 1,              // 0=Female, 1=Male
  "age": 45,
  "hypertension": 0,        // 0 or 1
  "heart_disease": 0,       // 0 or 1
  "smoking_history": 0,     // 0=never, 2=current, 3=former
  "bmi": 28.5,
  "HbA1c_level": 5.7,
  "blood_glucose_level": 140
}
```

### Heart Disease Prediction

```
POST /predict/heart-disease
Content-Type: application/json

{
  "Chest_Pain": 0,
  "Shortness_of_Breath": 0,
  "Fatigue": 0,
  "Palpitations": 0,
  "Dizziness": 0,
  "Swelling": 0,
  "Pain_Arms_Jaw_Back": 0,
  "Cold_Sweats_Nausea": 0,
  "High_BP": 1,
  "High_Cholesterol": 1,
  "Diabetes": 0,
  "Smoking": 0,
  "Obesity": 0,
  "Sedentary_Lifestyle": 1,
  "Family_History": 1,
  "Chronic_Stress": 0,
  "Gender": 1,
  "Age": 45
}
```

## How It Works

1. **Data Collection**: User fills out health assessment form
2. **API Request**: Frontend sends data to Flask backend
3. **ML Prediction**: Trained models analyze the data
4. **Risk Calculation**: Models return probability scores
5. **Results Display**: Frontend shows animated risk scores and recommendations

## Model Details

### Diabetes Model

- **Algorithm**: Random Forest Classifier
- **Features**: Gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose
- **Training Data**: 100,002 patient records
- **Performance**: ~96% accuracy

### Heart Disease Model

- **Algorithm**: Random Forest Classifier
- **Features**: 18 features including symptoms, risk factors, demographics
- **Training Data**: 70,002 patient records
- **Performance**: ~85-90% accuracy

## Technologies Used

- **Backend**: Python, Flask, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML**: Random Forest, pandas, numpy
- **Design**: Glassmorphism, CSS animations

## Disclaimer

⚠️ **This tool is for educational and informational purposes only.**

- Not a substitute for professional medical advice
- Always consult healthcare professionals for medical decisions
- Predictions are based on statistical models and may not be 100% accurate

## Author

Created by Prasham - 2026

## License

All rights reserved.
