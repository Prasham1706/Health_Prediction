"""
Healthcare Risk Prediction - ML Model Training Script
Trains Random Forest models for diabetes and heart disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Healthcare Risk Prediction - Model Training")
print("=" * 60)

# ============================================================================
# 1. DIABETES PREDICTION MODEL
# ============================================================================
print("\n[1/2] Training Diabetes Prediction Model...")
print("-" * 60)

# Load diabetes dataset
df_diabetes = pd.read_csv('diabetes_prediction_dataset.csv')
print(f"Loaded diabetes dataset: {df_diabetes.shape[0]} rows, {df_diabetes.shape[1]} columns")

# Data preprocessing
print("Preprocessing diabetes data...")

# Encode gender (Female=0, Male=1)
df_diabetes['gender'] = df_diabetes['gender'].map({'Female': 0, 'Male': 1})

# Encode smoking_history
smoking_map = {
    'never': 0,
    'No Info': 1,
    'current': 2,
    'former': 3,
    'ever': 4,
    'not current': 5
}
df_diabetes['smoking_history'] = df_diabetes['smoking_history'].map(smoking_map)

# Handle any missing values
df_diabetes = df_diabetes.dropna()

# Features and target
X_diabetes = df_diabetes.drop('diabetes', axis=1)
y_diabetes = df_diabetes['diabetes']

# Train-test split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
)

print(f"Training set: {X_train_d.shape[0]} samples")
print(f"Test set: {X_test_d.shape[0]} samples")

# Train Random Forest model
print("Training Random Forest classifier...")
diabetes_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
diabetes_model.fit(X_train_d, y_train_d)

# Evaluate model
y_pred_d = diabetes_model.predict(X_test_d)
accuracy_d = accuracy_score(y_test_d, y_pred_d)
precision_d = precision_score(y_test_d, y_pred_d)
recall_d = recall_score(y_test_d, y_pred_d)
f1_d = f1_score(y_test_d, y_pred_d)

print("\nDiabetes Model Performance:")
print(f"  Accuracy:  {accuracy_d:.4f}")
print(f"  Precision: {precision_d:.4f}")
print(f"  Recall:    {recall_d:.4f}")
print(f"  F1-Score:  {f1_d:.4f}")

# Feature importance
feature_importance_d = pd.DataFrame({
    'feature': X_diabetes.columns,
    'importance': diabetes_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
for idx, row in feature_importance_d.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model
joblib.dump(diabetes_model, 'diabetes_model.pkl')
print("\nDiabetes model saved as 'diabetes_model.pkl'")

# ============================================================================
# 2. HEART DISEASE PREDICTION MODEL
# ============================================================================
print("\n[2/2] Training Heart Disease Prediction Model...")
print("-" * 60)

# Load heart disease dataset
df_heart = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')
print(f"Loaded heart disease dataset: {df_heart.shape[0]} rows, {df_heart.shape[1]} columns")

# Data preprocessing
print("Preprocessing heart disease data...")

# All features are already numeric (0.0 or 1.0)
# No encoding needed

# Features and target
X_heart = df_heart.drop('Heart_Risk', axis=1)
y_heart = df_heart['Heart_Risk']

# Train-test split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)

print(f"Training set: {X_train_h.shape[0]} samples")
print(f"Test set: {X_test_h.shape[0]} samples")

# Train Random Forest model
print("Training Random Forest classifier...")
heart_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
heart_model.fit(X_train_h, y_train_h)

# Evaluate model
y_pred_h = heart_model.predict(X_test_h)
accuracy_h = accuracy_score(y_test_h, y_pred_h)
precision_h = precision_score(y_test_h, y_pred_h)
recall_h = recall_score(y_test_h, y_pred_h)
f1_h = f1_score(y_test_h, y_pred_h)

print("\nHeart Disease Model Performance:")
print(f"  Accuracy:  {accuracy_h:.4f}")
print(f"  Precision: {precision_h:.4f}")
print(f"  Recall:    {recall_h:.4f}")
print(f"  F1-Score:  {f1_h:.4f}")

# Feature importance
feature_importance_h = pd.DataFrame({
    'feature': X_heart.columns,
    'importance': heart_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
for idx, row in feature_importance_h.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model
joblib.dump(heart_model, 'heart_disease_model.pkl')
print("\nHeart disease model saved as 'heart_disease_model.pkl'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"\nDiabetes Model Accuracy:      {accuracy_d:.2%}")
print(f"Heart Disease Model Accuracy: {accuracy_h:.2%}")
print("\nModels saved successfully:")
print("  - diabetes_model.pkl")
print("  - heart_disease_model.pkl")
print("\nNext step: Run 'python app.py' to start the API server")
print("=" * 60)
