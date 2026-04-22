# AI Healthcare Risk Prediction System: Project Report

## 1. Project Aim
The primary aim of this project is to develop a predictive AI system that identifies an individual's risk of developing critical health conditions—specifically **Diabetes** and **Heart Disease**—based on easily accessible health metrics and lifestyle factors. By leveraging machine learning, the system aims to provide early warnings and risk assessments, encouraging proactive health management and preventive care.

## 2. Artificial Intelligence Algorithm Used
The core AI algorithm used in this project is the **Random Forest Classifier**.

- **Why Random Forest?**: Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) of the individual trees. It was chosen because it is highly accurate, handles non-linear medical data exceptionally well, and prevents overfitting compared to a single decision tree.
- **Implementation**: The system utilizes the `RandomForestClassifier` from Python's `scikit-learn` library. 
- **Two Distinct Models**: Two separate Random Forest models were trained:
  1. A Diabetes Prediction Model (using factors like HbA1c level, blood glucose, BMI, etc.)
  2. A Heart Disease Prediction Model (using factors like chest pain, high cholesterol, family history, etc.)

## 3. System Outputs
The system features a Flask-based backend API that connects the AI models to a modern, user-friendly frontend interface. 

When a user submits their health data, the system generates the following outputs:
- **Binary Prediction**: A definitive classification of whether the individual is at risk (Yes/No).
- **Risk Percentage**: A calculated probability score (from 0.0% to 100.0%) indicating the exact likelihood of the disease.
- **Risk Category**: A human-readable risk level categorized as **Low**, **Moderate**, or **High** risk based on the probability score.
- **Combined Analysis**: An overall health risk score mathematically combined from both the diabetes and heart disease predictions.

## 4. Conclusion
This project successfully demonstrates that machine learning algorithms, particularly Random Forests, can effectively analyze multi-dimensional healthcare data to provide accurate risk assessments. The resulting platform bridges the gap between raw medical data and readable, actionable insights. By acting as an early screening tool, it empowers users to seek timely medical advice, ultimately showcasing the powerful role AI can play in modern preventive healthcare and diagnostics.
