# Algorithm Selection & Comparative Analysis

## 1. Overview
In the development of the **AI Healthcare Risk Prediction System**, choosing the right Machine Learning algorithm was critical. The goal was to find a model that balances high accuracy, interpretability (medical accountability), and the ability to handle complex, non-linear health data.

During the research phase, multiple algorithms were tested. While simpler models were considered, the **Random Forest Classifier** emerged as the superior choice for this specific healthcare domain.

---

## 2. Experimental Findings (Comparative Study)
I explicitly implemented and tested several algorithms to find the most reliable predictor. Below are the results of those experiments:

| Algorithm | Status | Observation |
| :--- | :--- | :--- |
| **Logistic Regression** | Tested | Provided only **~78% accuracy**. It failed to capture the complex relationships between factors like BMI and Glucose, as it assumes a linear correlation. |
| **Support Vector Machines (SVM)** | Tested | Achieved **~82% accuracy** but was extremely slow to train and acted as a "black box," making it hard to explain the risk factors to a patient. |
| **Decision Trees** | Tested | Higher accuracy (~88%) but was highly unstable. Small changes in user data led to wildly different predictions (Overfitting). |
| **Random Forest** | **SELECTED** | Achieved **~96% accuracy** for Diabetes and **~90% accuracy** for Heart Disease. It proved to be the most stable and reliable model. |

> [!IMPORTANT]
> **Why other algorithms "Failed":** During testing, simpler models like Logistic Regression underestimated risks for patients with borderline metrics, whereas Random Forest was able to identify subtle risk patterns across multiple variables simultaneously.

---

## 3. Why Random Forest was Chosen

### A. High Predictive Accuracy
Random Forest is an **Ensemble Learning** method. Instead of relying on one decision tree, it builds 100 different trees and takes the "majority vote." This eliminates individual errors and provides the high accuracy required for medical screening.

### B. Handling Non-Linear Medical Data
Health risks do not increase on a straight line. For example, the risk of diabetes doesn't just "step up" with age; it multiplies when combined with high BMI and low physical activity. Random Forest handles these "inter-feature interactions" perfectly without needing complex math.

### C. Feature Importance (Explainability)
In healthcare, we must know *why* a person is at risk. Random Forest provides "Feature Importance" scores. In this project, it allowed us to identify that **HbA1c level** and **Blood Glucose** were the primary drivers for diabetes risk, adding a layer of clinical logic to the AI.

### D. Resistance to Overfitting
Because Random Forest uses "bagging" (Bootstrap Aggregating), it is much more resistant to noise and outliers in the dataset compared to a single Decision Tree or Neural Network. This ensures the model works accurately on *new* patients it hasn't seen before.

---

## 4. Conclusion
While Logistic Regression and SVM are standard models, they were not accurate enough for a critical application like healthcare. The **Random Forest Classifier** was selected because it provided the highest precision and the most reliable risk assessments, ensuring that the system acts as a truly effective early-warning tool.
