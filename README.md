🧠 Stroke Risk Prediction

A machine learning project that predicts stroke risk using patient health data. The goal is to help identify key risk factors such as hypertension and obesity, enabling early prevention and better decision-making in healthcare.

📊 Dataset

Source: Kaggle – Stroke Prediction Dataset

Size: 43,000+ patient records

Features:

Demographics: Age, Gender, Residence type, Marital status

Health indicators: Hypertension, Heart disease, BMI, Glucose level

Lifestyle: Smoking status, Work type

Target: stroke (0 = No stroke, 1 = Stroke)

⚙️ Features

✔️ End-to-end pipeline: preprocessing → modeling → evaluation
✔️ Handles missing values, outliers, and feature scaling
✔️ Implements Logistic Regression and Decision Tree models
✔️ Includes confusion matrix visualization for model evaluation
✔️ Identifies key risk factors like hypertension, obesity, and glucose level

🛠️ Tech Stack

Languages & Libraries: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

ML Models: Logistic Regression, Decision Tree

Tools: Jupyter Notebook, Git/GitHub

🚀 Quickstart

Clone the repository:

git clone https://github.com/YOUR_USERNAME/stroke-risk-prediction.git
cd stroke-risk-prediction

Install dependencies:

pip install -r requirements.txt

Run training:

python src/train_model.py

📈 Results

Logistic Regression: Simple and interpretable, good baseline performance

Decision Tree: Captures non-linear patterns, improves recall

Finding: Patients with hypertension and obesity had a 30% higher stroke risk

🎯 Future Improvements

Add Random Forest & XGBoost for better accuracy

Perform SHAP / feature importance analysis for explainability

Deploy as a Flask/Streamlit web app for interactive risk prediction

🙌 Acknowledgments

Dataset from Kaggle – Stroke Prediction Dataset


