# 🧠 Customer Churn Prediction Platform (FastAPI + ML + GenAI)

This project is a FastAPI-based machine learning platform designed to predict customer churn using traditional ML models and provide **natural language explanations** for predictions using **Gemini models (LLM)**.

---

## 🚀 Features

- Upload your dataset and define the target column
- Clean and preprocess data (missing value handling, encoding, scaling)
- Train ML models (`Logistic Regression`, `Random Forest`, `XGBoost`)
- Make predictions on new data
- Generate GenAI-powered explanations for each prediction

---

## 🏗️ Tech Stack

- **FastAPI** — Web API framework
- **Scikit-learn**, **XGBoost** — ML training and evaluation
- **Pandas**, **Joblib** — Data processing and persistence
- **Google Generative AI (Gemini Pro)** — Natural language explanations

---

## 📁 Project Structure

Assignment Churn Prediction Project/
├── app/
│ ├── main.py
│ ├── routers/
│ │ ├── upload.py
│ │ ├── cleaning.py
│ │ ├── train.py
│ │ └── predict.py
│ ├── utils/
│ │ ├── genai_utils.py
├── models/ # Saved models and encoders
├── data/ # Uploaded + cleaned data
├── samples/ # Sample input files
├── .env # Gemini API key (keep secret)
├── requirements.txt
└── README.md


STEPS TO DO:

API Endpoints
1. /upload – Dataset Upload & Target Selection

Upload a CSV file
Provide a target column (e.g., "Churn")

2. /data_cleaning – Preprocessing

{
  "missing_handling": "impute",    // or "drop"
  "encoding": "label",             // or "onehot"
  "scaling": "standard"            // or "minmax"
}

3. /train_model – Model Training

{
  "models": ["LogisticRegression", "RandomForest"]
}

4. /predict – Batch Prediction + LLM Explanation

Upload a test CSV file

Provide a trained model name