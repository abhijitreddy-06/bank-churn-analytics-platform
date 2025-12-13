# 🏦 Customer Churn Prediction Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Models-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

This project is an end-to-end Machine Learning web application designed to predict whether a bank customer is likely to churn (leave the bank). 

Unlike standard "black box" predictions, this application provides **Explainable AI (XAI)** insights. It uses **SHAP (SHapley Additive exPlanations)** to break down exactly *why* a specific customer is at risk, empowering relationship managers to take targeted retention actions.

## ✨ Key Features

* **Real-time Inference:** Instant predictions via a high-performance **FastAPI** backend.
* **Explainable AI:** Integration with **SHAP** to visualize top contributing risk factors (e.g., "Age increases risk by 15%").
* **Robust Pipeline:** Handles raw data preprocessing automatically using Scikit-Learn Pipelines.
* **Interactive UI:** A responsive, modern frontend built with vanilla JavaScript and CSS (no heavy framework overhead).
* **Data Validation:** Strict type checking and range validation using **Pydantic** schemas.

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3, JavaScript (ES6+), Fetch API
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Numpy, SHAP/Joblib

## 💼 Business Impact

Customer churn directly impacts bank profitability, as acquiring new customers is significantly more expensive than retaining existing ones.

This platform helps banks:
- Identify high-risk customers early
- Understand *why* a customer is likely to churn
- Enable relationship managers to take data-driven retention actions
- Reduce revenue loss and improve customer lifetime value (CLV)

## ⚙️ How It Works

1. User enters customer details in the frontend UI
2. Frontend sends data to FastAPI `/predict` endpoint
3. Backend validates input using Pydantic schemas
4. Preprocessing pipeline transforms raw features
5. XGBoost model predicts churn probability
6. SHAP explains feature contributions
7. Prediction + explanations are returned to the UI

## 🔌 API Example

**Endpoint:** `/predict`

**Request:**
```json
{
  "age": 42,
  "credit_score": 610,
  "balance": 125000,
  "tenure": 3,
  "is_active_member": 0
}
```
## 📂 Project Structure

```bash
customer-churn-prediction/
├── models/
│   └── xgboost_churn_model.pkl    # Trained pipeline (Preprocessor + Model)
├── static/
│   └── index.html                 # Single Page Application frontend
├── main.py                        # FastAPI application entry point
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```
## 🔧 4️⃣ Add “Model Performance” Section

Even simple metrics help a lot.

```md
## 📊 Model Performance

- Model: XGBoost (Hyperparameter Tuned)
- ROC-AUC: 0.86
- Accuracy: 82%
- Precision (Churn): 79%
- Recall (Churn): 81%

Evaluation performed on held-out test dataset.
