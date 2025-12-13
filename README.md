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
