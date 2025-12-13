from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import os
import numpy as np
from sklearn.pipeline import Pipeline

# --------- Load model package dynamically ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_churn_model.pkl")

# Fallback for local testing if path differs
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "A:\\practice\\models\\xgboost_churn_model.pkl"

try:
    package = joblib.load(MODEL_PATH)
    loaded_object = package["model"]
    threshold = float(package["threshold"])
    
    # --- FIX: Handle Pipeline for SHAP ---
    if isinstance(loaded_object, Pipeline):
        # 1. Extract the actual model (assuming it's the last step)
        model = loaded_object.steps[-1][1]
        
        # 2. Extract the preprocessor (all steps BEFORE the last)
        # We need this to transform data before SHAP sees it
        preprocessor = loaded_object[:-1]
        is_pipeline = True
    else:
        model = loaded_object
        preprocessor = None
        is_pipeline = False
    
    # Initialize SHAP Explainer on the RAW model
    explainer = shap.TreeExplainer(model)
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# --------- FastAPI app ----------
app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Input schema ----------
class CustomerFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# --------- Prediction endpoint ----------
@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    # Convert input to DataFrame
    data_dict = features.model_dump() if hasattr(features, 'model_dump') else features.dict()
    data = pd.DataFrame([data_dict])

    try:
        # ---------------------------------------------------------
        # 1. PREDICTION (Use the full pipeline)
        # ---------------------------------------------------------
        # The full pipeline handles preprocessing automatically
        if is_pipeline:
            full_pipeline = loaded_object
            prob = float(full_pipeline.predict_proba(data)[:, 1][0])
        else:
            prob = float(model.predict_proba(data)[:, 1][0])
            
        churn = prob >= threshold

        # ---------------------------------------------------------
        # 2. EXPLAINABILITY (SHAP)
        # ---------------------------------------------------------
        # SHAP needs the data exactly as the model sees it (numbers only, no strings).
        
        if is_pipeline:
            # Manually run the preprocessing steps
            shap_data = preprocessor.transform(data)
            
            # Try to get feature names after preprocessing (e.g. "Geography_France")
            # This works if the preprocessor has 'get_feature_names_out' (sklearn > 1.0)
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                # Fallback: if we can't get names, use generic ones
                feature_names = [f"Feature {i}" for i in range(shap_data.shape[1])]
        else:
            shap_data = data
            feature_names = data.columns.tolist()

        # Calculate SHAP values
        shap_values = explainer.shap_values(shap_data)
        
        # Handle SHAP return types (binary classification often returns list of 2 arrays)
        if isinstance(shap_values, list):
            # Index 1 is usually the positive class (Churn = 1)
            shap_values = shap_values[1]
            
        # Extract the single row
        single_row_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        # Match feature names to values
        # IMPORTANT: If OneHotEncoding happened, feature_names list will match the transformed data
        explanation = []
# NEW CODE (Removes 'num__' and 'cat__' prefixes)
        for name, value in zip(feature_names, single_row_shap):
            clean_name = str(name).split("__")[-1] if "__" in str(name) else str(name)
            explanation.append({
               "feature": clean_name,
                "impact": float(value)
                   })
    
    

        # Sort by absolute impact
        explanation.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return {
            "churn": bool(churn),
            "probability": prob,
            "threshold": threshold,
            "explanation": explanation[:3] # Top 3 factors
        }

    except Exception as e:
        # Print error to terminal for debugging
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --------- Serve frontend (static) ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")