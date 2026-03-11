import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow
import os

# ── MODEL LOADING ─────────────────────────────────────────────
# Load model once at startup — not on every request.
# Stored in a dict so it can be accessed by all endpoints.
MODEL_STORE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when API starts, clean up when it stops."""
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    MODEL_NAME = os.getenv("MODEL_NAME", "ChurnPredictor")
    MODEL_STORE["pipeline"] = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/Production"
    )
    print(f"✓ Model '{MODEL_NAME}' loaded from Production")
    yield
    MODEL_STORE.clear()

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using MLflow Production model",
    version="1.0.0",
    lifespan=lifespan
)


# ── PYDANTIC SCHEMA ───────────────────────────────────────────
# Defines exactly what fields the /predict endpoint expects.
# FastAPI validates every incoming request against this automatically.
class CustomerFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
                "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }

# ── HEALTH ENDPOINT ───────────────────────────────────────────
@app.get("/health")
def health():
    """Returns API status and whether model is loaded."""
    model_loaded = "pipeline" in MODEL_STORE
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": "1.0.0"
    }

# ── PREDICT ENDPOINT ──────────────────────────────────────────
@app.post("/predict")
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    Returns probability, binary prediction, and risk tier.
    """
    if "pipeline" not in MODEL_STORE:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow server is running."
        )
        
    try:
        # Convert Pydantic model to DataFrame (pipeline expects DataFrame)
        input_df = pd.DataFrame([customer.model_dump()])

        # Run through the full pipeline (preprocessing + prediction)
        pipeline = MODEL_STORE["pipeline"]
        churn_proba = pipeline.predict_proba(input_df)[0, 1]
        churn_pred  = int(churn_proba >= 0.5)

        # Risk tier logic
        if churn_proba >= 0.7:
            risk_tier = "High Risk"
        elif churn_proba >= 0.4:
            risk_tier = "Medium Risk"
        else:
            risk_tier = "Low Risk"

        return {
            "churn_probability": round(float(churn_proba), 4),
            "churn_prediction":  churn_pred,
            "risk_tier":         risk_tier,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")