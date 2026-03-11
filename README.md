# MLOps Churn Prediction Pipeline

![CI](https://github.com/Tushki4/mlops-churn-pipeline/actions/workflows/ci.yml/badge.svg)

End-to-end MLOps project: from raw data to a versioned, containerised, tested prediction API.
Built with sklearn, XGBoost, MLflow, FastAPI, Docker, and GitHub Actions.

## Architecture

```
Raw Data → EDA → sklearn Pipeline → MLflow Tracking
                                          ↓
                                   Model Registry (Production)
                                          ↓
                                     FastAPI API
                                          ↓
                                       Docker
                                          ↓
                                   GitHub Actions CI
```

## Dataset

IBM Telco Customer Churn — 7,043 customers, 26.5% churn rate.
[kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Model Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression ✓ | 0.842 | — | — | — |
| Random Forest | 0.840 | — | — | — |
| XGBoost | 0.828 | — | — | — |

*Fill in Precision/Recall/F1 from your MLflow UI*

**Best model:** Logistic Regression promoted to Production in MLflow Registry.

**Key finding:** A dummy model (always predict No) achieves 73.5% accuracy —
making accuracy meaningless here. ROC-AUC used as primary metric.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/mlops-churn-pipeline
cd mlops-churn-pipeline
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Train and register
mlflow server --host 127.0.0.1 --port 5000 &
python src/train.py
python src/register_model.py

# Start API
uvicorn src.api:app --port 8000

# Test
pytest tests/test_api.py -v
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 65.5, ...}'
```

Response:
```json
{
  "churn_probability": 0.7341,
  "churn_prediction": 1,
  "risk_tier": "High Risk"
}
```

## Project Structure

```
mlops-churn-pipeline/
├── src/
│   ├── features.py        # sklearn preprocessing pipeline
│   ├── train.py           # MLflow experiment tracking
│   ├── register_model.py  # Model Registry promotion
│   └── api.py             # FastAPI prediction endpoint
├── tests/
│   └── test_api.py        # 5 API tests
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_pipeline_check.ipynb
│   └── 03_feature_importance.ipynb
├── Dockerfile
└── .github/workflows/ci.yml
```

## Stack
Python · pandas · scikit-learn · XGBoost · MLflow · FastAPI · Pydantic · Docker · GitHub Actions · pytest