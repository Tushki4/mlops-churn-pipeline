import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score,
    f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from features import load_and_prepare, build_preprocessor, get_train_test_split



def train_and_log(model, model_name: str, params: dict,
                  X_train, X_test, y_train, y_test):
    """Train one model evaluate and log everything to MLflow"""
    with mlflow.start_run(run_name=model_name):
        
        # 1. Build Full pipeline
        
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # 2 Train
        
        pipeline.fit(X_train, y_train)
        
        # 3 Predict
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:,1]
        
        # 4 Calculate Metrics
        metrics = {
            'roc_auc':   roc_auc_score(y_test, y_pred_proba),
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
        }

        # ── 5. LOG PARAMS + METRICS ─────────────────────────────
        mlflow.log_param('model_type', model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            
         # ── 6. LOG CONFUSION MATRIX AS IMAGE ────────────────────
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn']).plot(ax=ax)
        ax.set_title(f'{model_name} — Confusion Matrix')
        plt.tight_layout()
        fig.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close(fig)
    
    # ── 7. LOG THE FULL PIPELINE AS A MODEL ─────────────────
        mlflow.sklearn.log_model(pipeline, 'model')

        # ── 8. PRINT SUMMARY ────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  {model_name}")
        print(f"{'─'*50}")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")

        return metrics['roc_auc']
    
def main():
    # ── LOAD DATA ───────────────────────────────────────────
    X, y = load_and_prepare('data/churn_raw.csv')
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]}")
    
        # ── SET MLFLOW EXPERIMENT ───────────────────────────────
    # All 3 runs will appear under this experiment name in the UI
    mlflow.set_experiment('churn-prediction')
    
  # ── DEFINE THE THREE MODELS ─────────────────────────────
    models = [
        (
            LogisticRegression(max_iter=1000, random_state=42),
            'LogisticRegression',
            {'max_iter': 1000, 'random_state': 42}
        ),
        (
            RandomForestClassifier(n_estimators=200, max_depth=10,
                                   random_state=42, n_jobs=-1),
            'RandomForest',
            {'n_estimators': 200, 'max_depth': 10, 'random_state': 42}
        ),
        (
            XGBClassifier(n_estimators=200, max_depth=6,
                          learning_rate=0.1, random_state=42,
                          eval_metric='logloss', verbosity=0),
            'XGBoost',
            {'n_estimators': 200, 'max_depth': 6,
             'learning_rate': 0.1, 'random_state': 42}
        ),
    ]
    
     # ── TRAIN ALL THREE ─────────────────────────────────────
    results = {}
    for model, name, params in models:
        auc = train_and_log(model, name, params,
                            X_train, X_test, y_train, y_test)
        results[name] = auc

    # ── FINAL SUMMARY ───────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  FINAL RESULTS — ROC-AUC")
    print(f"{'═'*50}")
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {auc:.4f}")
    best = max(results, key=results.get)
    print(f"\n  Best model: {best} ({results[best]:.4f})")
    print("\n  Open MLflow UI: mlflow ui --port 5000")


if __name__ == '__main__':
    main()
    