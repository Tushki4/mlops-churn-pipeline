import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "ChurnPredictor"
EXPERIMENT_NAME = "churn-prediction"
METRIC = "roc_auc"

def get_best_run(experiment_name: str, metric: str) -> str:
    """Find the run ID with the highest value of the given metric."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found. Run train.py first.")

    best = runs[0]
    print(f"Best run: {best.data.tags.get('mlflow.runName', best.info.run_id)}")
    print(f"  ROC-AUC: {best.data.metrics[metric]:.4f}")
    print(f"  Run ID:  {best.info.run_id}")
    return best.info.run_id

def register_and_promote(run_id: str, model_name: str):
    """Register model from a run and promote it to Production."""
    client = MlflowClient()

    # Register model — creates version 1 (or increments if name exists)
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    version = result.version
    print(f"\nRegistered '{model_name}' version {version}")

    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True  # previous Production → Archived
    )
    print(f"Promoted to Production (previous version archived)")
    print(f"\nLoad it in Day 5 with:")
    print(f"  mlflow.sklearn.load_model('models:/{model_name}/Production')")
    
  
if __name__ == '__main__':
    best_run_id = get_best_run(EXPERIMENT_NAME, METRIC)
    register_and_promote(best_run_id, MODEL_NAME)
    print("\n✓ Model registry complete. Open MLflow UI to verify.")   