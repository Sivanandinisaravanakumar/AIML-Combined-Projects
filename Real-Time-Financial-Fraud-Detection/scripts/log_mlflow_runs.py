import json
from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "ieee_cis"
mlflow.set_tracking_uri("sqlite:///mlflow_ieee.db")
mlflow.set_experiment("IEEE-CIS-Fraud-Detection")

ALIASES = {
    "f1_score": "f1", "F1": "f1",
    "ROC_AUC": "roc_auc",
    "average_precision": "pr_auc", "PR_AUC": "pr_auc",
    "Precision": "precision", "Recall": "recall",
    "best_threshold": "threshold", "Threshold": "threshold",
    "hybrid_best_validation_threshold": "threshold",
    "Best_Iteration": "best_iteration",
}

RUNS = [
    ("xgboost_baseline", "metrics.json", {"model": "XGBoost", "imbalance": "class weights", "threshold": "0.5"}),
    ("gnn_v1", "gnn_metrics.json", {"model": "GCN v1"}),
    ("gnn_v2", "gnn_metrics_v2.json", {"model": "GCN v2"}),
    ("hybrid_xgb_gnn", "hybrid_metrics.json", {"model": "XGBoost + GNN blend"}),
    ("xgboost_tuned_threshold", "threshold.json", {"model": "XGBoost", "threshold": "tuned"}),
]


def flatten(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield k, float(v)


for run_name, fname, params in RUNS:
    path = MODELS / fname
    if not path.exists():
        print("skip (missing):", fname)
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = {}
    for k, v in flatten(data):
        metrics[ALIASES.get(k, k)] = v
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.set_tag("source", "logged from saved training outputs")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(path))
        if run_name == "xgboost_baseline":
            mlflow.log_artifact(str(MODELS / "ieee_xgboost.json"))
            mlflow.log_artifact(str(MODELS / "feature_columns.json"))
    print("logged:", run_name, "->", sorted(metrics))