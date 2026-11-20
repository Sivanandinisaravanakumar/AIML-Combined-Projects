
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_xgboost.json"

# MLflow tracking setup
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("Financial-Fraud-Detection")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

params = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": float(scale_pos_weight),
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 42,
}

with mlflow.start_run():
    model = XGBClassifier(**params)

    print("Training XGBoost model...")

with mlflow.start_run(run_name="IEEE-CIS-XGBoost"):

    mlflow.log_params({
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "fraud_ratio": float(y.mean())
    })

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50
    )

    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "PR_AUC": float(average_precision_score(y_valid, probabilities)),
        "ROC_AUC": float(roc_auc_score(y_valid, probabilities)),
        "Precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "Recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "F1": float(f1_score(y_valid, predictions, zero_division=0)),
        "Confusion_Matrix": confusion_matrix(y_valid, predictions).tolist(),
        "Best_Iteration": int(model.best_iteration),
        "Threshold": 0.5
    }

    model.save_model(str(MODEL_DIR / "ieee_xgboost.json"))
    joblib.dump(encoder, MODEL_DIR / "ordinal_encoder.joblib")
    joblib.dump(medians, MODEL_DIR / "numeric_medians.joblib")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump({
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "all": feature_cols
        }, f, indent=4)

    mlflow.log_metrics({
        "PR_AUC": metrics["PR_AUC"],
        "ROC_AUC": metrics["ROC_AUC"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"]
    })

    mlflow.log_artifact(str(MODEL_DIR / "ieee_xgboost.json"))
    mlflow.log_artifact(str(MODEL_DIR / "ordinal_encoder.joblib"))
    mlflow.log_artifact(str(MODEL_DIR / "numeric_medians.joblib"))
    mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))
    mlflow.log_artifact(str(MODEL_DIR / "feature_columns.json"))

    print("\nIEEE-CIS MODEL TRAINING COMPLETE")
    print(json.dumps(metrics, indent=4))
    print("Saved model to:", MODEL_DIR)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())