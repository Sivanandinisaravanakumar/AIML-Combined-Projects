
from pathlib import Path
import os
from datetime import datetime, timezone
import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "ieee_cis"
CANDIDATE_ROOT = MODEL_DIR / "candidates"

BACKTEST_MODE = os.getenv("CANDIDATE_BACKTEST", "0") == "1"

INCOMING_FILE = (
    ROOT / "data" / "ieee_cis" / "processed" / "train_features.csv"
    if BACKTEST_MODE
    else ROOT / "data" / "ieee_cis" / "incoming" / "labelled_transactions.csv"
)

TRIGGER_FILE = ROOT / "reports" / "live_retraining_trigger.json"

MLFLOW_DB = ROOT / "mlflow.db"
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)
mlflow.set_experiment("IEEE-CIS-Fraud-Detection")

MIN_LABELLED_ROWS = 1000
RANDOM_STATE = 42


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "PR_AUC": float(
            average_precision_score(y_true, probabilities)
        ),
        "ROC_AUC": float(
            roc_auc_score(y_true, probabilities)
        ),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "Threshold": float(threshold),
    }


def prepare_features(frame, numeric_cols, categorical_cols,
                     encoder, medians):

    X = frame[numeric_cols + categorical_cols].copy()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X.loc[:, numeric_cols] = X[numeric_cols].fillna(medians)

    for col in categorical_cols:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    categorical_data = pd.DataFrame(
        encoder.transform(X[categorical_cols]),
        columns=categorical_cols,
        index=X.index,
    )

    return pd.concat(
        [X[numeric_cols].astype("float64"), categorical_data],
        axis=1,
    )[numeric_cols + categorical_cols].astype("float64")


def main():

    print("Checking retraining eligibility...")
    print("MODE:", "BACKTEST - existing labelled dataset" if BACKTEST_MODE else "LIVE - incoming confirmed labels")

    if not TRIGGER_FILE.exists():
        raise FileNotFoundError("Drift trigger file not found.")

    with open(TRIGGER_FILE, encoding="utf-8") as f:
        trigger = json.load(f)

    if not trigger.get("retraining_required", False):
        print("NO_ACTION: Drift threshold not reached.")
        return

    if not INCOMING_FILE.exists():
        print("WAITING_FOR_LABELS: No new labelled data.")
        return

    data = pd.read_csv(INCOMING_FILE)

    if "isFraud" not in data.columns:
        raise ValueError("Incoming data has no isFraud column.")

    if data["isFraud"].isna().any():
        raise ValueError("Missing labels detected. Training blocked.")

    data["isFraud"] = pd.to_numeric(
        data["isFraud"], errors="coerce"
    )

    if data["isFraud"].isna().any():
        raise ValueError("Invalid labels detected.")

    if not data["isFraud"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if data["isFraud"].nunique() != 2:
        raise ValueError("Both fraud and legitimate labels are required.")

    if len(data) < MIN_LABELLED_ROWS:
        print("WAITING_FOR_MORE_LABELS:", len(data))
        return

    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_config = json.load(f)

    numeric_cols = feature_config["numeric"]
    categorical_cols = feature_config["categorical"]
    all_cols = feature_config["all"]

    missing = set(all_cols) - set(data.columns)

    if missing:
        raise ValueError(
            f"Incoming data missing model features: {sorted(missing)}"
        )

    y = data["isFraud"].astype(int)
    X = data[all_cols].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_valid))

    # Candidate preprocessing: fit only on candidate training rows.
    medians = (
        X_train[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .median()
        .fillna(0)
    )

    candidate_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_train_prepared = X_train.copy()
    X_valid_prepared = X_valid.copy()

    for col in numeric_cols:
        X_train_prepared[col] = pd.to_numeric(
            X_train_prepared[col], errors="coerce"
        )
        X_valid_prepared[col] = pd.to_numeric(
            X_valid_prepared[col], errors="coerce"
        )

    X_train_prepared.loc[:, numeric_cols] = (
        X_train_prepared[numeric_cols].fillna(medians)
    )
    X_valid_prepared.loc[:, numeric_cols] = (
        X_valid_prepared[numeric_cols].fillna(medians)
    )

    for col in categorical_cols:
        X_train_prepared[col] = (
            X_train_prepared[col]
            .astype("string").fillna("__MISSING__").astype(str)
        )
        X_valid_prepared[col] = (
            X_valid_prepared[col]
            .astype("string").fillna("__MISSING__").astype(str)
        )

    train_cat = pd.DataFrame(
        candidate_encoder.fit_transform(
            X_train_prepared[categorical_cols]
        ),
        columns=categorical_cols,
        index=X_train_prepared.index,
    )

    valid_cat = pd.DataFrame(
        candidate_encoder.transform(
            X_valid_prepared[categorical_cols]
        ),
        columns=categorical_cols,
        index=X_valid_prepared.index,
    )

    X_train_ready = pd.concat(
        [X_train_prepared[numeric_cols], train_cat],
        axis=1,
    )[all_cols].astype("float64")

    X_valid_ready = pd.concat(
        [X_valid_prepared[numeric_cols], valid_cat],
        axis=1,
    )[all_cols].astype("float64")

    # Load current production model and preprocessing.
    baseline_model = XGBClassifier()
    baseline_model.load_model(
        str(MODEL_DIR / "ieee_xgboost.json")
    )

    baseline_encoder = joblib.load(
        MODEL_DIR / "ordinal_encoder.joblib"
    )
    baseline_medians = joblib.load(
        MODEL_DIR / "numeric_medians.joblib"
    )

    X_valid_baseline = prepare_features(
        X_valid,
        numeric_cols,
        categorical_cols,
        baseline_encoder,
        baseline_medians,
    )

    with open(MODEL_DIR / "threshold.json") as f:
        threshold = float(json.load(f)["best_threshold"])

    baseline_probabilities = baseline_model.predict_proba(
        X_valid_baseline
    )[:, 1]

    # Train candidate model.
    scale_pos_weight = float(
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    candidate_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
    )

    candidate_model.fit(
        X_train_ready,
        y_train,
        eval_set=[(X_valid_ready, y_valid)],
        verbose=50,
    )

    candidate_probabilities = candidate_model.predict_proba(
        X_valid_ready
    )[:, 1]

    baseline_metrics = calculate_metrics(
        y_valid, baseline_probabilities, threshold
    )

    candidate_metrics = calculate_metrics(
        y_valid, candidate_probabilities, threshold
    )

    # Conservative quality gate: all three must be maintained
    # or improved on the same validation rows.
    quality_passed = (
        candidate_metrics["PR_AUC"] >= baseline_metrics["PR_AUC"]
        and candidate_metrics["Recall"] >= baseline_metrics["Recall"]
        and candidate_metrics["F1"] >= baseline_metrics["F1"]
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    candidate_dir = CANDIDATE_ROOT / timestamp
    candidate_dir.mkdir(parents=True, exist_ok=False)

    candidate_model.save_model(
        str(candidate_dir / "ieee_xgboost.json")
    )
    joblib.dump(
        candidate_encoder,
        candidate_dir / "ordinal_encoder.joblib",
    )
    joblib.dump(
        medians,
        candidate_dir / "numeric_medians.joblib",
    )

    with open(candidate_dir / "feature_columns.json", "w") as f:
        json.dump(feature_config, f, indent=4)

    decision = {
        "candidate_id": timestamp,
        "status": (
            "READY_FOR_PROMOTION"
            if quality_passed else "REJECTED"
        ),
        "quality_gate_passed": bool(quality_passed),
        "production_model_changed": False,
        "validation_rows": len(X_valid),
        "threshold": threshold,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "quality_rules": {
            "PR_AUC_not_below_baseline": (
                candidate_metrics["PR_AUC"]
                >= baseline_metrics["PR_AUC"]
            ),
            "Recall_not_below_baseline": (
                candidate_metrics["Recall"]
                >= baseline_metrics["Recall"]
            ),
            "F1_not_below_baseline": (
                candidate_metrics["F1"]
                >= baseline_metrics["F1"]
            ),
        },
    }

    with mlflow.start_run(
        run_name=f"IEEE-CIS-Candidate-{timestamp}"
    ) as run:

        decision["mlflow_run_id"] = run.info.run_id

        mlflow.log_params({
            "candidate_id": timestamp,
            "train_rows": len(X_train),
            "validation_rows": len(X_valid),
            "scale_pos_weight": scale_pos_weight,
            "threshold": threshold,
        })

        mlflow.log_metrics({
            f"baseline_{k}": v
            for k, v in baseline_metrics.items()
            if k != "Threshold"
        })

        mlflow.log_metrics({
            f"candidate_{k}": v
            for k, v in candidate_metrics.items()
            if k != "Threshold"
        })

        mlflow.log_metric(
            "quality_gate_passed",
            int(quality_passed),
        )

        with open(candidate_dir / "metrics.json", "w") as f:
            json.dump(decision, f, indent=4)

        mlflow.log_artifacts(str(candidate_dir))

    with open(candidate_dir / "promotion_decision.json", "w") as f:
        json.dump(decision, f, indent=4)

    print("\nCANDIDATE TRAINING RESULT")
    print(json.dumps(decision, indent=4))
    print("Candidate artifacts:", candidate_dir)
    print("Production model was NOT changed.")


if __name__ == "__main__":
    main()
