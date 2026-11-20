from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

TRIGGER_FILE = ROOT / "reports" / "live_retraining_trigger.json"

# Only NEW, confirmed-labelled transactions belong here.
LABELLED_DATA_FILE = (
    ROOT / "data" / "ieee_cis" / "incoming"
    / "labelled_transactions.csv"
)

OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = OUTPUT_DIR / "retraining_orchestrator_status.json"

MIN_LABELLED_ROWS = 1000


def save_status(result):
    result["checked_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("\nRETRAINING ORCHESTRATOR STATUS")
    print(json.dumps(result, indent=4))
    print("Saved:", STATUS_FILE)


def main():

    if not TRIGGER_FILE.exists():
        raise FileNotFoundError(
            f"Drift trigger not found: {TRIGGER_FILE}"
        )

    with open(TRIGGER_FILE, "r", encoding="utf-8") as f:
        trigger = json.load(f)

    drift_share = float(trigger["drift_share"])
    drift_threshold = float(trigger["drift_threshold"])

    result = {
        "dataset": "IEEE-CIS",
        "drift_share": drift_share,
        "drift_threshold": drift_threshold,
        "retraining_required": bool(
            trigger["retraining_required"]
        ),
        "labelled_data_available": False,
        "labelled_rows": 0,
        "candidate_training_allowed": False,
        "production_model_changed": False,
    }

    # State 1: No significant drift.
    if not trigger["retraining_required"]:
        result["status"] = "NO_ACTION"
        result["reason"] = "Drift is below configured threshold."
        save_status(result)
        return

    # State 2: Drift detected, but no new labelled data.
    if not LABELLED_DATA_FILE.exists():
        result["status"] = "WAITING_FOR_LABELS"
        result["reason"] = (
            "Drift detected, but no new labelled transaction file exists."
        )
        save_status(result)
        return

    data = pd.read_csv(LABELLED_DATA_FILE)

    if "isFraud" not in data.columns:
        result["status"] = "WAITING_FOR_LABELS"
        result["reason"] = "isFraud column is missing."
        save_status(result)
        return

    labels = pd.to_numeric(data["isFraud"], errors="coerce")

    if labels.isna().any():
        result["status"] = "WAITING_FOR_LABELS"
        result["reason"] = "Some transactions have missing or invalid labels."
        save_status(result)
        return

    if not labels.isin([0, 1]).all():
        result["status"] = "WAITING_FOR_LABELS"
        result["reason"] = "Labels must contain only 0 or 1."
        save_status(result)
        return

    if labels.nunique() < 2:
        result["status"] = "WAITING_FOR_LABELS"
        result["reason"] = "Both legitimate and fraud labels are required."
        save_status(result)
        return

    result["labelled_data_available"] = True
    result["labelled_rows"] = len(data)

    if len(data) < MIN_LABELLED_ROWS:
        result["status"] = "WAITING_FOR_MORE_LABELS"
        result["reason"] = (
            f"Need at least {MIN_LABELLED_ROWS} labelled rows."
        )
        save_status(result)
        return

    # State 3: Eligible for a separate candidate-training process.
    result["status"] = "READY_FOR_CANDIDATE"
    result["candidate_training_allowed"] = True
    result["reason"] = (
        "Drift detected and sufficient confirmed labels are available."
    )

        # Run candidate trainer only after eligibility checks pass.
    trainer_file = ROOT / "src" / "train_ieee_candidate.py"

    process = subprocess.run(
        [sys.executable, str(trainer_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    result["candidate_trainer_executed"] = True
    result["candidate_trainer_return_code"] = process.returncode
    result["candidate_trainer_output"] = process.stdout[-3000:]
    result["candidate_trainer_error"] = process.stderr[-3000:]

    if process.returncode == 0:
        result["status"] = "CANDIDATE_TRAINER_EXECUTED"
        result["reason"] = (
            "Candidate trainer completed. Check its promotion decision "
            "before any production deployment."
        )
    else:
        result["status"] = "CANDIDATE_TRAINER_FAILED"
        result["reason"] = (
            "Candidate trainer failed. Production model was not changed."
        )

    save_status(result)


if __name__ == "__main__":
    main()